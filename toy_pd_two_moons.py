"""Toy version of ELF's progressive distillation (PD) on two-moons data.

The procedure follows ELF's released PD code (distillation branch,
scripts/progressive_distillation.sh and src/distill_train_step.py):

  * The teacher is the ORIGINAL flow model in every round.
  * Round k trains a student with N_k steps (16, 8, 4, 2, 1). The student takes one
    step of size h = 1 / N_k from z_{t_s}; the teacher takes T_k Euler sub-steps
    (4, 8, 16, 32, 64) over the same interval, so h * T_k = 1/64 in every round.
  * Round 1 starts from the teacher's weights, later rounds from the previous student.
  * t_s = u * (1 - h) with u logit-normal; the teacher's interior sub-step times are
    sorted logit-normal samples rescaled into [t_s, t_s + h], drawn per example.

The script runs the curriculum twice:
  grid = "random": interior sub-step times sampled per example (as in ELF's code)
  grid = "fixed" : interior sub-step times fixed to the quantiles of the same
                   logit-normal distribution, so the target is a deterministic
                   function of the student's input
and reports, for every round, the two terms of the regression loss on fresh inputs:
  student loss = approximation error + target noise.

Run    : python toy_pd_two_moons.py
Output : fig1_two_moons_pd.png / .pdf and a table printed to the terminal.

Differences from ELF: two dimensions, an MLP, no guidance, no self-conditioning,
no decoding step, and deterministic Euler sampling for the final one-step student.
"""
import argparse
import copy
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

S = 2.0                                   # noise scale of ELF's denoising branch
T_EPS = 0.05                              # lower bound on (1 - t), ELF's t_eps
P_MEAN, P_STD = -1.5, 0.8                 # ELF's logit-normal time distribution
STUDENT_STEPS = (16, 8, 4, 2, 1)
TEACHER_SUBSTEPS = (4, 8, 16, 32, 64)     # step size x sub-steps = 1/64 in every round


# --------------------------------data---------------------------------------------
def two_moons(n, noise=0.05):
    """n points from the two-moons distribution, centered and scaled to std ~ 1."""
    n_up = n // 2
    a_up = torch.rand(n_up) * math.pi
    a_lo = torch.rand(n - n_up) * math.pi
    upper = torch.stack([torch.cos(a_up), torch.sin(a_up)], dim=1)
    lower = torch.stack([1 - torch.cos(a_lo), 0.5 - torch.sin(a_lo)], dim=1)
    x = torch.cat([upper, lower]) + noise * torch.randn(n, 2)
    return (x - torch.tensor([0.5, 0.25])) / 0.7


def logit_normal(*shape):
    """t = sigmoid(N(P_MEAN, P_STD)), ELF's time distribution."""
    return torch.sigmoid(torch.randn(*shape) * P_STD + P_MEAN)


# --------------------------------model--------------------------------------------
class Net(nn.Module):
    """x-prediction network: (z_t, t) -> x_hat."""

    def __init__(self, width=256, n_freq=4):
        super().__init__()
        self.register_buffer("freq", math.pi * 2.0 ** torch.arange(n_freq))
        self.net = nn.Sequential(
            nn.Linear(2 + 1 + 2 * n_freq, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 2),
        )

    def forward(self, z, t):                      # z: (B, 2), t: (B, 1)
        ang = t * self.freq                       # (B, n_freq)
        return self.net(torch.cat([z, t, ang.sin(), ang.cos()], dim=1))


def velocity(net, z, t):
    """v = (x_hat - z) / (1 - t), with (1 - t) clamped at T_EPS."""
    return (net(z, t) - z) / (1 - t).clamp(min=T_EPS)


# ---------------------------teacher training---------------------------------------
def train_teacher(net, iters, bs, lr):
    """Flow matching loss of Eq. (1): || v_theta(z_t, t) - (x - s eps) ||^2."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for _ in range(iters):
        x = two_moons(bs)
        t = logit_normal(bs, 1)
        eps = torch.randn_like(x)
        z = t * x + (1 - t) * S * eps
        loss = (velocity(net, z, t) - (x - S * eps)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net


# -----------------------distillation-----------------------------------------------
def sub_step_times(t_s, h, n_sub, grid):
    """(B, n_sub + 1) times from t_s to t_s + h.

    random: interior points are sorted logit-normal samples, drawn per example.
    fixed : interior points are the quantiles i / n_sub of the same distribution.
    """
    b = t_s.shape[0]
    if grid == "random":
        u = torch.sort(logit_normal(b, n_sub - 1), dim=1).values
    else:
        q = torch.arange(1, n_sub) / n_sub
        u = torch.sigmoid(P_MEAN + P_STD * torch.special.ndtri(q)).expand(b, n_sub - 1)
    return torch.cat([t_s, t_s + h * u, t_s + h], dim=1)


@torch.no_grad()
def pd_target(teacher, z, t_s, h, n_sub, grid):
    """n_sub deterministic Euler sub-steps of the teacher over [t_s, t_s + h], then Eq. (2)."""
    times = sub_step_times(t_s, h, n_sub, grid)
    z_cur = z
    for i in range(n_sub):
        t_cur, t_next = times[:, i:i + 1], times[:, i + 1:i + 2]
        z_cur = z_cur + (t_next - t_cur) * velocity(teacher, z_cur, t_cur)
    return z + (1 - t_s) / h * (z_cur - z)        # x-target that reaches z_cur in one step


def student_input(n, h):
    """Forward-noised data at t_s = u * (1 - h), u logit-normal."""
    x = two_moons(n)
    t_s = logit_normal(n, 1) * max(1 - h, 1e-6)
    z = t_s * x + (1 - t_s) * S * torch.randn_like(x)
    return z, t_s


def pd_round(student, teacher, n_student, n_sub, grid, iters, bs, lr):
    """One PD round: regress the student's x-prediction onto the teacher's PD target."""
    h = 1.0 / n_student
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for _ in range(iters):
        z, t_s = student_input(bs, h)
        target = pd_target(teacher, z, t_s, h, n_sub, grid)
        loss = (student(z, t_s) - target).pow(2).sum(dim=1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return student


# ----------------------measurements------------------------------------------------
@torch.no_grad()
def loss_terms(student, teacher, n_student, n_sub, grid, n=2048, k=32):
    """Eq. (5) on fresh inputs: loss = approximation error + target noise.

    The teacher rollout is repeated k times from the same input.
    """
    h = 1.0 / n_student
    z, t_s = student_input(n, h)
    targets = torch.stack([pd_target(teacher, z, t_s, h, n_sub, grid) for _ in range(k)])
    pred = student(z, t_s)
    loss = (pred - targets).pow(2).sum(dim=-1).mean().item()
    approx = (pred - targets.mean(dim=0)).pow(2).sum(dim=-1).mean().item()
    noise = targets.var(dim=0, unbiased=False).sum(dim=-1).mean().item()
    return loss, approx, noise


@torch.no_grad()
def nn_dist(a, b):
    """Mean distance from each point of a to its nearest point of b."""
    return torch.cdist(a, b).min(dim=1).values.mean().item()


def quality(samples, data):
    """(off-data distance, coverage distance).

    off-data : generated -> nearest data point  (high = samples off the data)
    coverage : data -> nearest generated sample (high = data regions missed)
    """
    return nn_dist(samples, data), nn_dist(data, samples)


@torch.no_grad()
def teacher_samples(teacher, n, grid):
    """64 Euler sub-steps from noise: this is the target of the one-step student."""
    z0 = S * torch.randn(n, 2)
    return pd_target(teacher, z0, torch.zeros(n, 1), 1.0, TEACHER_SUBSTEPS[-1], grid)


@torch.no_grad()
def one_step_samples(student, n):
    z0 = S * torch.randn(n, 2)
    return student(z0, torch.zeros(n, 1))


# ---------------------------main---------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--teacher_iters", type=int, default=16000)
    p.add_argument("--pd_iters", type=int, default=2000)
    p.add_argument("--long", type=int, default=4, help="longer-training factor for the last round")
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--n_eval", type=int, default=5000)
    p.add_argument("--grid", default="both", choices=["both", "random", "fixed", "none"])
    p.add_argument("--ckpt_dir", default="", help="save / reuse teacher and one-step students here")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    def ckpt(name):
        return os.path.join(args.ckpt_dir, f"{name}_seed{args.seed}.pt") if args.ckpt_dir else ""

    data = two_moons(args.n_eval)
    floor = quality(two_moons(args.n_eval), data)
    print(f"data vs data               off-data {floor[0]:.3f}  coverage {floor[1]:.3f}", flush=True)

    base = Net()
    if ckpt("teacher") and os.path.exists(ckpt("teacher")):
        base.load_state_dict(torch.load(ckpt("teacher")))
    else:
        train_teacher(base, args.teacher_iters, 1024, lr=1e-3)
        if ckpt("teacher"):
            torch.save(base.state_dict(), ckpt("teacher"))
    base.eval()
    for grid in ("random", "fixed"):
        q = quality(teacher_samples(base, args.n_eval, grid), data)
        print(f"teacher, 64 sub-steps, {grid:6s} grid  off-data {q[0]:.3f}  coverage {q[1]:.3f}", flush=True)
    for n_student, n_sub in zip(STUDENT_STEPS, TEACHER_SUBSTEPS):
        _, _, noise = loss_terms(base, base, n_student, n_sub, "random")
        print(f"target noise, random grid, {n_student:2d}-step round: {noise:.5f}", flush=True)

    grids = ("random", "fixed") if args.grid == "both" else (() if args.grid == "none" else (args.grid,))
    trained = {}
    for grid in grids:
        print(f"\nPD with a {grid} teacher time grid")
        print("steps   loss     approx   noise", flush=True)
        student = copy.deepcopy(base)                     # round 1 starts from the teacher
        for n_student, n_sub in zip(STUDENT_STEPS, TEACHER_SUBSTEPS):
            before_last = copy.deepcopy(student)
            student.train()
            pd_round(student, base, n_student, n_sub, grid, args.pd_iters, args.bs, lr=3e-4)
            student.eval()
            loss, approx, noise = loss_terms(student, base, n_student, n_sub, grid)
            print(f"{n_student:5d}   {loss:.5f}  {approx:.5f}  {noise:.5f}", flush=True)
        q = quality(one_step_samples(student, args.n_eval), data)
        print(f"one-step student           off-data {q[0]:.3f}  coverage {q[1]:.3f}", flush=True)
        trained[grid] = student
        if ckpt("teacher"):
            torch.save(student.state_dict(), ckpt(f"student_{grid}"))

        if grid == "random" and args.long > 1:           # longer training of the last round
            long = before_last
            long.train()
            pd_round(long, base, 1, TEACHER_SUBSTEPS[-1], grid, args.long * args.pd_iters, args.bs, lr=3e-4)
            long.eval()
            loss, approx, noise = loss_terms(long, base, 1, TEACHER_SUBSTEPS[-1], grid)
            q = quality(one_step_samples(long, args.n_eval), data)
            print(f"1 (x{args.long})  {loss:.5f}  {approx:.5f}  {noise:.5f}   off-data {q[0]:.3f}  coverage {q[1]:.3f}",
                  flush=True)

    # ----------------------Fig. 1--------------------------------------------
    students = dict(trained)
    for grid in ("random", "fixed"):
        if grid not in students and ckpt("teacher") and os.path.exists(ckpt(f"student_{grid}")):
            net = Net()
            net.load_state_dict(torch.load(ckpt(f"student_{grid}")))
            students[grid] = net.eval()
    if len(students) == 2:
        panels = [
            ("Data", data),
            ("Teacher, random grid", teacher_samples(base, args.n_eval, "random")),
            ("One step, random grid", one_step_samples(students["random"], args.n_eval)),
            ("Teacher, fixed grid", teacher_samples(base, args.n_eval, "fixed")),
            ("One step, fixed grid", one_step_samples(students["fixed"], args.n_eval)),
        ]
        fig, axes = plt.subplots(1, 5, figsize=(12.5, 2.6), sharex=True, sharey=True)
        for ax, (title, pts) in zip(axes, panels):
            ax.scatter(pts[:, 0], pts[:, 1], s=0.5, alpha=0.5, color="#1f4e79", linewidths=0)
            ax.set_title(title, fontsize=9)
            ax.set_xlim(-2.6, 2.6)
            ax.set_ylim(-1.8, 1.8)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig("fig1_two_moons_pd.png", dpi=300)
        fig.savefig("fig1_two_moons_pd.pdf")
        print("\nsaved fig1_two_moons_pd.png / .pdf")


if __name__ == "__main__":
    main()
