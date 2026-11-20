"""Toy check for the RELF proposal (Fig. 1).

Data   : two moons (2-D).
Model  : x-prediction flow model with ELF's parameterization (Eq. 1 of the proposal).
PD     : progressive distillation 32 -> 16 -> 8 -> 4 -> 2 -> 1 steps (Eq. 2), run twice:
           grid = "random": the teacher's interior sub-step time is sampled per example
                            (this is what ELF's released distill_train_step does)
           grid = "fixed" : the interior sub-step time is the midpoint, so the target
                            is a deterministic function of the student's input
Report : for every round, the two terms of Eq. (5):
           student loss = approximation error + target noise
         and two sample-quality distances at the student's own step count.

Run    : python toy_pd_two_moons.py            (about 10 minutes on a 2-core CPU)
         python toy_pd_two_moons.py --seed 1   (repeat with other seeds)
Output : fig1_two_moons_pd.png / .pdf and a table printed to the terminal.
"""
import argparse
import copy
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

S = 1.0        # noise scale s. ELF uses s = 2 in the denoising branch; 1 is enough in 2-D.
T_EPS = 0.05   # lower bound on (1 - t) in the velocity, as in ELF's code (t_eps)


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
        t = torch.rand(bs, 1)
        eps = torch.randn_like(x)
        z = t * x + (1 - t) * S * eps
        loss = (velocity(net, z, t) - (x - S * eps)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net


@torch.no_grad()
def sample(net, n, steps):
    """Deterministic Euler sampler on a uniform grid, from t = 0 (noise) to t = 1 (data)."""
    z = S * torch.randn(n, 2)
    ts = torch.linspace(0, 1, steps + 1)
    for i in range(steps):
        t = ts[i].expand(n, 1)
        z = z + (ts[i + 1] - ts[i]) * velocity(net, z, t)
    return z


# -----------------------distillation----------------------------------------------- 
@torch.no_grad()
def pd_target(teacher, z, t_s, h, grid):
    """Two deterministic Euler sub-steps of the teacher over [t_s, t_s + h], then Eq. (2).

    grid = "fixed"  : interior time is the midpoint.
    grid = "random" : interior time is uniform in (t_s, t_s + h), sampled per example.
    """
    u = torch.full_like(t_s, 0.5) if grid == "fixed" else torch.rand_like(t_s)
    t_mid = t_s + u * h
    z_mid = z + (t_mid - t_s) * velocity(teacher, z, t_s)
    z_end = z_mid + (t_s + h - t_mid) * velocity(teacher, z_mid, t_mid)
    return z + (1 - t_s) / h * (z_end - z)        # x-target that reaches z_end in one step


def student_input(n, h):
    """Forward-noised data at a random start time t_s in [0, 1 - h]."""
    x = two_moons(n)
    t_s = torch.rand(n, 1) * (1 - h)
    z = t_s * x + (1 - t_s) * S * torch.randn_like(x)
    return z, t_s


def pd_round(teacher, n_student, grid, iters, bs, lr):
    """One PD round. The student starts from the teacher's weights."""
    student = copy.deepcopy(teacher)
    teacher.eval()
    h = 1.0 / n_student
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for _ in range(iters):
        z, t_s = student_input(bs, h)
        target = pd_target(teacher, z, t_s, h, grid)
        loss = (student(z, t_s) - target).pow(2).sum(dim=1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return student


# ----------------------measurements------------------------------------------------ 
@torch.no_grad()
def loss_terms(student, teacher, n_student, grid, n=4096, k=64):
    """Eq. (5) on fresh inputs: loss = approximation error + target noise.

    The teacher rollout is repeated k times from the same input.
    """
    h = 1.0 / n_student
    z, t_s = student_input(n, h)
    targets = torch.stack([pd_target(teacher, z, t_s, h, grid) for _ in range(k)])  # (k, n, 2)
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

    off-data : generated -> nearest data point  (high = samples off the data, low precision)
    coverage : data -> nearest generated sample (high = data regions missed, low recall)
    """
    return nn_dist(samples, data), nn_dist(data, samples)


# ---------------------------main--------------------------------------------------- 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--teacher_iters", type=int, default=8000)
    p.add_argument("--pd_iters", type=int, default=3000)
    p.add_argument("--long", type=int, default=4, help="longer-training factor for the last round")
    p.add_argument("--bs", type=int, default=1024)
    p.add_argument("--n_eval", type=int, default=5000)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    data = two_moons(args.n_eval)
    floor = quality(two_moons(args.n_eval), data)
    print(f"data vs data          off-data {floor[0]:.3f}  coverage {floor[1]:.3f}")

    base = train_teacher(Net(), args.teacher_iters, args.bs, lr=1e-3)
    q = quality(sample(base, args.n_eval, 32), data)
    print(f"teacher, 32 steps     off-data {q[0]:.3f}  coverage {q[1]:.3f}")
    q = quality(sample(base, args.n_eval, 1), data)
    print(f"teacher,  1 step      off-data {q[0]:.3f}  coverage {q[1]:.3f}")

    one_step = {}
    for grid in ("random", "fixed"):
        print(f"\nPD with a {grid} teacher time grid")
        print("steps   loss    approx  noise   off-data  coverage")
        teacher = base
        for n_student in (16, 8, 4, 2, 1):
            student = pd_round(teacher, n_student, grid, args.pd_iters, args.bs, lr=3e-4)
            loss, approx, noise = loss_terms(student, teacher, n_student, grid)
            q = quality(sample(student, args.n_eval, n_student), data)
            print(f"{n_student:5d}   {loss:.4f}  {approx:.4f}  {noise:.4f}  {q[0]:.3f}     {q[1]:.3f}")
            last_teacher, teacher = teacher, student
        one_step[grid] = student

        # longer training of the last round, same teacher
        student_long = pd_round(last_teacher, 1, grid, args.long * args.pd_iters, args.bs, lr=3e-4)
        loss, approx, noise = loss_terms(student_long, last_teacher, 1, grid)
        q = quality(sample(student_long, args.n_eval, 1), data)
        print(f"1 (x{args.long})  {loss:.4f}  {approx:.4f}  {noise:.4f}  {q[0]:.3f}     {q[1]:.3f}")

    # ----------------------Fig. 1-------------------------------------------- 
    panels = [
        ("Data", data),
        ("Teacher, 32 steps", sample(base, args.n_eval, 32)),
        ("One step, random grid", sample(one_step["random"], args.n_eval, 1)),
        ("One step, fixed grid", sample(one_step["fixed"], args.n_eval, 1)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.6), sharex=True, sharey=True)
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
