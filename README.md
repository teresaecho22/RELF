# relf

Toy experiments on progressive distillation (PD) for flow models that use the
x-prediction parameterization of Embedded Language Flows (ELF).

## `toy_pd_two_moons.py`

The script trains a two-dimensional x-prediction flow model on two-moons data and
distills it from 32 steps to 1 step in five PD rounds (32 → 16 → 8 → 4 → 2 → 1).
It runs the distillation twice:

- `random`: the teacher's interior sub-step time is sampled per example.
- `fixed`: the interior sub-step time is the midpoint, so the target is a
  deterministic function of the student's input.

For every round it reports the two terms of the regression loss on fresh inputs,

```
student loss = approximation error + target noise
```

where the target noise is estimated from 64 repeated teacher rollouts per input.
It also reports two sample-quality distances at the student's own step count:

- off-data distance: mean distance from a generated sample to its nearest data point.
- coverage distance: mean distance from a data point to its nearest generated sample.

### Run

```
pip install torch matplotlib
python toy_pd_two_moons.py            # about 9 minutes on a 2-core CPU
python toy_pd_two_moons.py --seed 1   # other seeds
```

Outputs: a table in the terminal, `fig1_two_moons_pd.png` and `fig1_two_moons_pd.pdf`.

### Result (seed 0, PyTorch 2.15.0)

![One-step samples after PD with a random and with a fixed teacher time grid](fig1_two_moons_pd.png)

| Model | Loss | Target noise | Approximation error | Off-data distance | Coverage distance |
| --- | --- | --- | --- | --- | --- |
| One step, random grid | 0.1036 | 0.1018 | 0.0018 | 0.119 | 0.103 |
| One step, random grid, 4x training | 0.1029 | 0.1012 | 0.0017 | 0.118 | 0.100 |
| One step, fixed grid | 0.0003 | 0.0000 | 0.0003 | 0.029 | 0.021 |
| Teacher, 32 steps (reference) | - | - | - | 0.017 | 0.016 |
| Fresh data against data (floor) | - | - | - | 0.012 | 0.013 |

Target noise per round with the random grid (16, 8, 4, 2, 1 steps):
0.0001, 0.0004, 0.0019, 0.0119, 0.1018.

With the random grid, 98% of the one-step loss is target noise, and four times more
training leaves the loss and both distances unchanged. With the fixed grid, the
target noise is zero and the one-step student is close to the 32-step teacher, but
not equal to it (off-data distance 0.029 against 0.017).

### Teacher training

The teacher samples the training time from a logit-normal distribution (mean 0,
std 1) and trains for 16,000 iterations. Two alternatives gave a worse 32-step
teacher in this toy (8,000 iterations each):

| Time sampling | Off-data distance | Coverage distance |
| --- | --- | --- |
| Uniform | 0.068 | 0.017 |
| Logit-normal, mean -1.5, std 0.8 (ELF's setting) | 0.152 | 0.144 |
| Logit-normal, mean 0, std 1 | 0.025 | 0.018 |

### Limitations

- One seed.
- The toy differs from ELF's released PD code: two sub-steps per student step, a
  uniform interior time, noise scale 1, no guidance, no self-conditioning, and a
  different time sampling for the teacher.
- The approximation error is close to zero in two dimensions. This need not hold
  for ELF's network.
