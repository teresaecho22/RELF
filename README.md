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

### Result (seeds 0, 1, 2; mean ± standard error)

![One-step samples after PD with a random and with a fixed teacher time grid](fig1_two_moons_pd.png)

The figure shows seed 0.

| Model | Loss | Target noise | Approximation error | Off-data distance | Coverage distance |
| --- | --- | --- | --- | --- | --- |
| One step, random grid | 0.0988 ± 0.0024 | 0.0971 ± 0.0023 | 0.0017 ± 0.0001 | 0.118 ± 0.003 | 0.098 ± 0.003 |
| One step, random grid, 4x training | 0.0980 ± 0.0025 | 0.0964 ± 0.0024 | 0.0016 ± 0.0001 | 0.116 ± 0.003 | 0.096 ± 0.003 |
| One step, fixed grid | 0.0002 ± 0.0001 | 0.0000 | 0.0002 ± 0.0001 | 0.028 ± 0.001 | 0.020 ± 0.001 |
| Teacher, 32 steps (reference) | - | - | - | 0.017 | 0.016 |
| Fresh data against data (floor) | - | - | - | 0.012 | 0.013 |

The standard error of the last two rows is below 0.001.

Mean target noise per round with the random grid (16, 8, 4, 2, 1 steps):
0.0001, 0.0004, 0.0019, 0.0120, 0.0971.

With the random grid, 98% of the one-step loss is target noise in every seed, and
four times more training leaves the loss and both distances within one standard
error. With the fixed grid, the target noise is zero and the one-step student is
close to the 32-step teacher, but not equal to it (off-data distance 0.028 against
0.017).

One-step values per seed:

| Seed | Random: loss | Random: target noise | Random: off-data | Fixed: loss | Fixed: off-data | Teacher: off-data |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.1036 | 0.1018 | 0.119 | 0.0003 | 0.029 | 0.017 |
| 1 | 0.0962 | 0.0946 | 0.112 | 0.0003 | 0.026 | 0.017 |
| 2 | 0.0966 | 0.0950 | 0.122 | 0.0001 | 0.030 | 0.018 |

Seed 0 ran on Linux with PyTorch 2.15.0. Seeds 1 and 2 ran on macOS.

### Teacher training

The teacher samples the training time from a logit-normal distribution (mean 0,
std 1) and trains for 16,000 iterations. Two alternatives gave a worse 32-step
teacher in this toy (8,000 iterations each, seed 0):

| Time sampling | Off-data distance | Coverage distance |
| --- | --- | --- |
| Uniform | 0.068 | 0.017 |
| Logit-normal, mean -1.5, std 0.8 (ELF's setting) | 0.152 | 0.144 |
| Logit-normal, mean 0, std 1 | 0.025 | 0.018 |

### Limitations

- Three seeds, run on two machines.
- The toy differs from ELF's released PD code: two sub-steps per student step, a
  uniform interior time, noise scale 1, no guidance, no self-conditioning, and a
  different time sampling for the teacher.
- The approximation error is close to zero in two dimensions. This need not hold
  for ELF's network.
