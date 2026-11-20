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
python toy_pd_two_moons.py            # about 10 minutes on a 2-core CPU
python toy_pd_two_moons.py --seed 1   # other seeds
```

Outputs: a table in the terminal, `fig1_two_moons_pd.png` and `fig1_two_moons_pd.pdf`.

### Result (seed 0, PyTorch 2.15.0)

| One-step student | Loss | Target noise | Approximation error | Off-data distance | Coverage distance |
| --- | --- | --- | --- | --- | --- |
| Random grid | 0.0789 | 0.0777 | 0.0012 | 0.151 | 0.115 |
| Random grid, 4x training | 0.0790 | 0.0778 | 0.0012 | 0.151 | 0.120 |
| Fixed grid | 0.0001 | 0.0000 | 0.0001 | 0.068 | 0.020 |
| Teacher, 32 steps (reference) | - | - | - | 0.063 | 0.017 |

Target noise per round with the random grid (16, 8, 4, 2, 1 steps):
0.0001, 0.0004, 0.0018, 0.0107, 0.0777.

### Limitations

- One seed.
- The 32-step teacher is not sharp: it places samples between the two moons.
  The figure should be regenerated after the teacher is trained better.
- The toy differs from ELF's released PD code: two sub-steps per student step, a
  uniform interior time, noise scale 1, no guidance and no self-conditioning.
- The approximation error is close to zero in two dimensions. This need not hold
  for ELF's network.
