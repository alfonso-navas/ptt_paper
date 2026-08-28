[![DOI](https://zenodo.org/badge/1317704094.svg)](https://doi.org/10.5281/zenodo.21710665)
# ptt_paper

Reference implementation of **Parallel Tempering along the Trajectory (PTT)**, a sampling
scheme that reuses the training trajectory of a Restricted Boltzmann Machine to improve
sampling during training.

Instead of building a replica ladder over temperatures, PTT builds it over _training time_:
the replicas are checkpoints of the same model saved at different points along its own
training run. Configurations are swapped between consecutive checkpoints, so well-mixed
samples from the early (smooth, easy-to-sample) models are propagated up to the current
model, which would otherwise be too rough to equilibrate.

This code accompanies the paper [Equilibrium Training of Energy-Based Models with Parallel Trajectory Tempering](https://arxiv.org/abs/2607.27077) by Nicolas Béreux,
Aurélien Decelle, Cyril Furtlehner and Beatriz Seoane.

It is built on top of [`rbms`](https://github.com/DsysDML/rbms), which provides the RBM
models, datasets, optimizers and HDF5 I/O.

## Installation

```bash
git clone https://github.com/DsysDML/ptt_paper
cd ptt_paper
uv sync
```

Requires Python >= 3.12 and a CUDA-capable GPU for anything beyond toy runs.
The `ptt` command is installed as a console script.

## Training

A training run is launched with `ptt train`. The only required argument is the dataset:

```bash
ptt train -d data/MNIST.h5 -o models/mnist_ptt.h5
```

A more typical run, setting the model size, the optimizer and the PTT knobs explicitly:

```bash
ptt train -d data/MNIST.h5 -o models/mnist_ptt.h5 --num_hiddens 500 --num_chains 2000 --batch_size 2000 --num_updates 10000 --learning_rate 0.01 --optim adam --target_acc_rate 0.25 --increment 1 --num_swaps 1 --n_save 50 --device cuda
```

### Dataset

| Option                | Default    | Description                                                                                    |
| --------------------- | ---------- | ---------------------------------------------------------------------------------------------- |
| `-d`, `--dataset`     | _required_ | Path to the training data. `.h5` is read as an HDF5 dataset, anything else is parsed as FASTA. |
| `--test_dataset`      | `None`     | Separate test set. If omitted, the training file is split according to `--train_size`.         |
| `--train_size`        | `0.6`      | Fraction of the dataset used for training when no test set is given.                           |
| `--test_size`         | `None`     | Fraction used for testing.                                                                     |
| `--subset_labels`     | `None`     | Restrict training to these integer labels (e.g. `--subset_labels 0 1 8`).                      |
| `--alphabet`          | `protein`  | Token encoding for FASTA input: `protein`, `rna`, `dna`, or a custom token string.             |
| `--use_weights`       | off        | Compute per-sequence reweighting (for MSA data).                                               |
| `--remove_duplicates` | off        | Drop duplicate samples before splitting.                                                       |
| `--seed`              | random     | Seed for the train/test split. **Set this for reproducible runs.**                             |

### Model

| Option          | Default  | Description                                                                                                                                                                     |
| --------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--num_hiddens` | `100`    | Number of hidden units.                                                                                                                                                         |
| `--num_chains`  | `2000`   | Number of parallel Markov chains per replica. Also the default basis for the reservoir size.                                                                                    |
| `--model_type`  | inferred | One of `BBRBM`, `PBRBM`, `IIRBM`, `BGRBM`, `IGRBM`, `PBM`. If omitted, chosen from the variable type of the dataset (binary → `BBRBM`, categorical → `PBRBM`, Ising → `IIRBM`). |

### PTT algorithm

These are the options specific to this repository; the defaults correspond to the settings
used in the paper.

| Option                  | Default           | Description                                                                                                                                                                                                                               |
| ----------------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--target_acc_rate`     | `0.25`            | Target swap acceptance rate between adjacent replicas. This is the main knob: the ladder grows or shrinks to keep the acceptance near this value. Lower it to use fewer replicas (cheaper, riskier); raise it for a denser, safer ladder. |
| `--increment`           | `1`               | Number of Gibbs steps performed on each replica between two swap attempts.                                                                                                                                                                |
| `--num_swaps`           | `1`               | Number of swap steps performed between two gradient updates.                                                                                                                                                                              |
| `--max_n_model`         | `2`               | Maximum number of replicas held in the ladder before it is collapsed back down.                                                                                                                                                           |
| `--target_n_model`      | `2`               | Number of replicas kept after a collapse. The discarded lower replicas are replaced by a reservoir of their equilibrium samples.                                                                                                          |
| `--reservoir_size`      | `10 * num_chains` | Number of configurations stored in that reservoir.                                                                                                                                                                                        |
| `--n_sample_steps`      | `10`              | Sampling steps performed after inserting or removing a replica, to restore equilibrium.                                                                                                                                                   |
| `--num_steps_annealing` | `100`             | Sampling steps per replica when initialising the chains by annealing along the ladder.                                                                                                                                                    |
| `--full_sampler`        | off               | Keep the complete ladder (all checkpoints) to generate reservoirs, instead of only the truncated one. More faithful, substantially more memory.                                                                                           |

If the acceptance rate collapses below the minimum, the run raises an
`AcceptanceRateException`; `ptt train` catches it, rewinds to the last checkpoint with a
healthy acceptance rate, halves the learning rate and resumes automatically.

### Optimization

| Option             | Default | Description                                                              |
| ------------------ | ------- | ------------------------------------------------------------------------ |
| `--num_updates`    | `10000` | Number of gradient updates.                                              |
| `--batch_size`     | `2000`  | Minibatch size.                                                          |
| `--learning_rate`  | `0.01`  | Learning rate.                                                           |
| `--optim`          | `sgd`   | `sgd`, `adam`, or `cossim` (SGD with a cosine-similarity adaptive rate). |
| `--max_lr`         | `10`    | Learning-rate cap, used by `cossim`.                                     |
| `--scale_lr`       | off     | Scale the learning rate by `1/sqrt(number of variables)`.                |
| `--mult_optim`     | off     | Use a separate optimizer per parameter group.                            |
| `--training_type`  | `pcd`   | `pcd`, `cd` or `rdm`.                                                    |
| `--no_center`      | off     | Use the non-centered gradient.                                           |
| `--L1`, `--L2`     | `0.0`   | Regularization strengths.                                                |
| `--normalize_grad` | off     | Normalize the gradient before each update.                               |
| `--max_norm_grad`  | `-1`    | Clip the gradient norm (`-1` disables).                                  |

### Saving and hardware

| Option             | Default  | Description                                                                                                                            |
| ------------------ | -------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `-o`, `--filename` | `RBM.h5` | Output HDF5 archive. Also the input file when restoring.                                                                               |
| `--n_save`         | `50`     | Number of checkpoints saved over the run. These are the models PTT draws its replicas from, so this sets the resolution of the ladder. |
| `--spacing`        | `exp`    | Checkpoint spacing, `exp` or `linear`. `exp` is strongly recommended: the model changes fastest early in training.                     |
| `--device`         | `cuda`   | PyTorch device.                                                                                                                        |
| `--dtype`          | `float`  | `int`, `half`, `float` or `double`.                                                                                                    |

> **Note:** `--overwrite` is currently always on — an existing output file is replaced
> without confirmation. Pass a fresh `-o` path for each run.

## Restoring a training

```bash
ptt train -d data/MNIST.h5 -o models/mnist_ptt.h5 --restore
```

Add `--update N` to restart from a specific checkpoint rather than the last one.

On restore, the dataset split, gradient settings and PTT hyperparameters are read back from
the archive, so the corresponding command-line flags are ignored. The learning rate and the
number of updates can still be overridden.

## Output

The run produces a single HDF5 archive containing:

- `update_<n>/` — one group per checkpoint: model parameters, parallel chains, elapsed
  time, learning rate, estimated `log_z`, and a `flags` subgroup. Checkpoints used as PTT
  replicas carry the `ptt` flag.
- `sampler/` — the serialized PTT sampler, including the reservoir, so sampling can be
  resumed without retraining.
- `hyperparameters/`, `dataset_args/`, `train_args/`, `grad_args/`, `sampling_args/`,
  `save_args/`, `ptt_args/` — the full configuration of the run.
- `trwa/` — autocorrelation curve and the integrated and exponential autocorrelation times
  of the replica round-trip, once measured.

## Citation

@article{bereux2026equilibrium,
title={Equilibrium Training of Energy-Based Models with Parallel Trajectory Tempering},
author={B{\'e}reux, Nicolas and Decelle, Aur{\'e}lien and Furtlehner, Cyril and Seoane, Beatriz},
journal={arXiv preprint arXiv:2607.27077},
year={2026}
}

## License

MIT. See [LICENSE](LICENSE).
