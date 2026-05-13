# PIM Launch Configs

`scripts/launch.py` builds a Slurm job from layered launch config:

1. `launch/defaults.yaml`
2. `launch/sites/<site>.yaml`
3. optional `launch/runs/<recipe>.yaml`
4. CLI overrides

Python training configs remain the source of truth for model/training behavior.
Launch YAML should only describe execution: site resources, container/runtime,
paths, checkpoint/resume choices, run naming, and explicit CLI `--options`.

## Direct Config Mode

Use this when you just made a normal Python config and want to send it with the
site defaults.

```bash
scripts/launch.py submit --site s3df \
  --config-dir panda/pretrain_geometry_combos \
  --config pretrain-sonata-v1m1-pilarnet-e050-head512-tail-wd20
```

Dry-run the generated Slurm script first:

```bash
scripts/launch.py dry-run --site s3df \
  --config-dir panda/pretrain_geometry_combos \
  --config pretrain-sonata-v1m1-pilarnet-e050-head512-tail-wd20
```

## Recipe Mode

Use a recipe when the launch itself has meaningful state: special resources,
checkpoint weights, resume behavior, W&B naming, or config overrides.

```bash
scripts/launch.py submit --site s3df launch/runs/e050_tail.yaml
scripts/launch.py submit --site nersc launch/runs/e050_tail.yaml
```

Changing `--site` swaps only the site overlay. The recipe should not need to
know whether it is running on S3DF or NERSC.

## Submission Behavior

For `--site s3df`, actual submission follows the repo convention from
`CLAUDE.md`: the launcher SSHes to `iana`, changes into the shared repo path,
activates the `pointcept-torch2.5.0-cu12.4` mamba environment, and runs `sbatch`
with the generated script.

For `--site nersc`, the launcher currently assumes it is being run on a NERSC
login node and submits locally with `sbatch`. The rendered job uses Shifter and
Perlmutter-style Slurm options from `launch/sites/nersc.yaml`.

## Overrides

Resource/site overrides:

```bash
scripts/launch.py submit --site s3df launch/runs/e050_tail.yaml \
  --account neutrino:ml-dev \
  --partition ampere \
  --set resources.time=00:30:00
```

Training `--options` overrides:

```bash
scripts/launch.py submit --site s3df launch/runs/e050_tail.yaml \
  --option epoch=1 \
  --option data.train.max_len=1000
```

Render without submitting:

```bash
scripts/launch.py submit --dry-run --site s3df launch/runs/e050_tail.yaml
```

## File Ownership

- `launch/defaults.yaml`: common launcher defaults.
- `launch/sites/s3df.yaml`: S3DF paths, account/partition, Singularity, `iana`
  submit behavior, and S3DF environment variables.
- `launch/sites/nersc.yaml`: NERSC paths, account/qos/constraint, Shifter, and
  Perlmutter environment variables.
- `launch/runs/*.yaml`: optional named launch recipes. 