#!/usr/bin/env python3
"""Compose site/run launch configs and submit PIM Slurm jobs."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
LAUNCH_DIR = ROOT / "launch"
PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_.]*)\}")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing launch config: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Launch config must be a mapping: {path}")
    return data


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def set_path(cfg: dict[str, Any], dotted_path: str, value: Any) -> None:
    if dotted_path.startswith("train.options."):
        cfg.setdefault("train", {}).setdefault("options", {})[
            dotted_path[len("train.options.") :]
        ] = value
        return

    cur: dict[str, Any] = cfg
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        next_value = cur.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise SystemExit(f"Cannot set {dotted_path}: {part} is not a mapping")
        cur = next_value
    cur[parts[-1]] = value


def flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out[path] = value
            out.update(flatten(value, path))
    return out


def format_string(value: str, context: dict[str, Any]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            raise SystemExit(f"Unknown placeholder {{{key}}} in launch config")
        return str(context[key])

    return PLACEHOLDER_RE.sub(repl, value)


def resolve_placeholders(data: Any, context: dict[str, Any]) -> Any:
    if isinstance(data, str):
        return format_string(data, context)
    if isinstance(data, list):
        return [resolve_placeholders(item, context) for item in data]
    if isinstance(data, dict):
        return {
            key: resolve_placeholders(value, context)
            for key, value in data.items()
        }
    return data


def resolve_all(cfg: dict[str, Any], timestamp: str) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg["timestamp"] = timestamp
    for _ in range(6):
        context = flatten(cfg)
        for key, value in cfg.get("paths", {}).items():
            context[key] = value
        context["repo_root"] = cfg.get("paths", {}).get("repo_root", "")
        new_cfg = resolve_placeholders(cfg, context)
        if new_cfg == cfg:
            return new_cfg
        cfg = new_cfg
    return cfg


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def option_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "None"
    return str(value)


def shell_join(parts: list[Any]) -> str:
    return shlex.join(str(part) for part in parts if part is not None and part != "")


def build_run_name(cfg: dict[str, Any], timestamp: str) -> str | None:
    run_cfg = cfg.get("run", {})
    train_cfg = cfg.get("train", {})
    name = run_cfg.get("name") or train_cfg.get("config")
    if not name:
        return None
    name = str(name)
    if as_bool(run_cfg.get("timestamp", True)) and timestamp not in name:
        name = f"{name}-{timestamp}"
    return name


def build_train_command(cfg: dict[str, Any], run_name: str | None) -> str:
    train_cfg = cfg.get("train", {})
    resources = cfg.get("resources", {})
    paths = cfg.get("paths", {})

    config_dir = train_cfg.get("config_dir")
    config = train_cfg.get("config")
    if not config_dir or not config:
        raise SystemExit("Need train.config_dir and train.config")

    repo_root = paths.get("repo_root", str(ROOT))
    train_path = train_cfg.get("train_path") or f"{repo_root}/scripts/train.sh"
    parts: list[Any] = [
        "sh",
        train_path,
        "-m",
        resources.get("nodes", 1),
        "-g",
        resources.get("gpus_per_node", 4),
        "-d",
        config_dir,
        "-c",
        config,
    ]

    if run_name:
        parts += ["-n", run_name]

    wandb_name = cfg.get("run", {}).get("wandb_name")
    if wandb_name is None:
        wandb_name = run_name
    if wandb_name:
        parts += ["-a", wandb_name]

    if train_cfg.get("weight"):
        parts += ["-w", train_cfg["weight"]]
    if as_bool(train_cfg.get("resume", False)):
        parts += ["-r", "true"]
    if as_bool(train_cfg.get("dev_mode", False)):
        parts += ["-C"]

    options = train_cfg.get("options") or {}
    if options:
        parts += ["--", "--options"]
        for key, value in options.items():
            if value is None:
                continue
            parts.append(f"{key}={option_value(value)}")

    return shell_join(parts)


def build_container_command(cfg: dict[str, Any], train_cmd: str) -> str:
    container = cfg.get("container", {})
    runtime = container.get("runtime")
    setup = container.get("setup") or []
    inner_cmd = " && ".join([*setup, train_cmd])

    if runtime == "singularity":
        parts: list[Any] = ["srun", "singularity", "run", "--nv"]
        binds = container.get("binds") or []
        if binds:
            parts += ["-B", ",".join(str(bind) for bind in binds)]
        parts += [container["image"], "bash", "-lc", inner_cmd]
        return shell_join(parts)

    if runtime == "shifter":
        parts = ["env"]
        for var in container.get("unset_env") or []:
            parts += ["-u", var]
        parts += ["srun", "shifter"]
        if container.get("module"):
            parts.append(f"--module={container['module']}")
        if container.get("image"):
            parts.append(f"--image={container['image']}")
        parts += ["/bin/bash", "-lc", inner_cmd]
        return shell_join(parts)

    if runtime in {None, "none"}:
        return train_cmd

    raise SystemExit(f"Unsupported container.runtime: {runtime}")


def slurm_directives(cfg: dict[str, Any], job_name: str) -> list[str]:
    resources = cfg.get("resources", {})
    slurm = cfg.get("slurm", {})

    directives = [
        ("job-name", job_name),
        ("nodes", resources.get("nodes")),
        ("ntasks-per-node", resources.get("tasks_per_node")),
        ("cpus-per-task", resources.get("cpus_per_task")),
        ("mem", resources.get("mem")),
        ("time", resources.get("time")),
        ("account", slurm.get("account")),
        ("partition", slurm.get("partition")),
        ("qos", slurm.get("qos")),
        ("constraint", slurm.get("constraint")),
        ("image", slurm.get("image")),
        ("module", slurm.get("module")),
        ("output", slurm.get("output")),
    ]

    gpu_directive = slurm.get("gpu_directive", "gres")
    gpus_per_node = resources.get("gpus_per_node")
    if gpu_directive == "gres":
        directives.insert(3, ("gres", f"gpu:{gpus_per_node}"))
    elif gpu_directive == "gpus-per-node":
        directives.insert(3, ("gpus-per-node", gpus_per_node))
    else:
        raise SystemExit(f"Unsupported slurm.gpu_directive: {gpu_directive}")

    return [
        f"#SBATCH --{key}={value}"
        for key, value in directives
        if value is not None and value != ""
    ]


def build_slurm_job_name(cfg: dict[str, Any], run_name: str) -> str:
    name = str(cfg.get("slurm", {}).get("job_name") or run_name)
    name = re.sub(r"[^A-Za-z0-9_.+-]+", "-", name).strip("-")
    return name[:128] or "pimm"


def render_script(cfg: dict[str, Any], run_name: str, train_cmd: str) -> str:
    repo_root = cfg.get("paths", {}).get("repo_root", str(ROOT))
    container_cmd = build_container_command(cfg, train_cmd)
    job_name = build_slurm_job_name(cfg, run_name)
    env = cfg.get("env") or {}
    exports = [
        f"export {key}={shlex.quote(str(value))}"
        for key, value in env.items()
        if value is not None
    ]

    lines = [
        "#!/bin/bash",
        *slurm_directives(cfg, job_name),
        "",
        "set -euo pipefail",
        "",
        "mkdir -p slurm_logs",
        *exports,
        f"cd {shlex.quote(str(repo_root))}",
        "",
        container_cmd,
        "",
    ]
    return "\n".join(lines)


def apply_cli(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cli: dict[str, Any] = {"train": {}, "resources": {}, "slurm": {}, "run": {}}

    for key in ("config_dir", "config", "weight"):
        value = getattr(args, key)
        if value is not None:
            cli["train"][key] = value
    if args.resume:
        cli["train"]["resume"] = True
    if args.dev_mode:
        cli["train"]["dev_mode"] = True
    if args.name:
        cli["run"]["name"] = args.name
    if args.wandb_name:
        cli["run"]["wandb_name"] = args.wandb_name
    if args.no_timestamp:
        cli["run"]["timestamp"] = False

    for key in ("nodes", "gpus_per_node", "tasks_per_node", "cpus_per_task", "mem", "time"):
        value = getattr(args, key)
        if value is not None:
            cli["resources"][key] = value

    for key in ("account", "partition", "qos", "constraint"):
        value = getattr(args, key)
        if value is not None:
            cli["slurm"][key] = value

    cfg = merge_dicts(cfg, {k: v for k, v in cli.items() if v})

    for item in args.option or []:
        if "=" not in item:
            raise SystemExit(f"--option must be KEY=VALUE, got: {item}")
        key, raw_value = item.split("=", 1)
        cfg.setdefault("train", {}).setdefault("options", {})[key] = parse_value(raw_value)

    for item in args.set or []:
        if "=" not in item:
            raise SystemExit(f"--set must be PATH=VALUE, got: {item}")
        key, raw_value = item.split("=", 1)
        set_path(cfg, key, parse_value(raw_value))

    return cfg


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_yaml(LAUNCH_DIR / "defaults.yaml")

    recipe_cfg: dict[str, Any] = {}
    if args.recipe:
        recipe_path = Path(args.recipe)
        if not recipe_path.is_absolute():
            recipe_path = ROOT / recipe_path
        recipe_cfg = load_yaml(recipe_path)

    site = args.site or recipe_cfg.get("site")
    if not site:
        raise SystemExit("Need --site or a recipe-level site")

    site_cfg = load_yaml(LAUNCH_DIR / "sites" / f"{site}.yaml")
    cfg = merge_dicts(cfg, site_cfg)
    cfg = merge_dicts(cfg, recipe_cfg)
    cfg = apply_cli(cfg, args)

    if not args.recipe and (
        not cfg.get("train", {}).get("config_dir")
        or not cfg.get("train", {}).get("config")
    ):
        raise SystemExit("Direct config mode needs --config-dir and --config")

    return cfg


def validate_local_config(cfg: dict[str, Any]) -> None:
    train_cfg = cfg.get("train", {})
    config_dir = train_cfg.get("config_dir")
    config = train_cfg.get("config")
    if not config_dir or not config:
        return
    path = ROOT / "configs" / str(config_dir) / f"{config}.py"
    if not path.exists():
        print(f"warning: local config not found: {path}", file=sys.stderr)


def submit(script: str, cfg: dict[str, Any]) -> None:
    repo_root = Path(str(cfg.get("paths", {}).get("repo_root", ROOT)))
    submit_cfg = cfg.get("submit") or {}
    submit_cwd_value = str(submit_cfg.get("cwd") or repo_root)

    if submit_cfg.get("host"):
        remote_parts = [f"cd {shlex.quote(submit_cwd_value)}"]
        remote_parts.extend(str(cmd) for cmd in submit_cfg.get("setup") or [])
        remote_parts.append("mkdir -p slurm_logs")
        remote_parts.append("sbatch")
        remote_inner = " && ".join(remote_parts)
        remote_cmd = f"bash -lc {shlex.quote(remote_inner)}"
        subprocess.run(
            ["ssh", str(submit_cfg["host"]), remote_cmd],
            input=script,
            text=True,
            check=True,
        )
        return

    submit_cwd = Path(submit_cwd_value)
    if not submit_cwd.exists():
        submit_cwd = repo_root if repo_root.exists() else ROOT
    (submit_cwd / "slurm_logs").mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w", suffix=".sbatch", prefix="pimm-launch-", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        script_path = f.name

    try:
        subprocess.run(["sbatch", script_path], cwd=submit_cwd, check=True)
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("submit", "dry-run"))
    parser.add_argument("recipe", nargs="?", help="Optional launch/runs/*.yaml recipe")
    parser.add_argument("--site", help="Site overlay name, e.g. s3df or nersc")
    parser.add_argument("--config-dir")
    parser.add_argument("--config")
    parser.add_argument("--name")
    parser.add_argument("--wandb-name")
    parser.add_argument("--weight")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dev-mode", action="store_true")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--gpus-per-node", type=int)
    parser.add_argument("--tasks-per-node", type=int)
    parser.add_argument("--cpus-per-task", type=int)
    parser.add_argument("--mem")
    parser.add_argument("--time")
    parser.add_argument("--account")
    parser.add_argument("--partition")
    parser.add_argument("--qos")
    parser.add_argument("--constraint")
    parser.add_argument(
        "--option",
        action="append",
        help="Add/override one training --options entry, e.g. epoch=10",
    )
    parser.add_argument(
        "--set",
        action="append",
        help="Set a launcher config path, e.g. resources.time=02:00:00",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the script instead of submitting when command is submit",
    )
    if hasattr(parser, "parse_intermixed_args"):
        return parser.parse_intermixed_args(argv)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg = resolve_all(load_config(args), timestamp)
    validate_local_config(cfg)

    run_name = build_run_name(cfg, timestamp)
    if not run_name:
        raise SystemExit("Could not determine run name")

    train_cmd = build_train_command(cfg, run_name)
    script = render_script(cfg, run_name, train_cmd)

    if args.command == "dry-run" or args.dry_run:
        print(script)
        return 0

    submit(script, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
