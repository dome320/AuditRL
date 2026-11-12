"""Modal app: train the judge on a cloud GPU and serve adapters behind an OpenAI-style API.

    cd backend
    uv run --extra cloud modal run auditrl/training/modal_app.py --stage sft \\
        --config ../data/training/configs/sft.yaml --dataset ../data/training/judge-v1
    uv run --extra cloud modal run auditrl/training/modal_app.py --stage grpo \\
        --config ../data/training/configs/grpo.yaml --dataset ../data/training/judge-v1
    uv run --extra cloud modal deploy auditrl/training/modal_app.py   # vLLM server

Needs a Modal account (`modal setup`) and a Modal secret named `wandb` holding
WANDB_API_KEY. Datasets and checkpoints live on the `auditrl-judge` volume:
/vol/datasets/<name>/ and /vol/runs/<run_name>/{adapter,checkpoints,result.json}.
"""

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import modal

APP_NAME = "auditrl-judge"
VOL = "/vol"
BASE_MODEL = os.environ.get("AUDITRL_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
GPU = os.environ.get("AUDITRL_TRAIN_GPU", "H100")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(APP_NAME, create_if_missing=True)

AUDITRL_DEPS = [
    "pydantic>=2.9",
    "pydantic-settings>=2.5",
    "sqlmodel>=0.0.22",
    "rapidfuzz>=3.10",
    "rank-bm25>=0.2.2",
    "pyyaml>=6",
    "pypdf>=5",
    "beautifulsoup4>=4.12",
    "lxml>=5",
    "openpyxl>=3.1",
    "markdown>=3.7",
]
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch>=2.6",
        "transformers>=4.56,<6",
        "trl>=1.17,<2",
        "peft>=0.21",
        "datasets>=3.6",
        "accelerate>=1.10",
        "wandb>=0.21",
        *AUDITRL_DEPS,
    )
    .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("auditrl")
)
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm>=0.10")
    .env({"HF_HOME": f"{VOL}/hf", "AUDITRL_BASE_MODEL": BASE_MODEL})
)


@app.function(
    image=train_image,
    gpu=GPU,
    timeout=8 * 3600,
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name("wandb")],
)
def train(config: dict, dataset: str, run_name: str, extra: dict) -> dict:
    from auditrl.training.trainer import LoraParams, TrainConfig, run_grpo, run_sft

    lora = LoraParams(**config.pop("lora", {}))
    cfg = TrainConfig(lora=lora, **config)
    data_dir = Path(VOL) / "datasets" / dataset
    out_dir = Path(VOL) / "runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if cfg.stage == "sft":
            return run_sft(cfg, data_dir, out_dir, run_name, extra)
        if not cfg.init_adapter:
            raise ValueError("grpo needs init_adapter: the run name of an SFT run")
        init = Path(VOL) / "runs" / cfg.init_adapter / "adapter"
        return run_grpo(cfg, data_dir, out_dir, run_name, init, extra)
    finally:
        volume.commit()


@app.function(
    image=vllm_image,
    gpu=GPU,
    volumes={VOL: volume},
    scaledown_window=15 * 60,
    timeout=24 * 3600,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def serve() -> None:
    """Serves the base model (the untuned baseline) plus every trained adapter as
    `judge-<run_name>`, OpenAI-compatible at <url>/v1."""
    volume.reload()
    adapters = sorted(p.parent for p in Path(VOL, "runs").glob("*/adapter/adapter_config.json"))
    cmd = [
        "vllm", "serve", os.environ["AUDITRL_BASE_MODEL"],
        "--host", "0.0.0.0", "--port", "8000",
        "--max-model-len", "4096",
        "--served-model-name", os.environ["AUDITRL_BASE_MODEL"],
    ]  # fmt: skip
    if adapters:
        cmd += ["--enable-lora", "--max-lora-rank", "64", "--lora-modules"]
        cmd += [f"judge-{a.parent.name}={a}" for a in adapters]
    subprocess.Popen(cmd)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@app.local_entrypoint()
def main(
    stage: str = "sft",
    config: str = "../data/training/configs/sft.yaml",
    dataset: str = "../data/training/judge-v1",
    run_name: str = "",
    force_rl: bool = False,
) -> None:
    from auditrl.training.trainer import TrainConfig

    cfg = TrainConfig.from_yaml(Path(config))
    if cfg.stage != stage:
        raise SystemExit(f"config stage {cfg.stage!r} does not match --stage {stage!r}")
    data_dir = Path(dataset).resolve()
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if stage == "grpo" and not force_rl:
        gate_path = data_dir.parent / "rl_gate.json"
        gate = json.loads(gate_path.read_text()) if gate_path.exists() else None
        if not gate or not gate.get("passed"):
            raise SystemExit(
                "RL gate not passed: verifier/reviewer agreement is unmeasured or too low "
                "(run `auditrl train rl-gate`). Use --force-rl to override deliberately."
            )
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_name = run_name or f"{stage}-{manifest['name']}-{stamp}"

    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(data_dir), f"/datasets/{manifest['name']}")
    extra = {"git_commit": _git_commit(), "launched_at": stamp}
    record = train.remote(cfg.to_dict(), manifest["name"], run_name, extra)

    runs = data_dir.parent / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{run_name}.json").write_text(
        json.dumps({**record, "dataset_hash": manifest["dataset_hash"], **extra}, indent=1) + "\n"
    )
    print(json.dumps(record["val_metrics"], indent=1))
    print(f"run record: {runs / f'{run_name}.json'}  wandb: {record['wandb_url']}")
