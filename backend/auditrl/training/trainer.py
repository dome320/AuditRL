"""LoRA SFT and GRPO for the claim judge (TRL + PEFT), tracked in Weights & Biases.

Provider-agnostic: runs on any CUDA machine with the `train` extra installed. The Modal app
(`modal_app.py`) calls these functions on a cloud GPU. Heavy imports stay inside functions
so the rest of AuditRL never needs torch.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from auditrl.judge.format import JUDGE_PROMPT_VERSION
from auditrl.training.rewards import summarize


@dataclass
class LoraParams:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"


@dataclass
class TrainConfig:
    stage: str = "sft"  # sft | grpo
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    seed: int = 0
    lora: LoraParams = field(default_factory=LoraParams)
    # SFT
    epochs: float = 2.0
    learning_rate: float = 1e-4
    batch_size: int = 4
    grad_accum: int = 4
    max_length: int = 2048
    warmup_ratio: float = 0.03
    # GRPO (continues training the SFT adapter)
    init_adapter: str | None = None  # run name of the SFT run on the volume
    max_steps: int = 300
    num_generations: int = 8
    max_completion_length: int = 384
    temperature: float = 0.9
    beta: float = 0.0
    # Evaluation and tracking
    eval_samples: int = 300
    wandb_project: str = "auditrl"
    wandb_entity: str | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":
        raw = yaml.safe_load(path.read_text()) or {}
        lora = LoraParams(**raw.pop("lora", {}))
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(lora=lora, **raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _wandb_init(cfg: TrainConfig, run_name: str, manifest: dict[str, Any], extra: dict[str, Any]):
    import wandb

    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=run_name,
        job_type=cfg.stage,
        config={
            **cfg.to_dict(),
            "dataset_name": manifest.get("name"),
            "dataset_hash": manifest.get("dataset_hash"),
            "dataset_counts": manifest.get("counts"),
            "bench_manifest_hash": manifest.get("bench_manifest_hash"),
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            **extra,
        },
    )


def _lora_config(cfg: TrainConfig):
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        task_type="CAUSAL_LM",
    )


def _load_base(cfg: TrainConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    return model, tok


def generate_eval(model, tok, rows: list[dict[str, Any]], batch_size: int = 16) -> dict[str, Any]:
    """Greedy-decode the judge on labeled rows and score with the reward/metric functions."""
    import torch

    tok.padding_side = "left"
    model.eval()
    outputs: list[str] = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts = [
            tok.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=False)
            for r in batch
        ]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=384, do_sample=False, pad_token_id=tok.pad_token_id
            )
        outputs += tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    return {"metrics": summarize(rows, outputs), "samples": outputs[:20]}


def run_sft(
    cfg: TrainConfig, data_dir: Path, out_dir: Path, run_name: str, extra: dict[str, Any]
) -> dict[str, Any]:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    manifest = json.loads((data_dir / "manifest.json").read_text())
    train_rows, val_rows = _rows(data_dir / "train.jsonl"), _rows(data_dir / "val.jsonl")
    eval_rows = val_rows[: cfg.eval_samples]
    cols = ("prompt", "completion")
    train_ds = Dataset.from_list([{k: r[k] for k in cols} for r in train_rows])
    val_ds = Dataset.from_list([{k: r[k] for k in cols} for r in eval_rows])

    run = _wandb_init(cfg, run_name, manifest, extra)
    model, tok = _load_base(cfg)
    args = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        max_length=cfg.max_length,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to=["wandb"],
        run_name=run_name,
        seed=cfg.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        peft_config=_lora_config(cfg),
    )
    result = trainer.train()
    return _finish(trainer, tok, run, cfg, out_dir, run_name, eval_rows, result.training_loss)


def run_grpo(
    cfg: TrainConfig,
    data_dir: Path,
    out_dir: Path,
    run_name: str,
    init_adapter_dir: Path,
    extra: dict[str, Any],
) -> dict[str, Any]:
    from datasets import Dataset
    from peft import PeftModel
    from trl import GRPOConfig, GRPOTrainer

    from auditrl.training.rewards import judge_reward

    manifest = json.loads((data_dir / "manifest.json").read_text())
    train_rows, val_rows = _rows(data_dir / "train.jsonl"), _rows(data_dir / "val.jsonl")
    eval_rows = val_rows[: cfg.eval_samples]
    train_ds = Dataset.from_list(
        [
            {
                "prompt": r["prompt"],
                "claim": r["claim"],
                "evidence": json.dumps(r["evidence"]),
                "target": json.dumps(r["target"]),
            }
            for r in train_rows
        ]
    )

    def reward(prompts, completions, claim, evidence, target, **kw):
        return judge_reward(
            prompts,
            completions,
            claim,
            [json.loads(e) for e in evidence],
            [json.loads(t) for t in target],
        )

    reward.__name__ = "judge_reward"

    run = _wandb_init(cfg, run_name, manifest, {**extra, "init_adapter": cfg.init_adapter})
    base, tok = _load_base(cfg)
    model = PeftModel.from_pretrained(base, str(init_adapter_dir), is_trainable=True)
    args = GRPOConfig(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.num_generations,
        gradient_accumulation_steps=cfg.grad_accum,
        num_generations=cfg.num_generations,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.temperature,
        beta=cfg.beta,
        max_steps=cfg.max_steps,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        report_to=["wandb"],
        run_name=run_name,
        seed=cfg.seed,
        log_completions=True,
    )
    trainer = GRPOTrainer(
        model=model, reward_funcs=[reward], args=args, train_dataset=train_ds, processing_class=tok
    )
    result = trainer.train()
    return _finish(trainer, tok, run, cfg, out_dir, run_name, eval_rows, result.training_loss)


def _finish(trainer, tok, run, cfg, out_dir, run_name, eval_rows, train_loss) -> dict[str, Any]:
    import wandb

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    evaluation = generate_eval(trainer.model, tok, eval_rows)
    run.log({f"val/{k}": v for k, v in evaluation["metrics"].items()})
    run.summary.update({f"val/{k}": v for k, v in evaluation["metrics"].items()})
    artifact = wandb.Artifact(
        f"judge-{run_name}",
        type="model",
        metadata={"base_model": cfg.base_model, "stage": cfg.stage},
    )
    artifact.add_dir(str(adapter_dir))
    run.log_artifact(artifact)
    record = {
        "run_name": run_name,
        "stage": cfg.stage,
        "base_model": cfg.base_model,
        "adapter_dir": str(adapter_dir),
        "train_loss": train_loss,
        "val_metrics": evaluation["metrics"],
        "val_samples": evaluation["samples"],
        "wandb_url": run.get_url(),
        "config": cfg.to_dict(),
    }
    (out_dir / "result.json").write_text(json.dumps(record, indent=1) + "\n")
    run.finish()
    return record
