"""
🚀 Training Entry Point — Orchestrates the full training pipeline.

Usage:
    python train.py                           # Default config (NANO, ~15M params)
    python train.py --config micro            # Smaller model (~4M params)
    python train.py --config debug            # Smoke test (~1M params, 3 epochs)
    python train.py --data-dir data/processed # Custom data directory
    python train.py --resume checkpoints/last.ckpt  # Resume from checkpoint

Requires:
    - Preprocessed data in data/processed/ (train.jsonl, val.jsonl)
    - Run the data pipeline first:
        python data/scripts/clean.py
        python data/scripts/augment.py
        python data/scripts/preprocess.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from rich.console import Console

from src.model.config import ModelConfig, NANO_CONFIG, MICRO_CONFIG, DEBUG_CONFIG
from src.training.lit_module import NanoLogicLitModule
from src.training.dataset import LogicDataModule
from src.tokenizer.tokenizer import load_tokenizer

console = Console()

CONFIGS = {
    "nano": NANO_CONFIG,
    "micro": MICRO_CONFIG,
    "debug": DEBUG_CONFIG,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the NanoLogicTransformer")
    parser.add_argument(
        "--config",
        type=str,
        default="nano",
        choices=list(CONFIGS.keys()),
        help="Model config preset (default: nano)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory with train.jsonl, val.jsonl",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="data/processed/tokenizer.json",
        help="Path to trained tokenizer",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max_epochs from config",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch_size from config",
    )
    args = parser.parse_args()

    # --- Config ---
    config = CONFIGS[args.config]
    if args.epochs:
        config.max_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size

    console.print("[bold cyan]🚀 NanoLogicTransformer Training[/]\n")
    console.print(f"   Config:    {args.config}")
    console.print(f"   Data:      {args.data_dir}")
    console.print(f"   Epochs:    {config.max_epochs}")
    console.print(f"   Batch:     {config.batch_size}")
    console.print(f"   Precision: {config.precision}")
    console.print(f"   Params:    {config.estimated_params}")

    # --- Tokenizer ---
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        console.print(f"\n[bold red]❌ Tokenizer not found: {tokenizer_path}[/]")
        console.print("[yellow]   Run the preprocessing pipeline first to train the tokenizer.[/]")
        return

    tokenizer = load_tokenizer(tokenizer_path)
    config.vocab_size = tokenizer.get_vocab_size()
    config.pad_token_id = tokenizer.token_to_id("<|pad|>")
    config.bos_token_id = tokenizer.token_to_id("<|bos|>")
    config.eos_token_id = tokenizer.token_to_id("<|eos|>")

    console.print(f"   Vocab:     {config.vocab_size}")

    # --- DataModule ---
    datamodule = LogicDataModule(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        num_workers=2,
        pad_id=config.pad_token_id,
    )

    # --- Model ---
    model = NanoLogicLitModule(config)
    console.print(f"\n{model.model.summary()}")

    # --- Callbacks ---
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="best-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor="val/loss",
            patience=10,
            mode="min",
            verbose=True,
        ),
        RichProgressBar(),
    ]

    # --- Logger ---
    logger = TensorBoardLogger("logs", name="nano-logic")

    # --- Trainer ---
    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        precision=config.precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=1,
        log_every_n_steps=10,
        val_check_interval=0.5,  # Validate twice per epoch
        deterministic=False,
    )

    # --- Train ---
    console.print("\n[bold green]🏋️ Starting training...[/]\n")
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)

    # --- Test ---
    if (Path(args.data_dir) / "test.jsonl").exists():
        console.print("\n[bold cyan]🧪 Running test evaluation...[/]\n")
        trainer.test(model, datamodule=datamodule)

    console.print("\n[bold green]✅ Training complete![/]")
    console.print(f"   Best model: {callbacks[0].best_model_path}")
    console.print(f"   Logs: logs/nano-logic/")


if __name__ == "__main__":
    main()
