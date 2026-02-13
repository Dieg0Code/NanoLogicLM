"""
🎯 Inference Entry Point — Translate natural language to propositional logic.

Usage:
    python inference.py "Si llueve entonces llevo paraguas"
    python inference.py "Si llueve entonces llevo paraguas" --beam 5
    python inference.py --interactive

Examples:
    $ python inference.py "Si el servidor crashea y no hay backup, se pierden los datos"
    📐 Fórmula: (p ∧ ¬q) → r
    🔤 ASCII:   (p & ~q) -> r
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rich.console import Console

from src.model.config import NANO_CONFIG
from src.model.transformer import NanoLogicTransformer
from src.training.lit_module import NanoLogicLitModule
from src.tokenizer.tokenizer import load_tokenizer
from src.tokenizer.special_tokens import SPECIAL_TOKENS
from src.inference.generator import greedy_decode, beam_search
from src.inference.parser import parse_sequence
from src.inference.explainer import explain

console = Console()


def load_model(
    checkpoint_path: str,
    tokenizer_path: str = "data/processed/tokenizer.json",
    device: str = "auto",
) -> tuple[NanoLogicTransformer, any]:
    """Load trained model and tokenizer.

    Args:
        checkpoint_path: Path to .ckpt file
        tokenizer_path: Path to trained tokenizer
        device: Device to load on

    Returns:
        (model, tokenizer) tuple
    """
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)

    # Load from Lightning checkpoint
    lit_module = NanoLogicLitModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model = lit_module.model
    model.eval()

    # Move to device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, tokenizer


def translate(
    text: str,
    model: NanoLogicTransformer,
    tokenizer,
    beam_width: int = 1,
    max_len: int = 128,
) -> str:
    """Translate natural language to propositional logic formula.

    Args:
        text: Natural language input
        model: Trained transformer
        tokenizer: Trained tokenizer
        beam_width: Beam width (1 = greedy)
        max_len: Max generation length

    Returns:
        Explanation string
    """
    device = next(model.parameters()).device

    # Encode source
    src_text = f"{SPECIAL_TOKENS.INPUT} {text}"
    src_encoding = tokenizer.encode(src_text)
    src_ids = torch.tensor([src_encoding.ids], dtype=torch.long, device=device)
    src_mask = model.create_padding_mask(src_ids, tokenizer.token_to_id(SPECIAL_TOKENS.PAD))

    # Encode
    with torch.no_grad():
        encoder_output = model.encode(src_ids, src_mask)

    # Decode
    bos_id = tokenizer.token_to_id(SPECIAL_TOKENS.BOS)
    eos_id = tokenizer.token_to_id(SPECIAL_TOKENS.EOS)

    if beam_width > 1:
        sequences = beam_search(
            model,
            encoder_output,
            src_mask,
            bos_id,
            eos_id,
            beam_width=beam_width,
            max_len=max_len,
        )
        token_ids = sequences[0] if sequences else []
    else:
        token_ids = greedy_decode(
            model,
            encoder_output,
            src_mask,
            bos_id,
            eos_id,
            max_len=max_len,
        )

    # Decode tokens back to string
    output_text = tokenizer.decode(token_ids)
    parsed = parse_sequence(output_text)

    return explain(text, parsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate NL → Propositional Logic")
    parser.add_argument("text", type=str, nargs="?", help="Text to translate")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/last.ckpt",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="data/processed/tokenizer.json",
        help="Tokenizer path",
    )
    parser.add_argument("--beam", type=int, default=1, help="Beam width (default: 1 = greedy)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    # Check checkpoint
    if not Path(args.checkpoint).exists():
        console.print(f"[bold red]❌ Checkpoint not found: {args.checkpoint}[/]")
        console.print("[yellow]   Train the model first: python train.py[/]")
        return

    console.print("[bold cyan]🧠 Loading model...[/]")
    model, tokenizer = load_model(args.checkpoint, args.tokenizer)
    console.print("[bold green]✅ Model loaded![/]\n")

    if args.interactive:
        console.print("[bold]Interactive mode — type 'exit' to quit\n[/]")
        while True:
            try:
                text = input("📝 > ").strip()
                if text.lower() in ("exit", "quit", "q"):
                    break
                if not text:
                    continue
                result = translate(text, model, tokenizer, beam_width=args.beam)
                console.print(f"\n{result}\n")
            except KeyboardInterrupt:
                break
        console.print("\n[bold]👋 ¡Hasta luego![/]")
    elif args.text:
        result = translate(args.text, model, tokenizer, beam_width=args.beam)
        console.print(result)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
