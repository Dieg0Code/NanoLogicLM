"""
train_tokenizer.py — Script para entrenar el tokenizer BPE de NanoLogic.

¿Qué hace?
    1. Lee los datos procesados (train.jsonl + val.jsonl)
    2. Entrena un tokenizer BPE con ellos
    3. Guarda el tokenizer entrenado en models/tokenizer/
    4. Muestra estadísticas y ejemplos de tokenización

¿Cuándo se ejecuta?
    UNA SOLA VEZ — después de preprocessar los datos y antes de entrenar el modelo.
    El tokenizer entrenado se reutiliza en todo el entrenamiento y la inferencia.

Uso:
    uv run python data/scripts/train_tokenizer.py

Opciones:
    --train       Path al archivo de entrenamiento (default: data/processed/train.jsonl)
    --val         Path al archivo de validación (default: data/processed/val.jsonl)
    --output      Directorio de salida (default: models/tokenizer)
    --vocab-size  Tamaño del vocabulario (default: 8000)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Agregar el directorio raíz al path para poder importar src.*
# Esto es necesario porque estamos en data/scripts/ pero importamos de src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rich.console import Console
from rich.table import Table

from src.tokenizer.special_tokens import SPECIAL_TOKENS
from src.tokenizer.tokenizer import NanoLogicTokenizer

console = Console()


def count_examples(file_path: str) -> int:
    """Cuenta cuántos ejemplos hay en un archivo JSONL."""
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def show_tokenization_examples(tokenizer: NanoLogicTokenizer, file_path: str) -> None:
    """Muestra ejemplos de cómo el tokenizer parte las secuencias.

    Esto es MUY útil para verificar que:
    - Los special tokens se mantienen como tokens individuales (no se parten)
    - Los conectores lógicos (∧, ∨, →, ↔, ¬) se tokenizan correctamente
    - Las palabras comunes en español se mantienen enteras (no se fragmentan mucho)
    """
    console.print("\n[bold cyan]📝 Ejemplos de tokenización:[/]")

    with open(file_path, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    # Mostrar 3 ejemplos variados: uno simple y de distintos bloques
    shown = 0
    seen_blocks = set()
    for ex in examples:
        block = ex.get("block", "")
        if block in seen_blocks:
            continue
        seen_blocks.add(block)

        console.print(f"\n[bold]Ejemplo {shown + 1}[/] ({ex.get('complexity', '?')} — {block}):")
        console.print(f"  [dim]NL:[/] {ex['natural_language_input'][:100]}...")
        console.print(f"  [dim]Fórmula:[/] {ex['formula']}")

        # Tokenizar solo la fórmula para mostrar cómo se parte
        formula_tokens = tokenizer.encode_to_tokens(ex["formula"])
        formula_ids = tokenizer.encode(ex["formula"])
        console.print(f"  [green]Tokens fórmula:[/] {formula_tokens}")
        console.print(f"  [green]IDs fórmula:[/] {formula_ids}")
        console.print(
            f"  [green]Largo secuencia completa:[/] {len(tokenizer.encode_example(ex))} tokens"
        )

        shown += 1
        if shown >= 3:
            break

    # Mostrar cómo se tokenizan los special tokens (verificar que NO se parten)
    console.print("\n[bold cyan]🔒 Verificación de special tokens:[/]")
    for token in SPECIAL_TOKENS.as_list():
        ids = tokenizer.encode(token)
        tokens = tokenizer.encode_to_tokens(token)
        # Si un special token se tokeniza en más de 1 token, hay un PROBLEMA
        status = "✅" if len(ids) == 1 else "❌ BROKEN!"
        console.print(f"  {status} {token:20s} → IDs: {ids}  Tokens: {tokens}")


def show_stats(tokenizer: NanoLogicTokenizer, train_path: str, val_path: str) -> None:
    """Muestra estadísticas del tokenizer entrenado."""

    # Tabla de info general
    table = Table(title="📊 Tokenizer Stats")
    table.add_column("Propiedad", style="bold")
    table.add_column("Valor", style="green")

    table.add_row("Vocab size", str(tokenizer.vocab_size))
    table.add_row("Special tokens", str(SPECIAL_TOKENS.count))
    table.add_row("BPE tokens", str(tokenizer.vocab_size - SPECIAL_TOKENS.count))
    table.add_row("PAD ID", str(tokenizer.pad_id))
    table.add_row("BOS ID", str(tokenizer.bos_id))
    table.add_row("EOS ID", str(tokenizer.eos_id))
    table.add_row("Train examples", str(count_examples(train_path)))
    table.add_row("Val examples", str(count_examples(val_path)))

    console.print(table)

    # Distribución de largos de secuencia
    console.print("\n[bold cyan]📏 Distribución de largos de secuencia:[/]")
    lengths = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                ids = tokenizer.encode_example(ex)
                lengths.append(len(ids))

    if lengths:
        lengths.sort()
        console.print(f"  Mínimo:     {lengths[0]} tokens")
        console.print(f"  Máximo:     {lengths[-1]} tokens")
        console.print(f"  Promedio:   {sum(lengths) / len(lengths):.0f} tokens")
        console.print(f"  Mediana:    {lengths[len(lengths) // 2]} tokens")
        console.print(f"  P95:        {lengths[int(len(lengths) * 0.95)]} tokens")
        console.print(f"  P99:        {lengths[int(len(lengths) * 0.99)]} tokens")

        # Esto es IMPORTANTE para decidir el max_length del modelo.
        # Si el P99 es 512, entonces max_length=512 cubre el 99% de los datos.
        console.print(
            f"\n  [bold yellow]💡 Sugerencia de max_length:[/] "
            f"{lengths[int(len(lengths) * 0.99)]} tokens (P99)"
        )


def main() -> None:
    """Punto de entrada principal del script."""

    parser = argparse.ArgumentParser(description="Entrena el tokenizer BPE para NanoLogic")
    parser.add_argument(
        "--train",
        default="data/processed/train.jsonl",
        help="Path al archivo de entrenamiento (JSONL)",
    )
    parser.add_argument(
        "--val",
        default="data/processed/val.jsonl",
        help="Path al archivo de validación (JSONL)",
    )
    parser.add_argument(
        "--output",
        default="models/tokenizer",
        help="Directorio donde guardar el tokenizer entrenado",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="Tamaño del vocabulario BPE (default: 8000)",
    )
    args = parser.parse_args()

    # ----- Verificar que los archivos existen -----
    for path_str in [args.train, args.val]:
        if not Path(path_str).exists():
            console.print(f"[bold red]❌ No se encontró: {path_str}[/]")
            console.print("   Asegúrate de haber corrido preprocess.py primero.")
            sys.exit(1)

    # ----- Entrenar -----
    console.print("[bold cyan]🔤 BPE Tokenizer Training[/]")
    console.print(f"   Train:      {args.train}")
    console.print(f"   Val:        {args.val}")
    console.print(f"   Output:     {args.output}")
    console.print(f"   Vocab size: {args.vocab_size}\n")

    tokenizer = NanoLogicTokenizer()

    console.print("[bold]🔄 Entrenando tokenizer...[/]")
    tokenizer.train(
        files=[args.train, args.val],
        vocab_size=args.vocab_size,
    )
    console.print("[bold green]✅ Tokenizer entrenado![/]")

    # ----- Guardar -----
    tokenizer.save(args.output)
    console.print(f"[bold green]✅ Guardado en {args.output}/tokenizer.json[/]")

    # ----- Mostrar estadísticas -----
    show_stats(tokenizer, args.train, args.val)

    # ----- Mostrar ejemplos -----
    show_tokenization_examples(tokenizer, args.train)

    console.print("\n[bold green]🎉 ¡Listo! El tokenizer está entrenado y guardado.[/]")
    console.print(f"   Para usarlo: NanoLogicTokenizer.load('{args.output}')")


if __name__ == "__main__":
    main()
