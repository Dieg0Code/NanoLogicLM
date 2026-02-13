"""
Preprocessing pipeline — Fases 13-19.

Balances the dataset and converts it to the final training format with special tokens.

Pipeline:
    Phase 13-16 (Balancing):
        - By complexity: ~33% each (simple/intermediate/advanced)
        - By connectors: uniform distribution of ∧, ∨, →, ↔, ¬
        - By atom count: uniform distribution of 2-7 atoms
        - By domain/block: none dominates >25%

    Phase 17 (Format):
        - JSON → sequences with special tokens
        - Phase 1 format (with thought): <|bos|><|input|> NL <|thought|> reasoning <|formula|> F <|eos|>
        - Phase 2 format (no thought):   <|bos|><|input|> NL <|formula|> F <|eos|>

    Phase 18 (Split):
        - Train/Val/Test: 80/10/10
        - Split by logical pattern (not random) to test generalization

    Phase 19 (Curriculum):
        - Order training set: simple → intermediate → advanced

Usage:
    python data/scripts/preprocess.py --input data/raw/dataset_augmented.json
    python data/scripts/preprocess.py --input data/raw/dataset_augmented.json --output-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

random.seed(42)

# Import special tokens
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.tokenizer.special_tokens import SPECIAL_TOKENS  # noqa: E402


# ============================================================
# Balancing (Phases 13-16)
# ============================================================


def get_connector_profile(formula: str) -> set[str]:
    """Extract which connectors are used in a formula."""
    connectors = set()
    if "¬" in formula or "~" in formula:
        connectors.add("¬")
    if "∧" in formula or "&" in formula:
        connectors.add("∧")
    if "∨" in formula or "|" in formula:
        connectors.add("∨")
    if "→" in formula or "->" in formula:
        connectors.add("→")
    if "↔" in formula or "<->" in formula:
        connectors.add("↔")
    return connectors


def get_atom_count(formula: str) -> int:
    """Count unique atoms in a formula."""
    atoms = set(re.findall(r"(?<![¬~\w])([a-z]\d*)", formula))
    return len(atoms)


def balance_by_complexity(
    examples: list[dict], target_ratio: dict[str, float] | None = None
) -> list[dict]:
    """Balance examples by complexity level.

    Default: ~33% each (simple, intermediate, advanced).
    Uses downsampling of over-represented classes.
    """
    if target_ratio is None:
        target_ratio = {"simple": 0.33, "intermediate": 0.34, "advanced": 0.33}

    by_complexity: dict[str, list[dict]] = {}
    for ex in examples:
        comp = ex.get("complexity", "unknown")
        by_complexity.setdefault(comp, []).append(ex)

    # Find target count (based on smallest class)
    min_class_size = min(len(v) for v in by_complexity.values()) if by_complexity else 0
    total = len(examples)

    balanced: list[dict] = []
    for comp, ratio in target_ratio.items():
        available = by_complexity.get(comp, [])
        target = min(int(total * ratio), len(available))
        if len(available) > target:
            balanced.extend(random.sample(available, target))
        else:
            balanced.extend(available)

    # Add any unknown complexity examples
    for comp, exs in by_complexity.items():
        if comp not in target_ratio:
            balanced.extend(exs)

    return balanced


def balance_by_block(examples: list[dict], max_ratio: float = 0.25) -> list[dict]:
    """Ensure no block/domain dominates more than max_ratio of the dataset."""
    by_block: dict[str, list[dict]] = {}
    for ex in examples:
        block = ex.get("block", "unknown")
        by_block.setdefault(block, []).append(ex)

    total = len(examples)
    max_per_block = int(total * max_ratio)

    balanced: list[dict] = []
    for block, exs in by_block.items():
        if len(exs) > max_per_block:
            balanced.extend(random.sample(exs, max_per_block))
        else:
            balanced.extend(exs)

    return balanced


def balance_dataset(examples: list[dict], verbose: bool = True) -> list[dict]:
    """Run the full balancing pipeline (Phases 13-16)."""
    if verbose:
        console.print(f"\n[bold cyan]⚖️ Balancing dataset[/]")
        console.print(f"   Input: {len(examples)} examples")

    # Phase 13: By complexity
    examples = balance_by_complexity(examples)
    if verbose:
        console.print(f"   After complexity balance: {len(examples)}")

    # Phase 16: By block/domain
    examples = balance_by_block(examples)
    if verbose:
        console.print(f"   After block balance: {len(examples)}")

    # Phases 14-15: We report distributions but don't aggressively filter
    # (would lose too much data for a small dataset)

    if verbose:
        console.print(f"\n   [bold green]✅ Balanced: {len(examples)} examples[/]")

    return examples


# ============================================================
# Format conversion (Phase 17)
# ============================================================


def format_with_thought(example: dict) -> str:
    """Format example WITH thought chain (Phase 1 training).

    Format: <|bos|><|input|> NL <|thought|> reasoning <|formula|> F <|eos|>
    """
    nl = example.get("natural_language_input", "")
    formula = example.get("output", {}).get("formula", "")

    # Extract thought as a single string
    thought = example.get("thought", {})
    reasoning_steps = thought.get("reasoning_steps", [])
    thought_text = " ".join(
        step.get("explanation", "") for step in reasoning_steps if isinstance(step, dict)
    )

    seq = (
        f"{SPECIAL_TOKENS.BOS}"
        f"{SPECIAL_TOKENS.INPUT} {nl} "
        f"{SPECIAL_TOKENS.THOUGHT} {thought_text} "
        f"{SPECIAL_TOKENS.FORMULA} {formula}"
        f"{SPECIAL_TOKENS.EOS}"
    )
    return seq


def format_without_thought(example: dict) -> str:
    """Format example WITHOUT thought chain (Phase 2 training).

    Format: <|bos|><|input|> NL <|formula|> F <|eos|>
    """
    nl = example.get("natural_language_input", "")
    formula = example.get("output", {}).get("formula", "")

    seq = (
        f"{SPECIAL_TOKENS.BOS}"
        f"{SPECIAL_TOKENS.INPUT} {nl} "
        f"{SPECIAL_TOKENS.FORMULA} {formula}"
        f"{SPECIAL_TOKENS.EOS}"
    )
    return seq


# ============================================================
# Train/Val/Test split (Phase 18)
# ============================================================


def get_formula_pattern(formula: str) -> str:
    """Extract the structural pattern of a formula (for stratified split).

    Replaces atoms with placeholders to get the logical structure.
    Example: (p ∧ q) → r  becomes  (X ∧ X) → X
    """
    # Replace atoms with X
    pattern = re.sub(r"[a-z]\d*", "X", formula)
    # Normalize whitespace
    pattern = re.sub(r"\s+", " ", pattern).strip()
    return pattern


def stratified_split(
    examples: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split dataset by logical pattern to test generalization.

    Examples with the same formula structure go to the same split.
    """
    # Group by pattern
    by_pattern: dict[str, list[dict]] = {}
    for ex in examples:
        formula = ex.get("output", {}).get("formula", "")
        pattern = get_formula_pattern(formula)
        by_pattern.setdefault(pattern, []).append(ex)

    # Shuffle patterns
    patterns = list(by_pattern.keys())
    random.shuffle(patterns)

    # Allocate patterns to splits
    total = len(examples)
    train_target = int(total * train_ratio)
    val_target = int(total * val_ratio)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []

    for pattern in patterns:
        exs = by_pattern[pattern]
        if len(train) < train_target:
            train.extend(exs)
        elif len(val) < val_target:
            val.extend(exs)
        else:
            test.extend(exs)

    return train, val, test


# ============================================================
# Curriculum ordering (Phase 19)
# ============================================================

COMPLEXITY_ORDER = {"simple": 0, "intermediate": 1, "advanced": 2}


def curriculum_sort(examples: list[dict]) -> list[dict]:
    """Sort examples by complexity: simple → intermediate → advanced."""
    return sorted(
        examples,
        key=lambda ex: COMPLEXITY_ORDER.get(ex.get("complexity", ""), 99),
    )


# ============================================================
# Full pipeline
# ============================================================


def preprocess_dataset(
    examples: list[dict],
    output_dir: Path,
    *,
    include_thought: bool = True,
    verbose: bool = True,
) -> dict:
    """Run the full preprocessing pipeline (Phases 13-19).

    Args:
        examples: List of augmented examples
        output_dir: Directory to write train.jsonl, val.jsonl, test.jsonl
        include_thought: Whether to include thought chain in sequences
        verbose: Print progress

    Returns:
        Stats dict
    """
    # Phase 13-16: Balance
    balanced = balance_dataset(examples, verbose=verbose)

    # Phase 18: Split
    train, val, test = stratified_split(balanced)

    if verbose:
        console.print(f"\n[bold cyan]📂 Split:[/]")
        console.print(f"   Train: {len(train)}")
        console.print(f"   Val:   {len(val)}")
        console.print(f"   Test:  {len(test)}")

    # Phase 19: Curriculum sort (train only)
    train = curriculum_sort(train)

    # Phase 17: Format and save
    output_dir.mkdir(parents=True, exist_ok=True)
    format_fn = format_with_thought if include_thought else format_without_thought

    splits = {"train": train, "val": val, "test": test}
    for split_name, split_data in splits.items():
        filepath = output_dir / f"{split_name}.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for ex in split_data:
                record = {
                    "sequence": format_fn(ex),
                    "natural_language_input": ex.get("natural_language_input", ""),
                    "formula": ex.get("output", {}).get("formula", ""),
                    "complexity": ex.get("complexity", ""),
                    "block": ex.get("block", ""),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if verbose:
            console.print(f"   ✅ Wrote {filepath} ({len(split_data)} examples)")

    stats = {
        "balanced_total": len(balanced),
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }

    # Print distribution tables
    if verbose:
        _print_distributions(balanced)

    return stats


def _print_distributions(examples: list[dict]) -> None:
    """Print distribution tables for the balanced dataset."""
    console.print("\n[bold]📊 Final Distributions:[/]\n")

    # Complexity
    table = Table(title="By Complexity")
    table.add_column("Level", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Ratio", justify="right")
    complexity_counts = Counter(ex.get("complexity", "?") for ex in examples)
    total = len(examples)
    for comp in ["simple", "intermediate", "advanced"]:
        count = complexity_counts.get(comp, 0)
        table.add_row(comp, str(count), f"{count / total:.1%}")
    console.print(table)

    # Connectors
    table = Table(title="By Connector")
    table.add_column("Connector", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Ratio", justify="right")
    connector_counts: Counter[str] = Counter()
    for ex in examples:
        formula = ex.get("output", {}).get("formula", "")
        for conn in get_connector_profile(formula):
            connector_counts[conn] += 1
    for conn in ["¬", "∧", "∨", "→", "↔"]:
        count = connector_counts.get(conn, 0)
        table.add_row(conn, str(count), f"{count / total:.1%}")
    console.print(table)

    # Atom count
    table = Table(title="By Atom Count")
    table.add_column("Atoms", style="cyan")
    table.add_column("Count", justify="right")
    atom_counts = Counter(
        get_atom_count(ex.get("output", {}).get("formula", "")) for ex in examples
    )
    for n in sorted(atom_counts.keys()):
        table.add_row(str(n), str(atom_counts[n]))
    console.print(table)


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess dataset: balance, format, split (Phases 13-19)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/dataset_augmented.json",
        help="Input JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--no-thought",
        action="store_true",
        help="Format without thought chain",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        console.print(f"[bold red]❌ File not found: {input_path}[/]")
        sys.exit(1)

    console.print(f"[bold]🔧 Preprocessing Pipeline[/]")
    console.print(f"   Input:  {input_path}")
    console.print(f"   Output: {output_dir}/")

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    examples = data.get("examples", data if isinstance(data, list) else [])

    stats = preprocess_dataset(
        examples,
        output_dir,
        include_thought=not args.no_thought,
    )

    console.print(f"\n[bold green]✅ Preprocessing complete![/]")
    console.print(f"   Train: {stats['train']} | Val: {stats['val']} | Test: {stats['test']}")


if __name__ == "__main__":
    main()
