"""
Dataset cleaning pipeline — Fases 1-6.

Validates, deduplicates, filters, and normalizes the raw dataset from DeepSeek.

Usage:
    python data/scripts/clean.py                                    # defaults
    python data/scripts/clean.py --input dataset.json               # custom input
    python data/scripts/clean.py --input data/raw/dataset.json --output data/raw/dataset_clean.json

Pipeline steps:
    1. Validate formula syntax (recursive descent parser)
    2. Check balanced parentheses
    3. Check atom consistency (declared vs used)
    4. Deduplicate (exact match on input + formula)
    5. Filter trivial formulas (subsample overly simple ones)
    6. Normalize format (whitespace, parentheses, case)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


# ============================================================
# Formula Parser — Recursive descent for propositional logic
# ============================================================

# Unicode connectors
CONNECTORS_UNICODE = {"¬", "∧", "∨", "→", "↔"}
# ASCII connectors
CONNECTORS_ASCII = {"~", "&", "|", "->", "<->"}
# All connectors (for detection)
ALL_CONNECTORS = CONNECTORS_UNICODE | CONNECTORS_ASCII


class FormulaParseError(Exception):
    """Raised when a formula cannot be parsed."""


class FormulaTokenizer:
    """Tokenizes a propositional logic formula string."""

    # Token types
    ATOM = "ATOM"
    NOT = "NOT"
    AND = "AND"
    OR = "OR"
    IMPLIES = "IMPLIES"
    BICONDITIONAL = "BICONDITIONAL"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    EOF = "EOF"

    # Mapping from symbols to token types
    SYMBOL_MAP = {
        "¬": "NOT",
        "~": "NOT",
        "∧": "AND",
        "&": "AND",
        "∨": "OR",
        "|": "OR",
        "→": "IMPLIES",
        "->": "IMPLIES",
        "↔": "BICONDITIONAL",
        "<->": "BICONDITIONAL",
    }

    def __init__(self, formula: str) -> None:
        self.formula = formula.strip()
        self.pos = 0  # Position in the formula string
        self.tokens: list[tuple[str, str]] = []
        self._tokenize()

    def _tokenize(self) -> None:
        """Convert formula string to a list of (type, value) tokens."""
        i = 0
        s = self.formula
        while i < len(s):
            ch = s[i]

            # Skip whitespace
            if ch.isspace():
                i += 1
                continue

            # Parentheses
            if ch == "(":
                self.tokens.append((self.LPAREN, "("))
                i += 1
                continue
            if ch == ")":
                self.tokens.append((self.RPAREN, ")"))
                i += 1
                continue

            # Multi-char operators: <->, ->
            if s[i : i + 3] == "<->":
                self.tokens.append((self.BICONDITIONAL, "<->"))
                i += 3
                continue
            if s[i : i + 2] == "->":
                self.tokens.append((self.IMPLIES, "->"))
                i += 2
                continue

            # Unicode operators
            if ch in self.SYMBOL_MAP:
                self.tokens.append((self.SYMBOL_MAP[ch], ch))
                i += 1
                continue

            # ASCII operators
            if ch in ("~", "&", "|"):
                self.tokens.append((self.SYMBOL_MAP[ch], ch))
                i += 1
                continue

            # Atoms: lowercase letters (possibly with subscripts/primes)
            if ch.isalpha() and ch.islower():
                atom = ch
                i += 1
                # Allow multi-char atoms like p1, p2, etc.
                while i < len(s) and (s[i].isdigit() or s[i] == "'"):
                    atom += s[i]
                    i += 1
                self.tokens.append((self.ATOM, atom))
                continue

            # Unknown character — skip but note it
            i += 1

        self.tokens.append((self.EOF, ""))


class FormulaParser:
    """Recursive descent parser for propositional logic formulas.

    Grammar:
        formula     → biconditional
        biconditional → implication (↔ implication)*
        implication → disjunction (→ disjunction)*
        disjunction → conjunction (∨ conjunction)*
        conjunction → unary (∧ unary)*
        unary       → ¬ unary | atom | ( formula )
        atom        → [a-z][0-9']*
    """

    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.pos = 0
        self.atoms_found: set[str] = set()

    def peek(self) -> tuple[str, str]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return (FormulaTokenizer.EOF, "")

    def consume(self, expected_type: str | None = None) -> tuple[str, str]:
        token = self.peek()
        if expected_type and token[0] != expected_type:
            raise FormulaParseError(
                f"Expected {expected_type}, got {token[0]} ('{token[1]}') at position {self.pos}"
            )
        self.pos += 1
        return token

    def parse(self) -> bool:
        """Parse the formula. Returns True if valid."""
        self.formula()
        # Should be at EOF
        if self.peek()[0] != FormulaTokenizer.EOF:
            raise FormulaParseError(f"Unexpected token after formula: {self.peek()[1]}")
        return True

    def formula(self) -> None:
        self.biconditional()

    def biconditional(self) -> None:
        self.implication()
        while self.peek()[0] == FormulaTokenizer.BICONDITIONAL:
            self.consume()
            self.implication()

    def implication(self) -> None:
        self.disjunction()
        # Right-associative: p → q → r = p → (q → r)
        if self.peek()[0] == FormulaTokenizer.IMPLIES:
            self.consume()
            self.implication()

    def disjunction(self) -> None:
        self.conjunction()
        while self.peek()[0] == FormulaTokenizer.OR:
            self.consume()
            self.conjunction()

    def conjunction(self) -> None:
        self.unary()
        while self.peek()[0] == FormulaTokenizer.AND:
            self.consume()
            self.unary()

    def unary(self) -> None:
        if self.peek()[0] == FormulaTokenizer.NOT:
            self.consume()
            self.unary()
        else:
            self.primary()

    def primary(self) -> None:
        token = self.peek()
        if token[0] == FormulaTokenizer.ATOM:
            self.consume()
            self.atoms_found.add(token[1])
        elif token[0] == FormulaTokenizer.LPAREN:
            self.consume()
            self.formula()
            self.consume(FormulaTokenizer.RPAREN)
        else:
            raise FormulaParseError(
                f"Expected atom or '(', got {token[0]} ('{token[1]}') at position {self.pos}"
            )


# ============================================================
# Validation functions
# ============================================================


def validate_formula_syntax(formula: str) -> tuple[bool, set[str]]:
    """Parse a formula and return (is_valid, atoms_found).

    Supports both Unicode (∧, ∨, →, ↔, ¬) and ASCII (&, |, ->, <->, ~).
    """
    try:
        tokenizer = FormulaTokenizer(formula)
        parser = FormulaParser(tokenizer.tokens)
        parser.parse()
        return True, parser.atoms_found
    except FormulaParseError:
        return False, set()


def check_balanced_parens(formula: str) -> bool:
    """Check that parentheses are balanced."""
    depth = 0
    for ch in formula:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth < 0:
            return False
    return depth == 0


def check_atom_consistency(example: dict) -> bool:
    """Check that declared atoms match atoms used in the formula.

    Allows formula to use a subset of declared atoms (some might be intermediate),
    but every atom in the formula must be declared.
    """
    thought = example.get("thought", {})
    declared_atoms = {a["atom"] for a in thought.get("identified_atoms", []) if isinstance(a, dict)}
    if not declared_atoms:
        return False

    formula = example.get("output", {}).get("formula", "")
    _, formula_atoms = validate_formula_syntax(formula)

    if not formula_atoms:
        # Try ASCII formula
        formula_ascii = example.get("output", {}).get("formula_ascii", "")
        _, formula_atoms = validate_formula_syntax(formula_ascii)

    if not formula_atoms:
        return False

    # Every atom in formula must be declared
    undeclared = formula_atoms - declared_atoms
    return len(undeclared) == 0


def normalize_formula(formula: str) -> str:
    """Normalize whitespace and formatting in a formula."""
    # Normalize whitespace around operators
    formula = formula.strip()
    # Collapse multiple spaces
    formula = re.sub(r"\s+", " ", formula)
    # Ensure space around binary operators (Unicode)
    for op in ("∧", "∨", "→", "↔"):
        formula = re.sub(rf"\s*{re.escape(op)}\s*", f" {op} ", formula)
    # Ensure space around binary operators (ASCII)
    formula = re.sub(r"\s*<->\s*", " <-> ", formula)
    formula = re.sub(r"\s*->\s*", " -> ", formula)
    formula = re.sub(r"\s*&\s*", " & ", formula)
    formula = re.sub(r"\s*\|\s*", " | ", formula)
    # Clean up spaces around parens
    formula = re.sub(r"\(\s+", "(", formula)
    formula = re.sub(r"\s+\)", ")", formula)
    # Final cleanup
    formula = re.sub(r"\s+", " ", formula).strip()
    return formula


# ============================================================
# Cleaning pipeline
# ============================================================


@dataclass
class CleaningStats:
    """Track what happened during cleaning."""

    total_input: int = 0
    invalid_syntax: int = 0
    unbalanced_parens: int = 0
    inconsistent_atoms: int = 0
    duplicates_removed: int = 0
    trivial_filtered: int = 0
    total_output: int = 0
    examples_by_complexity: dict[str, int] = field(default_factory=dict)
    examples_by_block: dict[str, int] = field(default_factory=dict)


def deduplicate(examples: list[dict]) -> tuple[list[dict], int]:
    """Remove exact duplicates by (input, formula) pair."""
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    removed = 0

    for ex in examples:
        key = (
            ex.get("natural_language_input", "").strip().lower(),
            ex.get("output", {}).get("formula", "").strip(),
        )
        if key in seen:
            removed += 1
        else:
            seen.add(key)
            unique.append(ex)

    return unique, removed


def filter_trivial(examples: list[dict], max_simple_ratio: float = 0.4) -> tuple[list[dict], int]:
    """Subsample overly simple formulas to maintain diversity.

    A formula is "trivial" if it has only 2 atoms and 1 connector.
    We keep at most max_simple_ratio of the total as trivial.
    """
    trivial: list[dict] = []
    non_trivial: list[dict] = []

    for ex in examples:
        formula = ex.get("output", {}).get("formula", "")
        _, atoms = validate_formula_syntax(formula)
        if len(atoms) <= 2:
            trivial.append(ex)
        else:
            non_trivial.append(ex)

    # How many trivial can we keep?
    total = len(examples)
    max_trivial = int(total * max_simple_ratio)

    if len(trivial) <= max_trivial:
        return examples, 0

    # Keep a random subset
    import random

    random.seed(42)
    kept_trivial = random.sample(trivial, max_trivial)
    filtered = len(trivial) - max_trivial

    return non_trivial + kept_trivial, filtered


def clean_dataset(
    examples: list[dict], *, verbose: bool = True
) -> tuple[list[dict], CleaningStats]:
    """Run the full cleaning pipeline on a list of examples.

    Pipeline:
        1. Validate formula syntax
        2. Check balanced parentheses
        3. Check atom consistency
        4. Deduplicate
        5. Filter trivial
        6. Normalize format
    """
    stats = CleaningStats(total_input=len(examples))

    if verbose:
        console.print(f"\n[bold cyan]🧹 Starting cleaning pipeline[/]")
        console.print(f"   Input: {len(examples)} examples\n")

    # --- Step 1-3: Validate each example ---
    valid_examples: list[dict] = []

    for ex in examples:
        formula = ex.get("output", {}).get("formula", "")
        formula_ascii = ex.get("output", {}).get("formula_ascii", "")

        # Step 1: Validate syntax (try Unicode first, then ASCII)
        is_valid, atoms = validate_formula_syntax(formula)
        if not is_valid:
            is_valid, atoms = validate_formula_syntax(formula_ascii)
            if not is_valid:
                stats.invalid_syntax += 1
                continue

        # Step 2: Balanced parentheses
        if not check_balanced_parens(formula) and not check_balanced_parens(formula_ascii):
            stats.unbalanced_parens += 1
            continue

        # Step 3: Atom consistency
        if not check_atom_consistency(ex):
            stats.inconsistent_atoms += 1
            continue

        # Step 6: Normalize (do it here to help dedup)
        if formula:
            ex["output"]["formula"] = normalize_formula(formula)
        if formula_ascii:
            ex["output"]["formula_ascii"] = normalize_formula(formula_ascii)

        valid_examples.append(ex)

    if verbose:
        console.print(f"   ✅ Syntax valid: {len(valid_examples)}")
        console.print(f"   ❌ Invalid syntax: {stats.invalid_syntax}")
        console.print(f"   ❌ Unbalanced parens: {stats.unbalanced_parens}")
        console.print(f"   ❌ Inconsistent atoms: {stats.inconsistent_atoms}")

    # --- Step 4: Deduplicate ---
    valid_examples, dups = deduplicate(valid_examples)
    stats.duplicates_removed = dups

    if verbose:
        console.print(f"   🔄 Duplicates removed: {dups}")

    # --- Step 5: Filter trivial ---
    valid_examples, trivial_count = filter_trivial(valid_examples)
    stats.trivial_filtered = trivial_count

    if verbose:
        console.print(f"   📉 Trivial subsampled: {trivial_count}")

    # --- Compute stats ---
    stats.total_output = len(valid_examples)
    stats.examples_by_complexity = dict(
        Counter(ex.get("complexity", "unknown") for ex in valid_examples)
    )
    stats.examples_by_block = dict(Counter(ex.get("block", "unknown") for ex in valid_examples))

    if verbose:
        console.print(f"\n   [bold green]✅ Output: {stats.total_output} clean examples[/]")
        retention = stats.total_output / stats.total_input * 100 if stats.total_input > 0 else 0
        console.print(f"   📊 Retention rate: {retention:.1f}%")

    return valid_examples, stats


def print_report(stats: CleaningStats) -> None:
    """Print a detailed cleaning report."""
    console.print("\n[bold]📊 Cleaning Report[/]\n")

    # Summary table
    table = Table(title="Pipeline Summary")
    table.add_column("Step", style="cyan")
    table.add_column("Removed", justify="right", style="red")
    table.add_column("Remaining", justify="right", style="green")

    remaining = stats.total_input
    table.add_row("Input", "-", str(remaining))

    remaining -= stats.invalid_syntax
    table.add_row("1. Invalid syntax", str(stats.invalid_syntax), str(remaining))

    remaining -= stats.unbalanced_parens
    table.add_row("2. Unbalanced parens", str(stats.unbalanced_parens), str(remaining))

    remaining -= stats.inconsistent_atoms
    table.add_row("3. Inconsistent atoms", str(stats.inconsistent_atoms), str(remaining))

    remaining -= stats.duplicates_removed
    table.add_row("4. Duplicates", str(stats.duplicates_removed), str(remaining))

    remaining -= stats.trivial_filtered
    table.add_row("5. Trivial filter", str(stats.trivial_filtered), str(remaining))

    table.add_row("6. Normalize", "0", str(remaining), style="bold")
    console.print(table)

    # Complexity distribution
    console.print("\n[bold]📈 Complexity Distribution:[/]")
    for comp, count in sorted(stats.examples_by_complexity.items()):
        pct = count / stats.total_output * 100 if stats.total_output else 0
        bar = "█" * int(pct / 2)
        console.print(f"   {comp:<15} {count:>5} ({pct:>5.1f}%) {bar}")

    # Block distribution
    console.print("\n[bold]📦 Block Distribution:[/]")
    for block, count in sorted(stats.examples_by_block.items()):
        pct = count / stats.total_output * 100 if stats.total_output else 0
        bar = "█" * int(pct / 2)
        console.print(f"   {block:<30} {count:>5} ({pct:>5.1f}%) {bar}")


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean and validate the raw dataset (Phases 1-6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python data/scripts/clean.py
    python data/scripts/clean.py --input dataset.json
    python data/scripts/clean.py --input data/raw/dataset.json --output data/raw/dataset_clean.json
        """,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="dataset.json",
        help="Input JSON file (default: dataset.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (default: <input>_clean.json)",
    )
    parser.add_argument(
        "--max-trivial-ratio",
        type=float,
        default=0.4,
        help="Maximum ratio of trivial examples to keep (default: 0.4)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[bold red]❌ File not found: {input_path}[/]")
        sys.exit(1)

    # Default output path
    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_clean{input_path.suffix}"
    else:
        output_path = Path(args.output)

    console.print(f"[bold]🔧 Dataset Cleaning Pipeline[/]")
    console.print(f"   Input:  {input_path}")
    console.print(f"   Output: {output_path}")

    # Load
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    examples = data.get("examples", data if isinstance(data, list) else [])
    console.print(f"   Loaded: {len(examples)} examples")

    # Clean
    clean_examples, stats = clean_dataset(examples)

    # Save
    output_data = {
        "examples": clean_examples,
        "cleaning_stats": {
            "total_input": stats.total_input,
            "total_output": stats.total_output,
            "invalid_syntax": stats.invalid_syntax,
            "unbalanced_parens": stats.unbalanced_parens,
            "inconsistent_atoms": stats.inconsistent_atoms,
            "duplicates_removed": stats.duplicates_removed,
            "trivial_filtered": stats.trivial_filtered,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Report
    print_report(stats)
    console.print(f"\n[bold green]✅ Saved to {output_path}[/]")


if __name__ == "__main__":
    main()
