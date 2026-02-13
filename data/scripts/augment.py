"""
Data augmentation pipeline — Fases 8-12.

Creates additional training examples through logical equivalences and transformations.

Phase 8-10  (Python puro, $0):
    - Logical equivalences: p → q ↔ ¬p ∨ q, De Morgan, double negation
    - Commutativity: p ∧ q → q ∧ p
    - Composition: combine 2 simple examples → 1 advanced

Phase 11-12 (API, ~$1.80):
    - Paraphrase: 2 rewordings of top 1,000 examples (same formula, different NL)
    - Adversarial: designed to confuse (false friends, tricky negations, scope ambiguity)

Usage:
    python data/scripts/augment.py --input data/raw/dataset_verified.json
    python data/scripts/augment.py --input data/raw/dataset_verified.json --with-api
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from pathlib import Path

from rich.console import Console

console = Console()

random.seed(42)


# ============================================================
# Logical Equivalences (Phase 8)
# ============================================================


def apply_implication_equivalence(formula: str) -> str | None:
    """p → q  ≡  ¬p ∨ q

    Finds the first implication and converts it.
    """
    # Simple pattern: A → B  (no nested implications)
    match = re.match(r"^(.+?)\s*→\s*(.+)$", formula)
    if match:
        a, b = match.group(1).strip(), match.group(2).strip()
        return f"¬({a}) ∨ ({b})"
    return None


def apply_de_morgan_and(formula: str) -> str | None:
    """¬(p ∧ q)  ≡  ¬p ∨ ¬q"""
    match = re.search(r"¬\((.+?)\s*∧\s*(.+?)\)", formula)
    if match:
        a, b = match.group(1).strip(), match.group(2).strip()
        replacement = f"(¬{a} ∨ ¬{b})"
        return formula[: match.start()] + replacement + formula[match.end() :]
    return None


def apply_de_morgan_or(formula: str) -> str | None:
    """¬(p ∨ q)  ≡  ¬p ∧ ¬q"""
    match = re.search(r"¬\((.+?)\s*∨\s*(.+?)\)", formula)
    if match:
        a, b = match.group(1).strip(), match.group(2).strip()
        replacement = f"(¬{a} ∧ ¬{b})"
        return formula[: match.start()] + replacement + formula[match.end() :]
    return None


def apply_double_negation(formula: str) -> str | None:
    """¬¬p  ≡  p"""
    match = re.search(r"¬¬(\w+)", formula)
    if match:
        atom = match.group(1)
        return formula[: match.start()] + atom + formula[match.end() :]
    return None


def add_double_negation(formula: str) -> str | None:
    """p  ≡  ¬¬p  (add double negation to a random atom)."""
    atoms = re.findall(r"(?<![¬\w])([a-z]\d*)", formula)
    if not atoms:
        return None
    target = random.choice(atoms)
    # Replace first occurrence only
    return formula.replace(target, f"¬¬{target}", 1)


# ============================================================
# Commutativity (Phase 9)
# ============================================================


def apply_commutativity(formula: str) -> str | None:
    """p ∧ q → q ∧ p  or  p ∨ q → q ∨ p"""
    # Find a binary operation
    for op in ("∧", "∨", "↔"):
        # Simple pattern: (A op B) where A and B are atoms or simple groups
        pattern = rf"\(([^()]+?)\s*{re.escape(op)}\s*([^()]+?)\)"
        match = re.search(pattern, formula)
        if match:
            a, b = match.group(1).strip(), match.group(2).strip()
            replacement = f"({b} {op} {a})"
            return formula[: match.start()] + replacement + formula[match.end() :]
    return None


# ============================================================
# Composition (Phase 10)
# ============================================================


def compose_examples(ex1: dict, ex2: dict) -> dict | None:
    """Combine two simple examples into one advanced example.

    Strategy: Connect their formulas with a random connective.
    """
    f1 = ex1.get("output", {}).get("formula", "")
    f2 = ex2.get("output", {}).get("formula", "")
    nl1 = ex1.get("natural_language_input", "")
    nl2 = ex2.get("natural_language_input", "")

    if not f1 or not f2 or not nl1 or not nl2:
        return None

    # Remap atoms in example 2 to avoid collision
    atoms1 = set(re.findall(r"(?<![¬\w])([a-z]\d*)", f1))
    atoms2 = set(re.findall(r"(?<![¬\w])([a-z]\d*)", f2))

    if atoms1 & atoms2:
        # Need to remap — use higher letters
        max_atom = max(ord(a[0]) for a in atoms1) if atoms1 else ord("p")
        remap = {}
        next_atom = chr(max_atom + 1)
        for atom in sorted(atoms2):
            if atom in atoms1:
                remap[atom] = next_atom
                next_atom = chr(ord(next_atom) + 1)
                if ord(next_atom) > ord("z"):
                    return None  # Ran out of atoms

        for old, new in remap.items():
            f2 = re.sub(rf"(?<![¬\w]){old}(?!\d)", new, f2)

    connector = random.choice(["∧", "∨", "→"])
    nl_connectors = {
        "∧": random.choice(["Además, ", "Y al mismo tiempo, ", "Junto con eso, "]),
        "∨": random.choice(["O alternativamente, ", "Ya sea eso o ", "O bien, "]),
        "→": random.choice(
            ["Lo cual implica que ", "Como consecuencia, ", "Si eso se cumple, entonces "]
        ),
    }

    composed = {
        "natural_language_input": f"{nl1}. {nl_connectors[connector].lower()}{nl2[0].lower()}{nl2[1:]}",
        "complexity": "advanced",
        "thought": {
            "reasoning_steps": [
                {"step": 1, "explanation": "Composición de dos proposiciones"},
            ],
            "identified_atoms": (
                ex1.get("thought", {}).get("identified_atoms", [])
                + ex2.get("thought", {}).get("identified_atoms", [])
            ),
            "identified_connectors": (
                ex1.get("thought", {}).get("identified_connectors", [])
                + [{"connector": connector, "natural_language_cue": nl_connectors[connector]}]
                + ex2.get("thought", {}).get("identified_connectors", [])
            ),
        },
        "output": {
            "formula": f"({f1}) {connector} ({f2})",
            "formula_ascii": "",  # Will need regeneration
        },
        "block": "🧩 Composed",
        "augmented": True,
        "augmentation_type": "composition",
    }
    return composed


# ============================================================
# Augmentation pipeline
# ============================================================

EQUIVALENCE_TRANSFORMS = [
    ("implication→disjunction", apply_implication_equivalence),
    ("de_morgan_and", apply_de_morgan_and),
    ("de_morgan_or", apply_de_morgan_or),
    ("double_negation_remove", apply_double_negation),
    ("double_negation_add", add_double_negation),
    ("commutativity", apply_commutativity),
]


def augment_single(example: dict) -> list[dict]:
    """Generate augmented variants of a single example."""
    augmented: list[dict] = []
    formula = example.get("output", {}).get("formula", "")

    for name, transform_fn in EQUIVALENCE_TRANSFORMS:
        result = transform_fn(formula)
        if result and result != formula:
            new_ex = copy.deepcopy(example)
            new_ex["output"]["formula"] = result
            new_ex["augmented"] = True
            new_ex["augmentation_type"] = name
            # Mark that NL stays the same but formula is equivalent
            new_ex["output"]["formula_ascii"] = ""  # Needs regeneration
            augmented.append(new_ex)

    return augmented


def augment_dataset(
    examples: list[dict],
    *,
    max_augmented_per_example: int = 2,
    max_compositions: int = 500,
    verbose: bool = True,
) -> list[dict]:
    """Run the full augmentation pipeline (Phases 8-10).

    Returns: original examples + augmented examples.
    """
    if verbose:
        console.print(f"\n[bold cyan]🔄 Starting augmentation pipeline[/]")
        console.print(f"   Input: {len(examples)} examples\n")

    augmented_all: list[dict] = []

    # Phase 8-9: Equivalences + Commutativity
    for ex in examples:
        variants = augment_single(ex)
        # Keep at most N variants per example
        if len(variants) > max_augmented_per_example:
            variants = random.sample(variants, max_augmented_per_example)
        augmented_all.extend(variants)

    if verbose:
        console.print(f"   ✅ Equivalences + Commutativity: +{len(augmented_all)} examples")

    # Phase 10: Composition
    simple_examples = [ex for ex in examples if ex.get("complexity") in ("simple", "intermediate")]
    compositions: list[dict] = []

    if len(simple_examples) >= 2:
        pairs = min(max_compositions, len(simple_examples) // 2)
        shuffled = random.sample(simple_examples, min(pairs * 2, len(simple_examples)))

        for i in range(0, len(shuffled) - 1, 2):
            composed = compose_examples(shuffled[i], shuffled[i + 1])
            if composed:
                compositions.append(composed)

    if verbose:
        console.print(f"   ✅ Compositions: +{len(compositions)} examples")

    result = examples + augmented_all + compositions

    if verbose:
        console.print(f"\n   [bold green]✅ Total: {len(result)} examples[/]")
        console.print(
            f"   📊 Original: {len(examples)} | "
            f"Equivalences: {len(augmented_all)} | "
            f"Compositions: {len(compositions)}"
        )

    return result


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment dataset with logical equivalences (Phases 8-10)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/dataset_verified.json",
        help="Input JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (default: <input>_augmented.json)",
    )
    parser.add_argument(
        "--max-per-example",
        type=int,
        default=2,
        help="Max augmented variants per example (default: 2)",
    )
    parser.add_argument(
        "--max-compositions",
        type=int,
        default=500,
        help="Max composed examples to generate (default: 500)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[bold red]❌ File not found: {input_path}[/]")
        sys.exit(1)

    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_augmented{input_path.suffix}"
    else:
        output_path = Path(args.output)

    console.print(f"[bold]🔄 Data Augmentation Pipeline[/]")
    console.print(f"   Input:  {input_path}")
    console.print(f"   Output: {output_path}")

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    examples = data.get("examples", data if isinstance(data, list) else [])

    result = augment_dataset(
        examples,
        max_augmented_per_example=args.max_per_example,
        max_compositions=args.max_compositions,
    )

    output_data = {
        "examples": result,
        "augmentation_stats": {
            "original": len(examples),
            "total": len(result),
            "added": len(result) - len(examples),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    console.print(f"\n[bold green]✅ Saved to {output_path}[/]")


if __name__ == "__main__":
    main()
