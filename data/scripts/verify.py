"""
Dataset verification with API — Phase 7.

Sends each example to DeepSeek for independent verification:
"Is this formula correct for this natural language statement?"

Features:
    - Incremental saving: saves after EVERY example (crash-safe)
    - Auto-resume: detects where it left off and continues
    - Auto-retry on API errors

Estimated cost: ~$0.50 for 2,000 examples.

Usage:
    python data/scripts/verify.py --input data/raw/dataset_clean.json
    python data/scripts/verify.py --input data/raw/dataset_clean.json --output data/raw/dataset_verified.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

console = Console()

load_dotenv()


VERIFICATION_PROMPT = """Analiza si esta fórmula de lógica proposicional es correcta para el enunciado dado.

ENUNCIADO: {input}

FÓRMULA: {formula}

ÁTOMOS DECLARADOS:
{atoms}

Responde SOLO con un JSON:
{{
    "is_correct": true/false,
    "confidence": 0.0-1.0,
    "issues": ["lista de problemas si hay"],
    "corrected_formula": "fórmula corregida si es necesario, o null"
}}"""


def verify_example(client: OpenAI, example: dict, max_retries: int = 3) -> dict | None:
    """Verify a single example using DeepSeek API.

    Retries up to max_retries times on failure.
    Returns the verification result dict, or None on persistent error.
    """
    nl_input = example.get("natural_language_input", "")
    formula = example.get("output", {}).get("formula", "")
    atoms = example.get("thought", {}).get("identified_atoms", [])

    atoms_str = "\n".join(
        f"  {a['atom']}: {a['definition']}" for a in atoms if isinstance(a, dict) and "atom" in a
    )

    prompt = VERIFICATION_PROMPT.format(input=nl_input, formula=formula, atoms=atoms_str)

    import re

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un verificador experto en lógica proposicional. Sé estricto pero justo.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
                timeout=60,
            )

            text = response.choices[0].message.content.strip()
            # Try to parse JSON
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*\n?", "", text)
                text = re.sub(r"\n?```\s*$", "", text)

            return json.loads(text)

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2**attempt  # exponential backoff: 1s, 2s, 4s
                console.print(
                    f"[yellow]⚠️ Retry {attempt + 1}/{max_retries}: {e} (waiting {wait}s)[/]"
                )
                time.sleep(wait)
            else:
                console.print(f"[yellow]⚠️ Failed after {max_retries} retries: {e}[/]")
                return None


def load_progress(progress_path: Path) -> tuple[list[dict], dict]:
    """Load existing progress file if it exists.

    Returns: (verified_examples, stats)
    """
    if progress_path.exists():
        with open(progress_path, encoding="utf-8") as f:
            data = json.load(f)
        examples = data.get("examples", [])
        stats = data.get("verification_stats", {})
        return examples, stats
    return [], {}


def save_progress(
    progress_path: Path,
    verified: list[dict],
    stats: dict,
) -> None:
    """Save current progress to disk (called after each example)."""
    output_data = {"examples": verified, "verification_stats": stats}
    # Write to temp file first, then rename (atomic write)
    tmp_path = progress_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(progress_path)


def get_verified_keys(verified: list[dict]) -> set[str]:
    """Get a set of keys (input + formula) that have already been verified."""
    keys = set()
    for ex in verified:
        key = (
            ex.get("natural_language_input", "").strip().lower()
            + "|||"
            + ex.get("output", {}).get("formula", "").strip()
        )
        keys.add(key)
    return keys


def verify_dataset(
    examples: list[dict],
    output_path: Path,
    *,
    min_confidence: float = 0.7,
    auto_correct: bool = True,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """Verify all examples with API, saving progress after each one.

    Returns: (verified_examples, stats_dict)
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        console.print("[bold red]❌ DEEPSEEK_API_KEY not found in .env[/]")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # Load existing progress
    verified, existing_stats = load_progress(output_path)
    verified_keys = get_verified_keys(verified)
    already_done = len(verified)

    corrected = existing_stats.get("corrected", 0)
    rejected = existing_stats.get("rejected", 0)
    api_errors = existing_stats.get("api_errors", 0)

    if already_done > 0:
        console.print(f"\n[bold cyan]🔄 Resuming from {already_done}/{len(examples)} examples[/]")

    # Filter out already-processed examples
    pending = []
    for ex in examples:
        key = (
            ex.get("natural_language_input", "").strip().lower()
            + "|||"
            + ex.get("output", {}).get("formula", "").strip()
        )
        if key not in verified_keys:
            pending.append(ex)

    if not pending:
        console.print("[bold green]✅ All examples already verified![/]")
        return verified, existing_stats

    console.print(f"   Pending: {len(pending)} examples\n")

    with Progress(
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
    ) as progress:
        task = progress.add_task("Verifying...", total=len(examples), completed=already_done)

        for ex in pending:
            result = verify_example(client, ex)

            if result is None:
                api_errors += 1
                # Keep the example on API error (benefit of the doubt)
                ex["verification"] = {"verified": False, "api_error": True}
                verified.append(ex)
            elif result.get("is_correct", False):
                ex["verification"] = {
                    "verified": True,
                    "confidence": result.get("confidence", 1.0),
                }
                verified.append(ex)
            elif (
                auto_correct
                and result.get("corrected_formula")
                and result.get("confidence", 0) >= min_confidence
            ):
                # Apply correction
                ex["output"]["formula"] = result["corrected_formula"]
                ex["verification"] = {
                    "verified": True,
                    "corrected": True,
                    "confidence": result.get("confidence", 0),
                    "issues": result.get("issues", []),
                }
                verified.append(ex)
                corrected += 1
            else:
                rejected += 1

            # Save progress after EVERY example
            stats = {
                "total_input": len(examples),
                "verified": len(verified),
                "corrected": corrected,
                "rejected": rejected,
                "api_errors": api_errors,
            }
            save_progress(output_path, verified, stats)

            progress.update(task, advance=1)

            # Rate limiting
            time.sleep(0.5)

    stats = {
        "total_input": len(examples),
        "verified": len(verified),
        "corrected": corrected,
        "rejected": rejected,
        "api_errors": api_errors,
    }

    if verbose:
        console.print(f"\n[bold]📊 Verification Results:[/]")
        console.print(f"   ✅ Verified: {len(verified)}")
        console.print(f"   🔧 Corrected: {corrected}")
        console.print(f"   ❌ Rejected: {rejected}")
        console.print(f"   ⚠️ API errors: {api_errors}")

    return verified, stats


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify dataset examples with DeepSeek API (Phase 7, ~$0.50)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/dataset_clean.json",
        help="Input JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Min confidence to accept corrections (default: 0.7)",
    )
    parser.add_argument(
        "--no-auto-correct",
        action="store_true",
        help="Don't auto-correct, just accept or reject",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[bold red]❌ File not found: {input_path}[/]")
        sys.exit(1)

    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_verified{input_path.suffix}"
    else:
        output_path = Path(args.output)

    console.print(f"[bold]🔍 Dataset Verification Pipeline[/]")
    console.print(f"   Input:  {input_path}")
    console.print(f"   Output: {output_path}")
    console.print(f"   💰 Estimated cost: ~$0.50")

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    examples = data.get("examples", data if isinstance(data, list) else [])

    verified, stats = verify_dataset(
        examples,
        output_path,
        min_confidence=args.min_confidence,
        auto_correct=not args.no_auto_correct,
    )

    # Final save (already saved incrementally, but just in case)
    save_progress(output_path, verified, stats)

    console.print(f"\n[bold green]✅ Saved to {output_path}[/]")


if __name__ == "__main__":
    while True:
        try:
            main()
            break  # Finished OK, exit
        except KeyboardInterrupt:
            console.print("\n[bold yellow]⏸️  Interrupted. Progress saved. Run again to resume.[/]")
            break  # User intentionally stopped, exit
        except Exception as e:
            console.print(f"\n[bold red]💥 Crash: {e}[/]")
            console.print("[bold cyan]🔄 Auto-restarting in 10 seconds... (progress saved)[/]")
            time.sleep(10)
            console.print("[bold cyan]🔄 Restarting...[/]\n")
