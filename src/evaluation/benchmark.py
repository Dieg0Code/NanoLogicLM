"""
benchmark.py — El microscopio de NanoLogic.

Desglosa el rendimiento del modelo por TODAS las dimensiones
para saber EXACTAMENTE donde falla y que mejorar.

Dimensiones de evaluacion:
    1. Por Complejidad: Simple / Intermediate / Advanced
    2. Por Conector: ∧, ∨, →, ↔, ¬
    3. Por Bloque Tematico: causal, temporal, normativo, etc.
    4. Por Largo de Formula: 1-2, 3-4, 5+ conectores
    5. Confusion Matrix de Conectores: ¿con que conector lo confunde?
    6. Scaling Analysis: ¿como escala con el numero de atomos?

Uso:
    report = run_benchmark(model, tokenizer, test_data)
    print(report)
    report.save("benchmarks/run_001.json")
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.evaluation.metrics import (
    evaluate_example,
    ExampleResult,
    extract_formula_from_sequence,
    get_main_connector,
)
from src.evaluation.truth_table import (
    FormulaParser,
    extract_atoms,
    extract_connectors as tt_extract_connectors,
)


# =====================================================================
# TIPOS DE DATOS
# =====================================================================


@dataclass
class CategoryResult:
    """Resultado de una categoria (ej: Simple, o conector ∧).

    Acumula metricas SOLO para los ejemplos de esa categoria.
    """

    name: str
    total: int = 0
    n_exact: int = 0
    n_semantic: int = 0
    n_valid: int = 0
    total_partial: float = 0.0
    total_atom_f1: float = 0.0
    total_connector_acc: float = 0.0
    total_comp_avg: float = 0.0

    @property
    def exact_match(self) -> float:
        return self.n_exact / self.total if self.total > 0 else 0.0

    @property
    def semantic_accuracy(self) -> float:
        return self.n_semantic / self.total if self.total > 0 else 0.0

    @property
    def syntax_valid_rate(self) -> float:
        return self.n_valid / self.total if self.total > 0 else 0.0

    @property
    def partial_credit(self) -> float:
        return self.total_partial / self.total if self.total > 0 else 0.0

    @property
    def atom_f1(self) -> float:
        return self.total_atom_f1 / self.total if self.total > 0 else 0.0

    @property
    def connector_accuracy(self) -> float:
        return self.total_connector_acc / self.total if self.total > 0 else 0.0

    @property
    def compositional_avg(self) -> float:
        return self.total_comp_avg / self.total if self.total > 0 else 0.0

    def add_example(self, ex: ExampleResult) -> None:
        """Agrega un ejemplo evaluado a esta categoria."""
        self.total += 1
        if ex.is_exact_match:
            self.n_exact += 1
        if ex.is_semantic_match is True:
            self.n_semantic += 1
        if ex.is_syntax_valid:
            self.n_valid += 1
        self.total_partial += ex.partial_credit_score
        self.total_atom_f1 += ex.atom_f1_score
        self.total_connector_acc += ex.connector_acc
        self.total_comp_avg += ex.compositional.average

    def to_dict(self) -> dict:
        """Serializa a dict para export JSON."""
        return {
            "name": self.name,
            "total": self.total,
            "exact_match": round(self.exact_match, 4),
            "semantic_accuracy": round(self.semantic_accuracy, 4),
            "syntax_valid_rate": round(self.syntax_valid_rate, 4),
            "partial_credit": round(self.partial_credit, 4),
            "atom_f1": round(self.atom_f1, 4),
            "connector_accuracy": round(self.connector_accuracy, 4),
            "compositional_avg": round(self.compositional_avg, 4),
        }

    def format_row(self) -> str:
        """Formatea como fila de tabla para impresion."""
        bar = "█" * int(self.semantic_accuracy * 20)
        bar = bar.ljust(20, "░")
        return (
            f"  {self.name:<20s} "
            f"{self.semantic_accuracy:>6.1%}  "
            f"{bar}  "
            f"(n={self.total}, exact={self.exact_match:.1%}, "
            f"partial={self.partial_credit:.2f})"
        )


# =====================================================================
# CONFUSION MATRIX DE CONECTORES
# =====================================================================


class ConnectorConfusionMatrix:
    """Matriz de confusion para conectores logicos.

    Registra que conector predijo el modelo vs cual era el correcto.
    """

    CONNECTORS = ["∧", "∨", "→", "↔", "¬"]

    def __init__(self) -> None:
        # matrix[real][predicted] = count
        self.matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.total = 0

    def add(self, real: str | None, predicted: str | None) -> None:
        """Registra una observacion: conector real vs predicho."""
        if real is None or predicted is None:
            return
        if real in self.CONNECTORS and predicted in self.CONNECTORS:
            self.matrix[real][predicted] += 1
            self.total += 1

    def to_dict(self) -> dict:
        """Serializa para JSON."""
        return {real: dict(preds) for real, preds in self.matrix.items()}

    def __str__(self) -> str:
        if not self.matrix:
            return "  (sin datos de conectores)"

        lines = []
        lines.append("  Confusion Matrix de Conectores:")
        lines.append("")

        # Header
        header = "         " + "".join(f" {c:>5s}" for c in self.CONNECTORS)
        lines.append(header)
        lines.append("        " + "─" * (6 * len(self.CONNECTORS) + 1))

        # Rows
        for real in self.CONNECTORS:
            row_data = self.matrix.get(real, {})
            if not row_data and real not in [r for r in self.matrix]:
                continue
            cells = []
            for pred in self.CONNECTORS:
                count = row_data.get(pred, 0)
                if count > 0 and pred == real:
                    cells.append(f" {count:>4d}✓")
                elif count > 0:
                    cells.append(f" {count:>4d}✗")
                else:
                    cells.append("     ·")
            lines.append(f"  {real:>4s}  │{''.join(cells)}")

        # Insights: top confusions
        confusions = []
        for real, preds in self.matrix.items():
            for pred, count in preds.items():
                if real != pred and count > 0:
                    confusions.append((count, real, pred))
        confusions.sort(reverse=True)

        if confusions:
            lines.append("")
            lines.append("  Top confusiones:")
            for count, real, pred in confusions[:5]:
                lines.append(f"    {real} → {pred}: {count} veces")

        return "\n".join(lines)


# =====================================================================
# BENCHMARK REPORT
# =====================================================================


@dataclass
class BenchmarkReport:
    """Reporte completo de benchmark con todos los desgloses.

    Contiene resultados por cada dimension de evaluacion
    y metodos para imprimir y exportar.
    """

    # Resultados por dimension
    by_complexity: dict[str, CategoryResult] = field(default_factory=dict)
    by_connector: dict[str, CategoryResult] = field(default_factory=dict)
    by_block: dict[str, CategoryResult] = field(default_factory=dict)
    by_length: dict[str, CategoryResult] = field(default_factory=dict)
    by_n_atoms: dict[str, CategoryResult] = field(default_factory=dict)

    # Confusion matrix
    confusion: ConnectorConfusionMatrix = field(default_factory=ConnectorConfusionMatrix)

    # Global
    global_result: CategoryResult = field(default_factory=lambda: CategoryResult(name="Global"))

    # Total de ejemplos
    total: int = 0

    def save(self, path: str | Path) -> None:
        """Guarda el reporte como JSON.

        Args:
            path: Ruta del archivo JSON de salida.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "total": self.total,
            "global": self.global_result.to_dict(),
            "by_complexity": {k: v.to_dict() for k, v in self.by_complexity.items()},
            "by_connector": {k: v.to_dict() for k, v in self.by_connector.items()},
            "by_block": {k: v.to_dict() for k, v in self.by_block.items()},
            "by_length": {k: v.to_dict() for k, v in self.by_length.items()},
            "by_n_atoms": {k: v.to_dict() for k, v in self.by_n_atoms.items()},
            "confusion_matrix": self.confusion.to_dict(),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def __str__(self) -> str:
        sections = []

        sections.append("=" * 70)
        sections.append(f"  BENCHMARK NANOLOGIC ({self.total} ejemplos)")
        sections.append("=" * 70)

        # Global
        sections.append("")
        sections.append(self.global_result.format_row())

        # By Complexity
        if self.by_complexity:
            sections.append("")
            sections.append("─" * 70)
            sections.append("  📊 Por Complejidad:")
            for key in ["simple", "intermediate", "advanced"]:
                if key in self.by_complexity:
                    sections.append(self.by_complexity[key].format_row())

        # By Connector
        if self.by_connector:
            sections.append("")
            sections.append("─" * 70)
            sections.append("  📊 Por Conector:")
            for cat in sorted(
                self.by_connector.values(),
                key=lambda c: c.semantic_accuracy,
                reverse=True,
            ):
                sections.append(cat.format_row())

        # By Block
        if self.by_block:
            sections.append("")
            sections.append("─" * 70)
            sections.append("  📊 Por Bloque Temático:")
            for cat in sorted(
                self.by_block.values(),
                key=lambda c: c.semantic_accuracy,
                reverse=True,
            ):
                sections.append(cat.format_row())

        # By Length
        if self.by_length:
            sections.append("")
            sections.append("─" * 70)
            sections.append("  📊 Por Largo de Fórmula:")
            for key in ["1-2 conectores", "3-4 conectores", "5+ conectores"]:
                if key in self.by_length:
                    sections.append(self.by_length[key].format_row())

        # By N Atoms (Scaling Analysis)
        if self.by_n_atoms:
            sections.append("")
            sections.append("─" * 70)
            sections.append("  📊 Scaling Analysis (por # de átomos):")
            for key in sorted(self.by_n_atoms.keys()):
                sections.append(self.by_n_atoms[key].format_row())

        # Confusion Matrix
        if self.confusion.total > 0:
            sections.append("")
            sections.append("─" * 70)
            sections.append(str(self.confusion))

        # Diagnostico
        sections.append("")
        sections.append("─" * 70)
        sections.append(_generate_diagnosis(self))

        sections.append("=" * 70)
        return "\n".join(sections)


# =====================================================================
# HELPERS
# =====================================================================


def _count_connectors_in_formula(formula: str) -> int:
    """Cuenta el numero de conectores en una formula."""
    try:
        ast = FormulaParser(formula).parse()
        connectors = tt_extract_connectors(ast)
        return len(connectors)
    except ValueError:
        return 0


def _count_atoms_in_formula(formula: str) -> int:
    """Cuenta el numero de atomos unicos en una formula."""
    try:
        ast = FormulaParser(formula).parse()
        return len(extract_atoms(ast))
    except ValueError:
        return 0


def _get_length_bucket(n_connectors: int) -> str:
    """Clasifica el largo de una formula por numero de conectores."""
    if n_connectors <= 2:
        return "1-2 conectores"
    elif n_connectors <= 4:
        return "3-4 conectores"
    else:
        return "5+ conectores"


def _get_main_connector_label(formula: str) -> str | None:
    """Extrae el conector principal de una formula como label."""
    try:
        ast = FormulaParser(formula).parse()
        return get_main_connector(ast)
    except ValueError:
        return None


def _generate_diagnosis(report: BenchmarkReport) -> str:
    """Genera diagnostico automatico basado en los resultados.

    Identifica las areas de mayor debilidad y sugiere acciones.
    """
    lines = ["  🔍 DIAGNÓSTICO AUTOMÁTICO:"]

    issues = []

    # Analizar por complejidad
    if "advanced" in report.by_complexity:
        adv = report.by_complexity["advanced"]
        if adv.semantic_accuracy < 0.5:
            issues.append(
                f"  ⚠️  Advanced accuracy baja ({adv.semantic_accuracy:.0%}). "
                f"Necesita mas epochs o mas datos advanced."
            )

    # Analizar conectores debiles
    for name, cat in sorted(
        report.by_connector.items(),
        key=lambda x: x[1].semantic_accuracy,
    ):
        if cat.semantic_accuracy < 0.7 and cat.total >= 5:
            issues.append(
                f"  ⚠️  Conector '{name}' accuracy baja ({cat.semantic_accuracy:.0%}, n={cat.total}). "
                f"Necesita mas datos con este conector."
            )

    # Analizar scaling
    atom_results = sorted(
        report.by_n_atoms.items(),
        key=lambda x: x[0],
    )
    if len(atom_results) >= 2:
        first = atom_results[0][1]
        last = atom_results[-1][1]
        drop = first.semantic_accuracy - last.semantic_accuracy
        if drop > 0.3:
            issues.append(
                f"  ⚠️  Caida de accuracy fuerte con mas atomos "
                f"({first.semantic_accuracy:.0%} → {last.semantic_accuracy:.0%}). "
                f"El modelo tiene techo de composicionalidad."
            )

    # Analizar confusion matrix
    top_confusions = []
    for real, preds in report.confusion.matrix.items():
        for pred, count in preds.items():
            if real != pred and count >= 3:
                top_confusions.append((count, real, pred))
    top_confusions.sort(reverse=True)

    if top_confusions:
        count, real, pred = top_confusions[0]
        issues.append(
            f"  ⚠️  Confunde '{real}' con '{pred}' ({count} veces). "
            f"Necesita datos que distingan estos conectores."
        )

    if not issues:
        lines.append("  ✅ El modelo funciona bien en todas las dimensiones.")
    else:
        lines.extend(issues)

    return "\n".join(lines)


# =====================================================================
# RUN BENCHMARK
# =====================================================================


@dataclass
class BenchmarkExample:
    """Un ejemplo para benchmark con metadata.

    Contiene la prediccion, referencia, y metadata del JSONL
    (complexity, block, etc.) para hacer el desglose.
    """

    prediction: str  # formula generada por el modelo
    reference: str  # formula ground truth
    complexity: str = ""  # "simple", "intermediate", "advanced"
    block: str = ""  # "causal", "temporal", etc.


def run_benchmark(examples: list[BenchmarkExample]) -> BenchmarkReport:
    """Ejecuta el benchmark completo sobre una lista de ejemplos.

    Para cada ejemplo:
        1. Evalua con TODAS las metricas de metrics.py
        2. Clasifica en cada dimension (complejidad, conector, etc.)
        3. Acumula resultados por categoria
        4. Construye confusion matrix
        5. Calcula scaling analysis

    Args:
        examples: Lista de BenchmarkExample con predicciones y metadata.

    Returns:
        BenchmarkReport con todos los desgloses.

    Ejemplo:
        examples = [
            BenchmarkExample(
                prediction="p → q",
                reference="p → q",
                complexity="simple",
                block="causal",
            ),
            ...
        ]
        report = run_benchmark(examples)
        print(report)
        report.save("benchmarks/run_001.json")
    """
    report = BenchmarkReport(total=len(examples))

    for ex in examples:
        # Evaluar con todas las metricas
        result = evaluate_example(ex.prediction, ex.reference)

        # === Global ===
        report.global_result.add_example(result)

        # === Por Complejidad ===
        if ex.complexity:
            key = ex.complexity.lower()
            if key not in report.by_complexity:
                report.by_complexity[key] = CategoryResult(name=key.capitalize())
            report.by_complexity[key].add_example(result)

        # === Por Bloque Tematico ===
        if ex.block:
            key = ex.block.lower()
            if key not in report.by_block:
                report.by_block[key] = CategoryResult(name=key.capitalize())
            report.by_block[key].add_example(result)

        # === Por Conector Principal (de la referencia) ===
        ref_connector = _get_main_connector_label(ex.reference)
        if ref_connector:
            if ref_connector not in report.by_connector:
                report.by_connector[ref_connector] = CategoryResult(name=ref_connector)
            report.by_connector[ref_connector].add_example(result)

        # === Por Largo de Formula (de la referencia) ===
        n_conn = _count_connectors_in_formula(ex.reference)
        length_bucket = _get_length_bucket(n_conn)
        if length_bucket not in report.by_length:
            report.by_length[length_bucket] = CategoryResult(name=length_bucket)
        report.by_length[length_bucket].add_example(result)

        # === Scaling Analysis: por numero de atomos ===
        n_atoms = _count_atoms_in_formula(ex.reference)
        atom_key = f"{n_atoms} átomos"
        if atom_key not in report.by_n_atoms:
            report.by_n_atoms[atom_key] = CategoryResult(name=atom_key)
        report.by_n_atoms[atom_key].add_example(result)

        # === Confusion Matrix ===
        pred_connector = _get_main_connector_label(ex.prediction)
        report.confusion.add(ref_connector, pred_connector)

    return report
