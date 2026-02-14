"""
metrics.py — Panel de metricas para evaluar NanoLogic.

Centraliza TODAS las metricas de evaluacion en un solo lugar.
Se usa despues de generar formulas con el modelo para medir
que tan bien lo hizo.

Metricas implementadas:
    1. Exact Match: ¿texto identico al ground truth?
    2. Semantic Accuracy: ¿logicamente equivalente? (tabla de verdad)
    3. Normalized Match: ¿igual despues de canonizar? (forma normal)
    4. Syntax Valid Rate: ¿la formula generada es parseable?
    5. Atom F1: ¿identifico los atomos correctos?
    6. Connector Accuracy: ¿uso los conectores correctos?
    7. Partial Credit: distancia de edicion entre ASTs (0-1)
    8. Compositional Score: evaluacion por capas (atomos → subfórmulas → conector → total)

Flujo tipico:
    results = evaluate_batch(predictions, references)
    print(results)
    # {
    #     "exact_match": 0.72,
    #     "semantic_accuracy": 0.80,
    #     "normalized_match": 0.76,
    #     "syntax_valid_rate": 0.85,
    #     "atom_f1": 0.91,
    #     "connector_accuracy": 0.87,
    #     "partial_credit": 0.82,
    # }
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.evaluation.truth_table import (
    ASTNode,
    AtomNode,
    BinaryNode,
    FormulaParser,
    NotNode,
    TokenType,
    are_equivalent,
    evaluate as eval_formula,
    extract_atoms,
    is_valid_formula,
    normalize_formula,
    _ast_to_string,
)


# =====================================================================
# EXTRACCION DE FORMULA DESDE SECUENCIA GENERADA
# =====================================================================
# El modelo genera secuencias completas con special tokens.
# Necesitamos extraer solo la parte de la formula.
#
# Secuencia: "<|bos|><|input|>... <|formula|> p → q <|eos|>"
# Queremos:  "p → q"


def extract_formula_from_sequence(sequence: str) -> str | None:
    """Extrae la formula de una secuencia generada.

    Busca el contenido entre <|formula|> y <|eos|> (o final del string).

    Args:
        sequence: Secuencia completa generada por el modelo.

    Returns:
        La formula extraida, o None si no se encuentra.

    Ejemplo:
        extract_formula_from_sequence(
            "<|bos|><|input|> Si llueve <|formula|> p → q <|eos|>"
        )
        → "p → q"
    """
    # Buscar entre <|formula|> y <|eos|> (o final)
    pattern = r"<\|formula\|>\s*(.*?)(?:\s*<\|eos\|>|$)"
    match = re.search(pattern, sequence, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# =====================================================================
# EXTRACCION DE CONECTORES
# =====================================================================
# Para medir connector accuracy necesitamos extraer los conectores
# del AST y compararlos.


CONNECTOR_SYMBOLS = {"∧", "∨", "→", "↔", "¬"}


def extract_connectors(node: ASTNode) -> list[str]:
    """Extrae la lista de conectores de un AST (in-order traversal).

    Args:
        node: Raiz del AST.

    Returns:
        Lista de conectores en orden de aparicion.

    Ejemplo:
        ast = parse("(p ∧ q) → ¬r")
        extract_connectors(ast)  → ["∧", "→", "¬"]
    """
    connector_map = {
        "AND": "∧",
        "OR": "∨",
        "IMPLIES": "→",
        "BICONDITIONAL": "↔",
    }

    connectors: list[str] = []

    if isinstance(node, NotNode):
        connectors.append("¬")
        connectors.extend(extract_connectors(node.operand))

    elif isinstance(node, BinaryNode):
        connectors.extend(extract_connectors(node.left))
        connectors.append(connector_map.get(node.operator.name, "?"))
        connectors.extend(extract_connectors(node.right))

    return connectors


# =====================================================================
# METRICAS INDIVIDUALES
# =====================================================================


def exact_match(prediction: str, reference: str) -> bool:
    """¿La prediccion es identica al ground truth?

    Normaliza espacios antes de comparar.
    """
    pred = " ".join(prediction.strip().split())
    ref = " ".join(reference.strip().split())
    return pred == ref


def semantic_match(prediction: str, reference: str) -> bool | None:
    """¿La prediccion es logicamente equivalente al ground truth?

    Uses truth table comparison.

    Returns:
        True si equivalentes, False si no, None si alguna es invalida.
    """
    valid_pred, _ = is_valid_formula(prediction)
    valid_ref, _ = is_valid_formula(reference)

    if not valid_pred or not valid_ref:
        return None

    try:
        return are_equivalent(prediction, reference)
    except (ValueError, RecursionError):
        return None


def normalized_match(prediction: str, reference: str) -> bool:
    """¿Son iguales despues de normalizar a forma canonica?

    Mas rapido que tabla de verdad, captura equivalencias triviales.
    """
    norm_pred = normalize_formula(prediction)
    norm_ref = normalize_formula(reference)
    return norm_pred == norm_ref


def atom_f1(prediction: str, reference: str) -> dict[str, float]:
    """F1 Score de atomos: ¿identifico las variables correctas?

    Returns:
        Dict con precision, recall, f1.

    Ejemplo:
        atom_f1("p ∧ q → s", "p ∧ q → r")
        → {"precision": 0.67, "recall": 0.67, "f1": 0.67}
    """
    try:
        pred_ast = FormulaParser(prediction).parse()
        ref_ast = FormulaParser(reference).parse()
    except ValueError:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_atoms = extract_atoms(pred_ast)
    ref_atoms = extract_atoms(ref_ast)

    if not pred_atoms and not ref_atoms:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if not pred_atoms or not ref_atoms:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    correct = pred_atoms & ref_atoms
    precision = len(correct) / len(pred_atoms)
    recall = len(correct) / len(ref_atoms)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def connector_accuracy(prediction: str, reference: str) -> float:
    """¿Uso los conectores correctos?

    Compara la lista de conectores (in-order) entre prediccion y referencia.
    Retorna el % de conectores correctos.

    Ejemplo:
        connector_accuracy("p ∧ q → r", "p ∨ q → r")
        Pred connectors: [∧, →]
        Ref connectors:  [∨, →]
        Match: 1/2 = 0.5
    """
    try:
        pred_ast = FormulaParser(prediction).parse()
        ref_ast = FormulaParser(reference).parse()
    except ValueError:
        return 0.0

    pred_conn = extract_connectors(pred_ast)
    ref_conn = extract_connectors(ref_ast)

    if not ref_conn:
        return 1.0 if not pred_conn else 0.0

    # Comparar usando LCS (longest common subsequence) para ser mas
    # robusto a reordenamientos por conmutatividad
    lcs_len = _lcs_length(pred_conn, ref_conn)
    return lcs_len / len(ref_conn)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest Common Subsequence (programacion dinamica)."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


# =====================================================================
# PARTIAL CREDIT: distancia de edicion entre ASTs
# =====================================================================
# Calcula que tan "cerca" esta la prediccion del ground truth
# basandose en la estructura del arbol (no en el texto).


def _count_nodes(node: ASTNode) -> int:
    """Cuenta el numero de nodos en un AST."""
    if isinstance(node, AtomNode):
        return 1
    if isinstance(node, NotNode):
        return 1 + _count_nodes(node.operand)
    if isinstance(node, BinaryNode):
        return 1 + _count_nodes(node.left) + _count_nodes(node.right)
    return 0


def _tree_edit_distance(a: ASTNode | None, b: ASTNode | None) -> int:
    """Calcula la distancia de edicion simplificada entre dos ASTs.

    Operaciones:
        - Insertar nodo: costo 1
        - Eliminar nodo: costo 1
        - Reemplazar nodo: costo 1 (si son diferentes)

    Esta es una version simplificada del Tree Edit Distance de Zhang-Shasha.
    No es optima pero es suficiente para nuestro uso.
    """
    # Caso base: uno o ambos son None
    if a is None and b is None:
        return 0
    if a is None:
        return _count_nodes(b)  # type: ignore
    if b is None:
        return _count_nodes(a)

    # Ambos son atomos
    if isinstance(a, AtomNode) and isinstance(b, AtomNode):
        return 0 if a.name == b.name else 1

    # Ambos son negaciones
    if isinstance(a, NotNode) and isinstance(b, NotNode):
        return _tree_edit_distance(a.operand, b.operand)

    # Ambos son operaciones binarias
    if isinstance(a, BinaryNode) and isinstance(b, BinaryNode):
        # Costo de cambiar el operador
        op_cost = 0 if a.operator == b.operator else 1

        # Comparar hijos en ambos ordenes (para conmutatividad)
        direct = _tree_edit_distance(a.left, b.left) + _tree_edit_distance(a.right, b.right)

        # Solo intentar orden invertido para operadores conmutativos
        if a.operator == b.operator and a.operator.name in ("AND", "OR", "BICONDITIONAL"):
            swapped = _tree_edit_distance(a.left, b.right) + _tree_edit_distance(a.right, b.left)
            return op_cost + min(direct, swapped)

        return op_cost + direct

    # Tipos diferentes: costo de eliminar uno e insertar otro
    return _count_nodes(a) + _count_nodes(b)


def partial_credit(prediction: str, reference: str) -> float:
    """Calcula el credito parcial basado en distancia de edicion de ASTs.

    Retorna un valor entre 0.0 (completamente diferente) y 1.0 (identico).

    Ejemplo:
        partial_credit("p ∧ q → r", "p ∧ q → r")    → 1.0
        partial_credit("p ∧ q → r", "p ∧ q → s")    → 0.8 (1 nodo diferente de 5)
        partial_credit("p", "(p ∧ q) → (r ∨ s)")     → ~0.1 (muy diferente)
    """
    try:
        pred_ast = FormulaParser(prediction).parse()
        ref_ast = FormulaParser(reference).parse()
    except ValueError:
        return 0.0

    distance = _tree_edit_distance(pred_ast, ref_ast)
    max_nodes = max(_count_nodes(pred_ast), _count_nodes(ref_ast))

    if max_nodes == 0:
        return 1.0

    # Normalizar: 0 distancia = 1.0 credito, max distancia = 0.0 credito
    return max(0.0, 1.0 - distance / max_nodes)


# =====================================================================
# COMPOSITIONAL METRICS: evaluacion por capas
# =====================================================================
# Separa la evaluacion en 4 niveles de composicion.
# Esto dice EXACTAMENTE donde se rompe la cadena de razonamiento.
#
# Nivel 1 — Atomos:            p, q, r, s
# Nivel 2 — Sub-formulas:     p ∧ q, ¬s, r ∨ ¬s
# Nivel 3 — Conector principal: →
# Nivel 4 — Formula completa


def extract_subformulas(node: ASTNode) -> list[str]:
    """Extrae todas las sub-formulas de un AST.

    Recorre el arbol y convierte cada sub-arbol a string.
    Excluye atomos individuales (esos ya se miden con atom_f1).

    Args:
        node: Raiz del AST.

    Returns:
        Lista de sub-formulas como strings.

    Ejemplo:
        ast = parse("(p ∧ q) → (r ∨ ¬s)")
        extract_subformulas(ast)
        → ["p ∧ q", "¬s", "r ∨ ¬s", "(p ∧ q) → (r ∨ ¬s)"]
    """
    subformulas: list[str] = []

    def _collect(n: ASTNode) -> None:
        if isinstance(n, AtomNode):
            return  # atomos se miden por separado

        if isinstance(n, NotNode):
            subformulas.append(_ast_to_string(n))
            _collect(n.operand)

        elif isinstance(n, BinaryNode):
            _collect(n.left)
            _collect(n.right)
            subformulas.append(_ast_to_string(n))

    _collect(node)
    return subformulas


def get_main_connector(node: ASTNode) -> str | None:
    """Extrae el conector principal (raiz) de una formula.

    Args:
        node: Raiz del AST.

    Returns:
        Simbolo del conector principal, o None si es un atomo.

    Ejemplo:
        get_main_connector(parse("(p ∧ q) → r"))  → "→"
        get_main_connector(parse("¬p"))            → "¬"
        get_main_connector(parse("p"))             → None
    """
    connector_map = {
        "AND": "∧",
        "OR": "∨",
        "IMPLIES": "→",
        "BICONDITIONAL": "↔",
    }

    if isinstance(node, NotNode):
        return "¬"
    if isinstance(node, BinaryNode):
        return connector_map.get(node.operator.name, "?")
    return None


@dataclass
class CompositionalResult:
    """Resultado de evaluacion compositional por capas.

    Cada campo es un score entre 0.0 y 1.0.
    """

    atom_score: float = 0.0  # Nivel 1: atomos correctos
    subformula_score: float = 0.0  # Nivel 2: sub-formulas correctas
    connector_score: float = 0.0  # Nivel 3: conector principal correcto
    full_score: float = 0.0  # Nivel 4: formula completa equivalente

    @property
    def average(self) -> float:
        """Promedio de los 4 niveles."""
        return (
            self.atom_score + self.subformula_score + self.connector_score + self.full_score
        ) / 4

    def __str__(self) -> str:
        return (
            f"Compositional: atoms={self.atom_score:.2f} "
            f"subfórmulas={self.subformula_score:.2f} "
            f"conector={self.connector_score:.2f} "
            f"completa={self.full_score:.2f} "
            f"(avg={self.average:.2f})"
        )


def compositional_score(prediction: str, reference: str) -> CompositionalResult:
    """Evalua la prediccion en 4 niveles de composicion.

    Nivel 1 — Atomos: ¿identifico las variables correctas?
    Nivel 2 — Sub-formulas: ¿armo las sub-expresiones correctas?
    Nivel 3 — Conector principal: ¿eligio el conector correcto en la raiz?
    Nivel 4 — Formula completa: ¿es logicamente equivalente?

    Args:
        prediction: Formula generada.
        reference: Formula ground truth.

    Returns:
        CompositionalResult con scores por nivel.

    Ejemplo:
        compositional_score("(p ∧ q) → r", "(p ∧ q) → (r ∨ s)")
        → atoms=0.75, subfórmulas=0.50, conector=1.00, completa=0.00
        # Acertó el conector → y el antecedente p∧q, pero falta r∨s
    """
    result = CompositionalResult()

    try:
        pred_ast = FormulaParser(prediction).parse()
        ref_ast = FormulaParser(reference).parse()
    except ValueError:
        return result

    # --- Nivel 1: Atomos (F1) ---
    pred_atoms = extract_atoms(pred_ast)
    ref_atoms = extract_atoms(ref_ast)

    if ref_atoms:
        correct = pred_atoms & ref_atoms
        precision = len(correct) / len(pred_atoms) if pred_atoms else 0.0
        recall = len(correct) / len(ref_atoms)
        if precision + recall > 0:
            result.atom_score = 2 * precision * recall / (precision + recall)
    else:
        result.atom_score = 1.0 if not pred_atoms else 0.0

    # --- Nivel 2: Sub-formulas ---
    pred_subs = set(extract_subformulas(pred_ast))
    ref_subs = set(extract_subformulas(ref_ast))

    if ref_subs:
        # Comparar sub-formulas semanticamente (no solo texto)
        matched = 0
        used_pred = set()
        for ref_sub in ref_subs:
            for pred_sub in pred_subs - used_pred:
                try:
                    if are_equivalent(pred_sub, ref_sub):
                        matched += 1
                        used_pred.add(pred_sub)
                        break
                except (ValueError, RecursionError):
                    continue
        result.subformula_score = matched / len(ref_subs)
    else:
        result.subformula_score = 1.0 if not pred_subs else 0.0

    # --- Nivel 3: Conector principal ---
    pred_main = get_main_connector(pred_ast)
    ref_main = get_main_connector(ref_ast)
    result.connector_score = 1.0 if pred_main == ref_main else 0.0

    # --- Nivel 4: Formula completa ---
    sem = semantic_match(prediction, reference)
    result.full_score = 1.0 if sem is True else 0.0

    return result


# =====================================================================
# EVALUACION POR EJEMPLO
# =====================================================================


@dataclass
class ExampleResult:
    """Resultado de evaluar un solo ejemplo.

    Contiene todas las metricas para una prediccion vs referencia.
    """

    prediction: str
    reference: str
    is_exact_match: bool = False
    is_semantic_match: bool | None = None
    is_normalized_match: bool = False
    is_syntax_valid: bool = False
    syntax_error: str = ""
    atom_precision: float = 0.0
    atom_recall: float = 0.0
    atom_f1_score: float = 0.0
    connector_acc: float = 0.0
    partial_credit_score: float = 0.0
    compositional: CompositionalResult = field(default_factory=CompositionalResult)


def evaluate_example(prediction: str, reference: str) -> ExampleResult:
    """Evalua un solo ejemplo con TODAS las metricas.

    Args:
        prediction: Formula generada por el modelo.
        reference: Formula ground truth.

    Returns:
        ExampleResult con todas las metricas.
    """
    result = ExampleResult(prediction=prediction, reference=reference)

    # 1. Exact Match
    result.is_exact_match = exact_match(prediction, reference)

    # 2. Syntax Valid
    valid, error = is_valid_formula(prediction)
    result.is_syntax_valid = valid
    result.syntax_error = error

    # 3. Normalized Match
    result.is_normalized_match = normalized_match(prediction, reference)

    # 4. Semantic Match (solo si ambas son validas)
    result.is_semantic_match = semantic_match(prediction, reference)

    # 5. Atom F1
    f1_scores = atom_f1(prediction, reference)
    result.atom_precision = f1_scores["precision"]
    result.atom_recall = f1_scores["recall"]
    result.atom_f1_score = f1_scores["f1"]

    # 6. Connector Accuracy
    result.connector_acc = connector_accuracy(prediction, reference)

    # 7. Partial Credit
    result.partial_credit_score = partial_credit(prediction, reference)

    # 8. Compositional Score
    result.compositional = compositional_score(prediction, reference)

    return result


# =====================================================================
# EVALUACION POR BATCH
# =====================================================================


@dataclass
class BatchResults:
    """Resultados agregados de evaluar un batch de ejemplos.

    Contiene las metricas promediadas y los resultados individuales.
    """

    # Metricas agregadas (promedios)
    exact_match: float = 0.0
    semantic_accuracy: float = 0.0
    normalized_match: float = 0.0
    syntax_valid_rate: float = 0.0
    atom_f1: float = 0.0
    connector_accuracy: float = 0.0
    partial_credit: float = 0.0
    compositional_avg: float = 0.0
    compositional_atoms: float = 0.0
    compositional_subformulas: float = 0.0
    compositional_connector: float = 0.0
    compositional_full: float = 0.0

    # Conteos
    total: int = 0
    n_exact: int = 0
    n_semantic: int = 0
    n_normalized: int = 0
    n_valid: int = 0

    # Resultados individuales (para analisis detallado)
    examples: list[ExampleResult] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        """Retorna un dict con las metricas principales."""
        return {
            "exact_match": self.exact_match,
            "semantic_accuracy": self.semantic_accuracy,
            "normalized_match": self.normalized_match,
            "syntax_valid_rate": self.syntax_valid_rate,
            "atom_f1": self.atom_f1,
            "connector_accuracy": self.connector_accuracy,
            "partial_credit": self.partial_credit,
            "compositional_avg": self.compositional_avg,
            "compositional_atoms": self.compositional_atoms,
            "compositional_subformulas": self.compositional_subformulas,
            "compositional_connector": self.compositional_connector,
            "compositional_full": self.compositional_full,
        }

    def __str__(self) -> str:
        lines = [
            f"{'=' * 50}",
            f"  METRICAS DE EVALUACION ({self.total} ejemplos)",
            f"{'=' * 50}",
            f"  Exact Match:        {self.exact_match:.1%}  ({self.n_exact}/{self.total})",
            f"  Semantic Accuracy:  {self.semantic_accuracy:.1%}  ({self.n_semantic}/{self.total})",
            f"  Normalized Match:   {self.normalized_match:.1%}  ({self.n_normalized}/{self.total})",
            f"  Syntax Valid Rate:  {self.syntax_valid_rate:.1%}  ({self.n_valid}/{self.total})",
            f"  Atom F1:            {self.atom_f1:.3f}",
            f"  Connector Accuracy: {self.connector_accuracy:.1%}",
            f"  Partial Credit:     {self.partial_credit:.3f}",
            f"  {'─' * 48}",
            f"  Compositional:",
            f"    Atoms:            {self.compositional_atoms:.3f}",
            f"    Sub-formulas:     {self.compositional_subformulas:.3f}",
            f"    Main Connector:   {self.compositional_connector:.3f}",
            f"    Full Equivalence: {self.compositional_full:.3f}",
            f"    Average:          {self.compositional_avg:.3f}",
            f"{'=' * 50}",
        ]
        return "\n".join(lines)


def evaluate_batch(
    predictions: list[str],
    references: list[str],
) -> BatchResults:
    """Evalua un batch de predicciones vs referencias.

    Args:
        predictions: Lista de formulas generadas por el modelo.
        references: Lista de formulas ground truth.

    Returns:
        BatchResults con metricas agregadas y resultados individuales.

    Ejemplo:
        results = evaluate_batch(
            predictions=["p ∧ q", "q → p", "p ∨ q"],
            references=["q ∧ p", "p → q", "p ∨ q"],
        )
        print(results)
    """
    assert len(predictions) == len(references), (
        f"predictions ({len(predictions)}) y references ({len(references)}) "
        f"deben tener el mismo largo"
    )

    results = BatchResults(total=len(predictions))

    total_atom_f1 = 0.0
    total_connector = 0.0
    total_partial = 0.0
    total_comp_atoms = 0.0
    total_comp_subs = 0.0
    total_comp_conn = 0.0
    total_comp_full = 0.0
    n_semantic_valid = 0  # cuantos se pudieron evaluar semanticamente

    for pred, ref in zip(predictions, references):
        example = evaluate_example(pred, ref)
        results.examples.append(example)

        # Conteos
        if example.is_exact_match:
            results.n_exact += 1
        if example.is_normalized_match:
            results.n_normalized += 1
        if example.is_syntax_valid:
            results.n_valid += 1
        if example.is_semantic_match is True:
            results.n_semantic += 1
            n_semantic_valid += 1
        elif example.is_semantic_match is False:
            n_semantic_valid += 1
        # Si is_semantic_match is None, no se pudo evaluar → se excluye

        # Acumuladores para promedios
        total_atom_f1 += example.atom_f1_score
        total_connector += example.connector_acc
        total_partial += example.partial_credit_score
        total_comp_atoms += example.compositional.atom_score
        total_comp_subs += example.compositional.subformula_score
        total_comp_conn += example.compositional.connector_score
        total_comp_full += example.compositional.full_score

    n = results.total
    if n > 0:
        results.exact_match = results.n_exact / n
        results.normalized_match = results.n_normalized / n
        results.syntax_valid_rate = results.n_valid / n
        results.atom_f1 = total_atom_f1 / n
        results.connector_accuracy = total_connector / n
        results.partial_credit = total_partial / n
        results.compositional_atoms = total_comp_atoms / n
        results.compositional_subformulas = total_comp_subs / n
        results.compositional_connector = total_comp_conn / n
        results.compositional_full = total_comp_full / n
        results.compositional_avg = (
            results.compositional_atoms
            + results.compositional_subformulas
            + results.compositional_connector
            + results.compositional_full
        ) / 4

    if n_semantic_valid > 0:
        results.semantic_accuracy = results.n_semantic / n_semantic_valid
    else:
        results.semantic_accuracy = 0.0

    return results


# =====================================================================
# ANALISIS DE ERRORES
# =====================================================================


def error_analysis(results: BatchResults) -> str:
    """Genera un reporte de analisis de errores.

    Identifica los patrones de error mas comunes:
    - ¿Falla en atomos o conectores?
    - ¿Genera formulas invalidas?
    - ¿Que tipo de error es mas frecuente?

    Args:
        results: BatchResults de evaluate_batch.

    Returns:
        String con el reporte formateado.
    """
    lines = [
        f"\n{'=' * 50}",
        f"  ANALISIS DE ERRORES",
        f"{'=' * 50}",
    ]

    # Clasificar errores
    syntax_errors = []
    semantic_errors = []
    near_misses = []  # partial credit > 0.7 pero no exacto

    for ex in results.examples:
        if not ex.is_syntax_valid:
            syntax_errors.append(ex)
        elif ex.is_semantic_match is False:
            semantic_errors.append(ex)
            if ex.partial_credit_score > 0.7:
                near_misses.append(ex)

    lines.append(f"\n  Errores de sintaxis: {len(syntax_errors)}")
    for ex in syntax_errors[:5]:
        lines.append(f"    Pred: {ex.prediction}")
        lines.append(f"    Error: {ex.syntax_error}")
        lines.append("")

    lines.append(f"  Errores semanticos: {len(semantic_errors)}")
    for ex in semantic_errors[:5]:
        lines.append(f"    Pred: {ex.prediction}")
        lines.append(f"    Ref:  {ex.reference}")
        lines.append(f"    Partial: {ex.partial_credit_score:.2f}  Atom F1: {ex.atom_f1_score:.2f}")
        lines.append("")

    if near_misses:
        lines.append(f"  Near misses (partial > 0.7): {len(near_misses)}")
        lines.append("  (El modelo casi acierta — probablemente necesita mas datos)")

    # Diagnostico
    lines.append(f"\n  {'=' * 50}")
    lines.append(f"  DIAGNOSTICO")
    lines.append(f"  {'=' * 50}")

    if results.syntax_valid_rate < 0.8:
        lines.append("  ⚠️  Syntax rate bajo: el modelo no aprendio la gramatica.")
        lines.append("     → Necesita mas epochs o mas datos simples.")
    elif results.atom_f1 < 0.8:
        lines.append("  ⚠️  Atom F1 bajo: no identifica las variables correctas.")
        lines.append("     → Necesita mas variedad de atomos en los datos.")
    elif results.connector_accuracy < 0.8:
        lines.append("  ⚠️  Connector accuracy bajo: identifica atomos pero confunde conectores.")
        lines.append("     → Necesita mas datos de los conectores mas dificiles.")
    elif results.semantic_accuracy < 0.8:
        lines.append("  ⚠️  Semantic accuracy baja con buenos atomos y conectores:")
        lines.append("     → Problema de estructura/parentesis. Necesita formulas mas complejas.")
    else:
        lines.append("  ✅  El modelo funciona bien en todas las dimensiones.")

    return "\n".join(lines)
