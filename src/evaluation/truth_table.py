"""
truth_table.py — Parser de formulas logicas + generador de tablas de verdad.

Este modulo es el corazon de la evaluacion semantica de NanoLogic.
Permite determinar si dos formulas logicas son EQUIVALENTES, incluso
si estan escritas de forma diferente.

Ejemplo:
    "p ∧ q → r"  y  "q ∧ p → r"  son equivalentes (conmutatividad).
    Exact match diria que son diferentes (texto distinto).
    Tabla de verdad dice que son iguales (mismos valores).

Flujo:
    1. Tokenizar: "p ∧ q → r"  →  [ATOM(p), AND, ATOM(q), IMPLIES, ATOM(r)]
    2. Parsear: tokens → AST (arbol de sintaxis abstracta)
    3. Evaluar: AST + asignacion de valores → True/False
    4. Generar tabla: todas las combinaciones de valores → columna de resultados
    5. Comparar: dos tablas iguales → formulas equivalentes

Gramatica soportada (BNF):
    formula    ::= biconditional
    biconditional ::= implication ('↔' implication)*
    implication   ::= disjunction ('→' disjunction)*
    disjunction   ::= conjunction ('∨' conjunction)*
    conjunction   ::= unary ('∧' unary)*
    unary         ::= '¬' unary | atom | '(' formula ')'
    atom          ::= [a-z][a-z0-9]*

Precedencia de operadores (de menor a mayor):
    1. ↔  (bicondicional)
    2. →  (implicacion)
    3. ∨  (disyuncion)
    4. ∧  (conjuncion)
    5. ¬  (negacion)

Esto sigue el estandar de logica proposicional clasica.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from itertools import product


# =====================================================================
# TOKENIZER DE FORMULAS
# =====================================================================
# Convierte un string de formula en una lista de tokens.
# Soporta conectores en Unicode (∧, ∨, →, ↔, ¬) y ASCII (^, v, ->, <->).


class TokenType(Enum):
    """Tipos de token en una formula logica."""

    ATOM = auto()       # Variable proposicional: p, q, r, ...
    NOT = auto()        # Negacion: ¬, ~, !
    AND = auto()        # Conjuncion: ∧, ^, &
    OR = auto()         # Disyuncion: ∨, v (como operador), |
    IMPLIES = auto()    # Implicacion: →, ->, =>
    BICONDITIONAL = auto()  # Bicondicional: ↔, <->, <=>
    LPAREN = auto()     # Parentesis izquierdo: (
    RPAREN = auto()     # Parentesis derecho: )
    EOF = auto()        # Fin de la formula


@dataclass
class Token:
    """Un token individual de una formula."""

    type: TokenType
    value: str

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r})"


# Mapeo de strings a tipos de token.
# Soportamos multiples notaciones para cada conector.
CONNECTORS: dict[str, TokenType] = {
    # Negacion
    "¬": TokenType.NOT,
    "~": TokenType.NOT,
    "!": TokenType.NOT,
    "not": TokenType.NOT,
    # Conjuncion
    "∧": TokenType.AND,
    "^": TokenType.AND,
    "&": TokenType.AND,
    "and": TokenType.AND,
    # Disyuncion
    "∨": TokenType.OR,
    "|": TokenType.OR,
    "or": TokenType.OR,
    # Implicacion
    "→": TokenType.IMPLIES,
    "->": TokenType.IMPLIES,
    "=>": TokenType.IMPLIES,
    "implies": TokenType.IMPLIES,
    # Bicondicional
    "↔": TokenType.BICONDITIONAL,
    "<->": TokenType.BICONDITIONAL,
    "<=>": TokenType.BICONDITIONAL,
    "iff": TokenType.BICONDITIONAL,
}


def tokenize(formula: str) -> list[Token]:
    """Convierte una formula en una lista de tokens.

    Soporta notacion Unicode y ASCII:
        "p ∧ q → r"     (Unicode)
        "p ^ q -> r"     (ASCII)
        "p and q implies r"  (texto)
        "¬(p ∨ q)"       (con parentesis)

    Args:
        formula: String de una formula logica.

    Returns:
        Lista de tokens.

    Raises:
        ValueError: Si encuentra un caracter no reconocido.
    """
    tokens: list[Token] = []
    i = 0
    formula = formula.strip()

    while i < len(formula):
        char = formula[i]

        # Ignorar espacios
        if char.isspace():
            i += 1
            continue

        # Parentesis
        if char == "(":
            tokens.append(Token(TokenType.LPAREN, "("))
            i += 1
            continue
        if char == ")":
            tokens.append(Token(TokenType.RPAREN, ")"))
            i += 1
            continue

        # Conectores multi-caracter: <->, <=>, ->. =>
        if i + 2 < len(formula):
            three = formula[i : i + 3]
            if three in CONNECTORS:
                tokens.append(Token(CONNECTORS[three], three))
                i += 3
                continue

        if i + 1 < len(formula):
            two = formula[i : i + 2]
            if two in CONNECTORS:
                tokens.append(Token(CONNECTORS[two], two))
                i += 2
                continue

        # Conectores de un caracter (Unicode)
        if char in CONNECTORS:
            tokens.append(Token(CONNECTORS[char], char))
            i += 1
            continue

        # Atomo: letra minuscula seguida opcionalmente de digitos
        # p, q, r, p1, p2, var1, etc.
        if char.isalpha() and char.islower():
            start = i
            while i < len(formula) and (formula[i].isalnum() or formula[i] == "_"):
                i += 1
            word = formula[start:i]

            # Verificar si es un keyword (not, and, or, implies, iff)
            if word.lower() in CONNECTORS:
                tokens.append(Token(CONNECTORS[word.lower()], word))
            else:
                tokens.append(Token(TokenType.ATOM, word))
            continue

        # Atomo en mayuscula (P, Q, R) — tambien soportado
        if char.isalpha() and char.isupper():
            start = i
            while i < len(formula) and (formula[i].isalnum() or formula[i] == "_"):
                i += 1
            word = formula[start:i]
            tokens.append(Token(TokenType.ATOM, word.lower()))
            continue

        raise ValueError(f"Caracter no reconocido en posicion {i}: '{char}' en formula '{formula}'")

    tokens.append(Token(TokenType.EOF, ""))
    return tokens


# =====================================================================
# AST (Abstract Syntax Tree)
# =====================================================================
# Representacion en arbol de una formula logica.
# Cada nodo es una operacion o un atomo.


class ASTNode:
    """Nodo base del arbol de sintaxis abstracta."""

    pass


@dataclass
class AtomNode(ASTNode):
    """Atomo proposicional (hoja del arbol).

    Ejemplo: 'p', 'q', 'r'
    """

    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass
class NotNode(ASTNode):
    """Negacion: ¬φ

    Ejemplo: ¬p → NotNode(operand=AtomNode('p'))
    """

    operand: ASTNode

    def __repr__(self) -> str:
        return f"¬({self.operand})"


@dataclass
class BinaryNode(ASTNode):
    """Operacion binaria: φ ○ ψ

    Ejemplo: p ∧ q → BinaryNode(AND, AtomNode('p'), AtomNode('q'))
    """

    operator: TokenType
    left: ASTNode
    right: ASTNode

    def __repr__(self) -> str:
        op_symbols = {
            TokenType.AND: "∧",
            TokenType.OR: "∨",
            TokenType.IMPLIES: "→",
            TokenType.BICONDITIONAL: "↔",
        }
        op = op_symbols.get(self.operator, "?")
        return f"({self.left} {op} {self.right})"


# =====================================================================
# PARSER: tokens → AST
# =====================================================================
# Recursive descent parser que respeta precedencia de operadores.
#
# Precedencia (de menor a mayor):
#   biconditional < implication < disjunction < conjunction < negation
#
# Cada nivel de precedencia tiene su propia funcion de parsing.
# Las funciones se llaman recursivamente de menor a mayor precedencia.


class FormulaParser:
    """Parser de recursive descent para formulas logicas.

    Ejemplo:
        parser = FormulaParser("p ∧ q → r")
        ast = parser.parse()
        # ast = BinaryNode(IMPLIES, BinaryNode(AND, Atom(p), Atom(q)), Atom(r))
    """

    def __init__(self, formula: str) -> None:
        self.tokens = tokenize(formula)
        self.pos = 0

    def parse(self) -> ASTNode:
        """Parsea la formula completa y retorna el AST.

        Raises:
            ValueError: Si la formula tiene errores de sintaxis.
        """
        ast = self._biconditional()

        if self._current().type != TokenType.EOF:
            raise ValueError(
                f"Tokens inesperados despues de la formula: "
                f"'{self._current().value}' en posicion {self.pos}"
            )

        return ast

    def _current(self) -> Token:
        """Token actual sin avanzar."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token(TokenType.EOF, "")

    def _advance(self) -> Token:
        """Consume y retorna el token actual."""
        token = self._current()
        self.pos += 1
        return token

    def _expect(self, token_type: TokenType) -> Token:
        """Consume un token del tipo esperado o lanza error."""
        token = self._current()
        if token.type != token_type:
            raise ValueError(
                f"Se esperaba {token_type.name}, se encontro "
                f"{token.type.name} ('{token.value}') en posicion {self.pos}"
            )
        return self._advance()

    # --- Niveles de precedencia (de menor a mayor) ---

    def _biconditional(self) -> ASTNode:
        """biconditional ::= implication ('↔' implication)*"""
        left = self._implication()
        while self._current().type == TokenType.BICONDITIONAL:
            self._advance()
            right = self._implication()
            left = BinaryNode(TokenType.BICONDITIONAL, left, right)
        return left

    def _implication(self) -> ASTNode:
        """implication ::= disjunction ('→' disjunction)*

        Nota: la implicacion es RIGHT-ASSOCIATIVE.
        p → q → r se parsea como p → (q → r), NO como (p → q) → r.
        """
        left = self._disjunction()
        if self._current().type == TokenType.IMPLIES:
            self._advance()
            # Recursion a la derecha (right-associative)
            right = self._implication()
            left = BinaryNode(TokenType.IMPLIES, left, right)
        return left

    def _disjunction(self) -> ASTNode:
        """disjunction ::= conjunction ('∨' conjunction)*"""
        left = self._conjunction()
        while self._current().type == TokenType.OR:
            self._advance()
            right = self._conjunction()
            left = BinaryNode(TokenType.OR, left, right)
        return left

    def _conjunction(self) -> ASTNode:
        """conjunction ::= unary ('∧' unary)*"""
        left = self._unary()
        while self._current().type == TokenType.AND:
            self._advance()
            right = self._unary()
            left = BinaryNode(TokenType.AND, left, right)
        return left

    def _unary(self) -> ASTNode:
        """unary ::= '¬' unary | atom | '(' formula ')'"""
        if self._current().type == TokenType.NOT:
            self._advance()
            operand = self._unary()
            return NotNode(operand)

        if self._current().type == TokenType.LPAREN:
            self._advance()
            node = self._biconditional()
            self._expect(TokenType.RPAREN)
            return node

        if self._current().type == TokenType.ATOM:
            token = self._advance()
            return AtomNode(token.value)

        raise ValueError(
            f"Se esperaba atomo, '¬', o '(' pero se encontro "
            f"{self._current().type.name} ('{self._current().value}') "
            f"en posicion {self.pos}"
        )


# =====================================================================
# EVALUADOR: AST + asignacion → True/False
# =====================================================================


def evaluate(node: ASTNode, assignment: dict[str, bool]) -> bool:
    """Evalua un AST con una asignacion de valores.

    Args:
        node: Raiz del AST.
        assignment: Dict de variable → valor. Ej: {'p': True, 'q': False}

    Returns:
        Valor de verdad de la formula con esa asignacion.

    Raises:
        ValueError: Si hay un atomo sin valor asignado.

    Ejemplo:
        ast = parse("p ∧ q")
        evaluate(ast, {'p': True, 'q': True})   → True
        evaluate(ast, {'p': True, 'q': False})  → False
    """
    if isinstance(node, AtomNode):
        if node.name not in assignment:
            raise ValueError(f"Atomo '{node.name}' no tiene valor asignado")
        return assignment[node.name]

    if isinstance(node, NotNode):
        return not evaluate(node.operand, assignment)

    if isinstance(node, BinaryNode):
        left = evaluate(node.left, assignment)
        right = evaluate(node.right, assignment)

        if node.operator == TokenType.AND:
            return left and right
        elif node.operator == TokenType.OR:
            return left or right
        elif node.operator == TokenType.IMPLIES:
            # p → q  es equivalente a  ¬p ∨ q
            # False → anything = True
            return (not left) or right
        elif node.operator == TokenType.BICONDITIONAL:
            # p ↔ q  es equivalente a  (p → q) ∧ (q → p)
            return left == right

    raise ValueError(f"Nodo desconocido: {type(node)}")


# =====================================================================
# EXTRACCION DE ATOMOS
# =====================================================================


def extract_atoms(node: ASTNode) -> set[str]:
    """Extrae todos los atomos (variables) de un AST.

    Args:
        node: Raiz del AST.

    Returns:
        Set de nombres de atomos. Ej: {'p', 'q', 'r'}
    """
    if isinstance(node, AtomNode):
        return {node.name}
    if isinstance(node, NotNode):
        return extract_atoms(node.operand)
    if isinstance(node, BinaryNode):
        return extract_atoms(node.left) | extract_atoms(node.right)
    return set()


# =====================================================================
# TABLA DE VERDAD
# =====================================================================


def truth_table(
    formula: str | ASTNode,
) -> tuple[list[str], list[tuple[dict[str, bool], bool]]]:
    """Genera la tabla de verdad completa de una formula.

    Args:
        formula: Formula como string o AST ya parseado.

    Returns:
        Tupla de:
            - Lista de nombres de atomos (ordenados alfabeticamente)
            - Lista de (asignacion, resultado) para cada fila

    Ejemplo:
        atoms, rows = truth_table("p → q")
        # atoms = ['p', 'q']
        # rows = [
        #     ({'p': False, 'q': False}, True),   # F → F = T
        #     ({'p': False, 'q': True},  True),   # F → T = T
        #     ({'p': True,  'q': False}, False),  # T → F = F
        #     ({'p': True,  'q': True},  True),   # T → T = T
        # ]
    """
    if isinstance(formula, str):
        ast = FormulaParser(formula).parse()
    else:
        ast = formula

    atoms = sorted(extract_atoms(ast))
    rows: list[tuple[dict[str, bool], bool]] = []

    # Generar todas las combinaciones de True/False para cada atomo
    for values in product([False, True], repeat=len(atoms)):
        assignment = dict(zip(atoms, values))
        result = evaluate(ast, assignment)
        rows.append((assignment, result))

    return atoms, rows


# =====================================================================
# EQUIVALENCIA SEMANTICA
# =====================================================================


def are_equivalent(formula_a: str, formula_b: str) -> bool:
    """Determina si dos formulas son logicamente equivalentes.

    Dos formulas son equivalentes si tienen la misma tabla de verdad
    para TODAS las combinaciones posibles de valores de sus atomos.

    Esto captura:
        - Conmutatividad: p ∧ q ≡ q ∧ p
        - De Morgan: ¬(p ∧ q) ≡ ¬p ∨ ¬q
        - Doble negacion: ¬¬p ≡ p
        - Cualquier equivalencia logica valida

    Args:
        formula_a: Primera formula.
        formula_b: Segunda formula.

    Returns:
        True si son logicamente equivalentes, False si no.

    Raises:
        ValueError: Si alguna formula tiene errores de sintaxis.

    Ejemplo:
        are_equivalent("p ∧ q", "q ∧ p")          → True
        are_equivalent("¬(p ∧ q)", "¬p ∨ ¬q")     → True
        are_equivalent("p → q", "¬p ∨ q")          → True
        are_equivalent("p ∧ q", "p ∨ q")           → False
    """
    ast_a = FormulaParser(formula_a).parse()
    ast_b = FormulaParser(formula_b).parse()

    # Obtener la UNION de atomos de ambas formulas
    # Si formula_a usa {p, q} y formula_b usa {p, q, r},
    # evaluamos con {p, q, r} para ambas.
    atoms_a = extract_atoms(ast_a)
    atoms_b = extract_atoms(ast_b)
    all_atoms = sorted(atoms_a | atoms_b)

    # Comparar resultado para CADA combinacion de valores
    for values in product([False, True], repeat=len(all_atoms)):
        assignment = dict(zip(all_atoms, values))
        result_a = evaluate(ast_a, assignment)
        result_b = evaluate(ast_b, assignment)
        if result_a != result_b:
            return False

    return True


# =====================================================================
# VALIDACION SINTACTICA
# =====================================================================


def is_valid_formula(formula: str) -> tuple[bool, str]:
    """Verifica si una formula es sintacticamente valida.

    Args:
        formula: String de formula.

    Returns:
        Tupla de (es_valida, mensaje_de_error).
        Si es valida, mensaje es "OK".

    Ejemplo:
        is_valid_formula("p → q")         → (True, "OK")
        is_valid_formula("p → → q")       → (False, "Se esperaba atomo...")
        is_valid_formula("p ∧ (q → r")    → (False, "Se esperaba )...")
        is_valid_formula("")              → (False, "Formula vacia")
    """
    formula = formula.strip()
    if not formula:
        return False, "Formula vacia"

    try:
        FormulaParser(formula).parse()
        return True, "OK"
    except ValueError as e:
        return False, str(e)


# =====================================================================
# NORMALIZACION DE FORMULAS
# =====================================================================
# Normalizar a forma canonica para comparacion rapida.
# Esto captura equivalencias "triviales" sin tabla de verdad.


def normalize_formula(formula: str) -> str:
    """Normaliza una formula a forma canonica.

    Pasos:
        1. Parsear a AST
        2. Eliminar doble negacion (¬¬p → p)
        3. Ordenar operandos conmutativos (∧, ∨, ↔) alfabeticamente
        4. Serializar de vuelta a string

    Args:
        formula: Formula a normalizar.

    Returns:
        Formula normalizada como string.

    Ejemplo:
        normalize_formula("q ∧ p")           → "p ∧ q"
        normalize_formula("¬¬p")             → "p"
        normalize_formula("(q ∨ p) ∧ r")     → "(p ∨ q) ∧ r"
    """
    try:
        ast = FormulaParser(formula).parse()
        normalized = _normalize_node(ast)
        return _ast_to_string(normalized)
    except ValueError:
        # Si no se puede parsear, retornar tal cual
        return formula.strip()


def _normalize_node(node: ASTNode) -> ASTNode:
    """Normaliza un nodo del AST recursivamente."""
    if isinstance(node, AtomNode):
        return node

    if isinstance(node, NotNode):
        inner = _normalize_node(node.operand)
        # Doble negacion: ¬¬p → p
        if isinstance(inner, NotNode):
            return inner.operand
        return NotNode(inner)

    if isinstance(node, BinaryNode):
        left = _normalize_node(node.left)
        right = _normalize_node(node.right)

        # Operadores conmutativos: ordenar operandos alfabeticamente
        if node.operator in (TokenType.AND, TokenType.OR, TokenType.BICONDITIONAL):
            left_str = _ast_to_string(left)
            right_str = _ast_to_string(right)
            if left_str > right_str:
                left, right = right, left

        return BinaryNode(node.operator, left, right)

    return node


def _ast_to_string(node: ASTNode) -> str:
    """Convierte un AST a string con notacion estandar (Unicode)."""
    if isinstance(node, AtomNode):
        return node.name

    if isinstance(node, NotNode):
        inner = _ast_to_string(node.operand)
        if isinstance(node.operand, BinaryNode):
            return f"¬({inner})"
        return f"¬{inner}"

    if isinstance(node, BinaryNode):
        op_symbols = {
            TokenType.AND: "∧",
            TokenType.OR: "∨",
            TokenType.IMPLIES: "→",
            TokenType.BICONDITIONAL: "↔",
        }
        op = op_symbols[node.operator]
        left = _ast_to_string(node.left)
        right = _ast_to_string(node.right)

        # Agregar parentesis si el hijo tiene menor precedencia
        if isinstance(node.left, BinaryNode) and _precedence(node.left.operator) < _precedence(node.operator):
            left = f"({left})"
        if isinstance(node.right, BinaryNode) and _precedence(node.right.operator) < _precedence(node.operator):
            right = f"({right})"

        return f"{left} {op} {right}"

    return str(node)


def _precedence(op: TokenType) -> int:
    """Retorna la precedencia de un operador (mayor = mas fuerte)."""
    return {
        TokenType.BICONDITIONAL: 1,
        TokenType.IMPLIES: 2,
        TokenType.OR: 3,
        TokenType.AND: 4,
    }.get(op, 0)


# =====================================================================
# PRETTY PRINT DE TABLA DE VERDAD
# =====================================================================


def format_truth_table(formula: str) -> str:
    """Genera una tabla de verdad formateada para imprimir.

    Ejemplo:
        print(format_truth_table("p → q"))

        p | q | p → q
        --|---|------
        F | F |   T
        F | T |   T
        T | F |   F
        T | T |   T
    """
    atoms, rows = truth_table(formula)
    header = " | ".join(atoms) + " | " + formula
    separator = "-|-".join("-" * len(a) for a in atoms) + "-|-" + "-" * len(formula)

    lines = [header, separator]
    for assignment, result in rows:
        values = " | ".join("T" if assignment[a] else "F" for a in atoms)
        result_str = "T" if result else "F"
        padding = " " * (len(formula) // 2)
        lines.append(f"{values} | {padding}{result_str}")

    return "\n".join(lines)
