"""
Special Tokens — Los marcadores que le dan estructura a las secuencias.

¿Qué son?
    Son tokens "inventados" que NO existen en el lenguaje natural.
    Le dicen al modelo dónde empieza y termina cada sección de la secuencia.
    Sin ellos, el modelo no sabría distinguir el input de la fórmula.

¿Cómo se usan?
    Una secuencia de entrenamiento se ve así:

    FASE 1 (con Chain-of-Thought completo):
    <|bos|><|input|> Si llueve me mojo <|output|><|thought|> "si...entonces"
    indica implicación <|atoms|> p: llueve | q: me mojo <|connectors|>
    →: si...entonces <|formula|> p → q <|eos|>

    FASE 2 (destilada — sin thought):
    <|bos|><|input|> Si llueve me mojo <|output|><|formula|> p → q <|eos|>

¿Por qué frozen=True?
    Para que nadie pueda modificar los tokens por accidente.
    Si alguien escribe SPECIAL_TOKENS.PAD = "otro_valor", Python lanza error.
    Los tokens deben ser CONSTANTES en todo el proyecto.

¿Por qué el formato <|...|>?
    Es el estándar que usan GPT, LLaMA, Mistral, etc.
    Los delimitadores <| y |> hacen que sea imposible confundir
    un special token con texto normal del usuario.
"""

from __future__ import annotations

from dataclasses import dataclass


# =================================================================
# La clase SpecialTokens: define TODOS los tokens especiales
# =================================================================
#
# frozen=True → inmutable, nadie puede cambiar los valores
# Se usa UNA SOLA INSTANCIA global (SPECIAL_TOKENS) en todo el proyecto.
#


@dataclass(frozen=True)
class SpecialTokens:
    """Definición de todos los tokens especiales del modelo.

    Estos tokens se agregan al vocabulario del tokenizer BPE
    y el modelo aprende a usarlos durante el entrenamiento.
    """

    # ----- Tokens de control (estándar en todos los modelos) -----

    # PAD = "padding" — relleno para igualar el largo de secuencias en un batch.
    # Ejemplo: si una secuencia tiene 50 tokens y otra 80, la de 50 se rellena
    # con 30 PAD tokens para que ambas tengan 80.
    # El modelo aprende a IGNORAR estos tokens (se enmascaran en la loss).
    PAD: str = "<|pad|>"

    # BOS = "beginning of sequence" — marca el inicio de una secuencia.
    # Es lo primero que ve el modelo. Le dice "aquí empieza todo".
    BOS: str = "<|bos|>"

    # EOS = "end of sequence" — marca el final de la generación.
    # Cuando el modelo genera este token, PARA de generar.
    # Es el estándar de la industria (GPT, LLaMA, Mistral todos usan eos).
    EOS: str = "<|eos|>"

    # UNK = "unknown" — para caracteres que el tokenizer no conoce.
    # Si aparece un carácter raro que no está en el vocabulario BPE,
    # se reemplaza por UNK. Idealmente nunca debería aparecer.
    UNK: str = "<|unk|>"

    # ----- Tokens de estructura (específicos de NanoLogic) -----
    # Estos definen las SECCIONES de cada secuencia de entrenamiento.

    # INPUT = marca el inicio del texto en lenguaje natural (español).
    # Todo lo que viene después de este token es la frase del usuario.
    INPUT: str = "<|input|>"

    # OUTPUT = marca el inicio de lo que el modelo debe GENERAR.
    # En entrenamiento: el modelo ve todo hasta aquí y aprende a generar el resto.
    # En inferencia: ponemos el input y el modelo genera desde aquí.
    OUTPUT: str = "<|output|>"

    # THOUGHT = marca el inicio del razonamiento paso a paso (Chain-of-Thought).
    # Fase 1: el modelo aprende a PENSAR antes de dar la fórmula.
    # Fase 2: se elimina esta sección (destilación) y el modelo genera directo.
    THOUGHT: str = "<|thought|>"

    # ATOMS = marca la sección donde se listan los átomos proposicionales.
    # Ejemplo: "p: llueve | q: me mojo"
    # Le enseña al modelo a IDENTIFICAR las proposiciones antes de formular.
    ATOMS: str = "<|atoms|>"

    # CONNECTORS = marca la sección donde se listan los conectores lógicos.
    # Ejemplo: "→: si...entonces | ∧: y además"
    # Le enseña al modelo a CLASIFICAR las relaciones lógicas.
    CONNECTORS: str = "<|connectors|>"

    # FORMULA = marca el inicio de la fórmula de lógica proposicional.
    # Esta es la salida PRINCIPAL del modelo — lo que realmente importa.
    # Todo lo anterior (thought, atoms, connectors) es andamiaje para llegar aquí.
    FORMULA: str = "<|formula|>"

    def as_list(self) -> list[str]:
        """Devuelve todos los special tokens como lista.

        ¿Para qué? El tokenizer BPE necesita saber cuáles son los
        tokens especiales para agregarlos al vocabulario y no
        partirlos en sub-tokens. Sin esto, el BPE vería "<|bos|>"
        y lo partiría en ["<", "|", "b", "o", "s", "|", ">"] — mal.
        """
        return [
            self.PAD,
            self.BOS,
            self.EOS,
            self.UNK,
            self.INPUT,
            self.OUTPUT,
            self.THOUGHT,
            self.ATOMS,
            self.CONNECTORS,
            self.FORMULA,
        ]

    def as_dict(self) -> dict[str, str]:
        """Devuelve un diccionario nombre → token.

        Útil para debug y para pasar al tokenizer de HuggingFace
        que espera un dict con nombres específicos como
        "pad_token", "bos_token", "eos_token", etc.
        """
        return {
            "pad_token": self.PAD,
            "bos_token": self.BOS,
            "eos_token": self.EOS,
            "unk_token": self.UNK,
            # Los siguientes son tokens adicionales (HuggingFace los llama
            # "additional_special_tokens")
            "additional_special_tokens": [
                self.INPUT,
                self.OUTPUT,
                self.THOUGHT,
                self.ATOMS,
                self.CONNECTORS,
                self.FORMULA,
            ],
        }

    @property
    def count(self) -> int:
        """Número total de special tokens.

        Importante para calcular el tamaño del vocabulario:
        vocab_size = tokens_BPE + special_tokens.count
        """
        return len(self.as_list())


# =================================================================
# Instancia global — ESTA es la que se importa en todo el proyecto
# =================================================================
#
# En vez de crear un SpecialTokens() en cada archivo, importamos
# esta instancia única:
#
#     from src.tokenizer.special_tokens import SPECIAL_TOKENS
#
# Así garantizamos que TODOS los archivos usan los MISMOS tokens.
#

SPECIAL_TOKENS = SpecialTokens()
