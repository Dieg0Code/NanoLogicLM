"""
NanoLogicTokenizer — El wrapper del tokenizer BPE para NanoLogic.

¿Qué hace este archivo?
    Envuelve el tokenizer BPE de HuggingFace (la librería `tokenizers`)
    en una clase propia que sabe cómo:
    1. Entrenar un tokenizer desde cero con nuestros datos
    2. Codificar texto → números (encode)
    3. Decodificar números → texto (decode)
    4. Formatear ejemplos completos con special tokens

¿Por qué un wrapper y no usar HuggingFace directo?
    Porque queremos agregar lógica específica de NanoLogic:
    - Formateo automático con special tokens (<|input|>, <|formula|>, etc.)
    - Padding y truncamiento configurado para nuestro modelo
    - Métodos helpers como encode_example() que entienden nuestro dataset
    - Un solo punto de configuración para vocab_size, max_length, etc.

Flujo de uso:
    # ENTRENAR (una vez):
    tokenizer = NanoLogicTokenizer()
    tokenizer.train(files=["train.jsonl", "val.jsonl"], vocab_size=8000)
    tokenizer.save("models/tokenizer/")

    # USAR (siempre):
    tokenizer = NanoLogicTokenizer.load("models/tokenizer/")
    ids = tokenizer.encode("Si llueve me mojo")
    text = tokenizer.decode(ids)
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from tokenizers.normalizers import NFC

from src.tokenizer.special_tokens import SPECIAL_TOKENS


class NanoLogicTokenizer:
    """Tokenizer BPE adaptado para NanoLogic.

    Attributes:
        _tokenizer: El tokenizer interno de HuggingFace (tipo `Tokenizer`).
                     Es privado porque el usuario no debería tocarlo directamente.
        _trained: Bandera que indica si el tokenizer ya fue entrenado o cargado.
    """

    def __init__(self) -> None:
        """Inicializa un tokenizer BPE vacío (sin vocabulario).

        IMPORTANTE: Después de crear el objeto, debes:
        - Llamar a train() para entrenarlo con datos, o
        - Llamar a load() para cargar uno ya entrenado.
        Si intentas encode/decode sin entrenar, va a fallar.
        """

        # ===================================================================
        # 1. Crear el modelo BPE vacío
        # ===================================================================
        # BPE = Byte Pair Encoding — el algoritmo que decide cómo partir el texto.
        # unk_token = el token para caracteres desconocidos.
        # En teoría no debería usarse nunca (BPE puede caer hasta bytes),
        # pero lo definimos por seguridad.
        self._tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS.UNK))

        # ===================================================================
        # 2. Normalizador: NFC (Unicode Normalization Form C)
        # ===================================================================
        # ¿Qué hace? Normaliza caracteres Unicode a una forma canónica.
        # Ejemplo: "é" puede representarse como 1 carácter (é) o 2 (e + ´).
        # NFC unifica ambas representaciones a 1 carácter.
        # Importante porque nuestros datos tienen caracteres Unicode: ∧, ∨, →, ↔, ¬
        self._tokenizer.normalizer = NFC()

        # ===================================================================
        # 3. Pre-tokenizador: ByteLevel
        # ===================================================================
        # ¿Qué hace? Antes de aplicar BPE, divide el texto en "palabras".
        # ByteLevel convierte cada byte a un carácter visible y respeta espacios.
        #
        # Ejemplo:
        #   "Si llueve" → ["Si", " llueve"]  (nota el espacio antes de "llueve")
        #
        # ¿Por qué ByteLevel y no Whitespace?
        # - ByteLevel es lo que usa GPT-2, GPT-3, etc.
        # - Puede representar CUALQUIER texto (incluso emojis, símbolos raros)
        # - Nunca genera <UNK> porque cae hasta el nivel de bytes individuales
        #
        # add_prefix_space=False: no agregar espacio al inicio del texto.
        # Queremos que "<|bos|>" no tenga espacio antes.
        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # ===================================================================
        # 4. Decoder: ByteLevel
        # ===================================================================
        # El decoder revierte lo que hizo el pre-tokenizador.
        # Cuando decodificamos IDs → texto, el decoder convierte los bytes
        # de vuelta a caracteres legibles.
        self._tokenizer.decoder = decoders.ByteLevel()

        # ===================================================================
        # 5. Post-procesador: ByteLevel
        # ===================================================================
        # Limpia el texto decodificado (remueve artefactos del byte-level).
        self._tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        # Inicialmente NO está entrenado
        self._trained = False

    # ===================================================================
    # ENTRENAR
    # ===================================================================

    def train(
        self,
        files: list[str],
        vocab_size: int = 8000,
        min_frequency: int = 2,
    ) -> None:
        """Entrena el tokenizer BPE con los archivos dados.

        Args:
            files: Lista de paths a archivos de texto (uno por línea).
                   Pueden ser .jsonl — solo extraeremos el campo "sequence".
            vocab_size: Tamaño del vocabulario final (BPE tokens + special tokens).
                        Para NanoLogic, ~8000 es suficiente.
                        Más grande = más parámetros en la embedding sin beneficio.
            min_frequency: Frecuencia mínima para que un par se fusione.
                          2 significa que un par debe aparecer al menos 2 veces
                          para ser considerado. Evita fusiones de cosas raras.

        ¿Qué pasa internamente?
            1. Lee todos los textos de los archivos
            2. Los normaliza (NFC)
            3. Los pre-tokeniza (ByteLevel)
            4. Aplica el algoritmo BPE: fusiona pares frecuentes iterativamente
            5. Se detiene cuando alcanza vocab_size
            6. Agrega los special tokens al vocabulario
        """
        # ----- Preparar los archivos de texto para el entrenamiento -----
        # Si los archivos son JSONL, necesitamos extraer solo el texto.
        # Creamos archivos temporales con solo el texto plano.
        text_files = []
        for file_path in files:
            path = Path(file_path)
            if path.suffix == ".jsonl":
                # Extraer el campo 'sequence' de cada línea JSON
                # y guardarlo como texto plano en un archivo temporal
                temp_path = path.with_suffix(".txt.tmp")
                with (
                    open(path, "r", encoding="utf-8") as f_in,
                    open(temp_path, "w", encoding="utf-8") as f_out,
                ):
                    for line in f_in:
                        if line.strip():
                            data = json.loads(line)
                            # Usamos 'sequence' porque ya tiene los special tokens
                            f_out.write(data["sequence"] + "\n")
                text_files.append(str(temp_path))
            else:
                text_files.append(str(path))

        # ----- Configurar el trainer BPE -----
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            # Los special tokens se agregan AL INICIO del vocabulario.
            # Esto garantiza que tengan IDs bajos y predecibles:
            # PAD=0, BOS=1, EOS=2, UNK=3, INPUT=4, OUTPUT=5, etc.
            special_tokens=SPECIAL_TOKENS.as_list(),
            # show_progress: barra de progreso durante el entrenamiento
            show_progress=True,
        )

        # ----- ¡Entrenar! -----
        self._tokenizer.train(files=text_files, trainer=trainer)
        self._trained = True

        # ----- Limpiar archivos temporales -----
        for tf in text_files:
            if tf.endswith(".txt.tmp"):
                Path(tf).unlink(missing_ok=True)

    # ===================================================================
    # GUARDAR Y CARGAR
    # ===================================================================

    def save(self, directory: str) -> None:
        """Guarda el tokenizer entrenado en disco.

        Crea un archivo tokenizer.json con todo el vocabulario y las reglas
        de fusión (merges). Este archivo es todo lo que necesitas para
        reconstruir el tokenizer después.

        Args:
            directory: Directorio donde guardar. Se crea si no existe.
        """
        self._check_trained("save")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path / "tokenizer.json"))

    @classmethod
    def load(cls, path_or_dir: str) -> NanoLogicTokenizer:
        """Carga un tokenizer previamente entrenado desde disco.

        Args:
            path_or_dir: Ruta al archivo tokenizer.json O al directorio que lo contiene.

        Returns:
            Una instancia de NanoLogicTokenizer lista para usar.
        """
        instance = cls.__new__(cls)  # Crear instancia sin llamar a __init__

        path = Path(path_or_dir)

        # Si es un directorio, buscar tokenizer.json dentro
        if path.is_dir():
            path = path / "tokenizer.json"

        if not path.exists():
            msg = f"No se encontró el tokenizer en: {path}."
            if path.name == "tokenizer.json":
                msg += " ¿Ya entrenaste el tokenizer?"
            raise FileNotFoundError(msg)

        instance._tokenizer = Tokenizer.from_file(str(path))
        instance._trained = True
        return instance

    # ===================================================================
    # ENCODE Y DECODE — Las operaciones fundamentales
    # ===================================================================

    def encode(self, text: str) -> list[int]:
        """Convierte texto en una lista de IDs numéricos.

        Esta es la operación más básica del tokenizer.

        Args:
            text: Cualquier string de texto.

        Returns:
            Lista de IDs enteros. Cada ID corresponde a un token
            en el vocabulario.

        Ejemplo:
            >>> tokenizer.encode("Si llueve me mojo")
            [45, 892, 12, 567]
        """
        self._check_trained("encode")
        return self._tokenizer.encode(text).ids

    def encode_to_tokens(self, text: str) -> list[str]:
        """Convierte texto en una lista de tokens (strings).

        Útil para debug — para ver exactamente cómo se partió el texto.

        Args:
            text: Cualquier string de texto.

        Returns:
            Lista de strings, cada uno es un token.

        Ejemplo:
            >>> tokenizer.encode_to_tokens("Si llueve me mojo")
            ["Si", " llueve", " me", " mojo"]
        """
        self._check_trained("encode_to_tokens")
        return self._tokenizer.encode(text).tokens

    def decode(self, ids: list[int], skip_special: bool = False) -> str:
        """Convierte IDs numéricos de vuelta a texto legible.

        Args:
            ids: Lista de IDs enteros.
            skip_special: Si True, omite los special tokens del output.
                         Útil para ver solo el texto "puro" sin marcadores.

        Returns:
            String de texto reconstruido.

        Ejemplo:
            >>> tokenizer.decode([45, 892, 12, 567])
            "Si llueve me mojo"
        """
        self._check_trained("decode")
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special)

    # ===================================================================
    # ENCODE_EXAMPLE — Convierte un ejemplo del dataset a secuencia numérica
    # ===================================================================

    def encode_example(self, example: dict) -> list[int]:
        """Codifica un ejemplo completo del dataset procesado.

        Toma un dict con 'sequence' (que ya tiene los special tokens
        formateados por preprocess.py) y lo convierte en IDs numéricos.

        Args:
            example: Un dict del dataset con al menos el campo 'sequence'.
                    Ejemplo: {"sequence": "<|bos|><|input|> Si llueve...", ...}

        Returns:
            Lista de IDs numéricos lista para alimentar al modelo.

        ¿Por qué no formateamos con special tokens aquí?
            Porque preprocess.py ya lo hizo. El campo 'sequence' ya contiene
            todo formateado: <|bos|><|input|> texto <|output|>... <|eos|>
            Solo necesitamos tokenizarlo.
        """
        self._check_trained("encode_example")
        return self.encode(example["sequence"])

    # ===================================================================
    # PROPIEDADES — Información útil del tokenizer
    # ===================================================================

    @property
    def vocab_size(self) -> int:
        """Tamaño total del vocabulario (BPE tokens + special tokens).

        Este número es CRUCIAL — define el tamaño de la embedding layer
        del modelo: nn.Embedding(vocab_size, d_model).
        """
        self._check_trained("vocab_size")
        return self._tokenizer.get_vocab_size()

    @property
    def pad_id(self) -> int:
        """ID numérico del token PAD.

        Se usa en el DataLoader para rellenar secuencias más cortas
        y en la loss function para ignorar estos tokens.
        """
        self._check_trained("pad_id")
        return self._tokenizer.token_to_id(SPECIAL_TOKENS.PAD)

    @property
    def bos_id(self) -> int:
        """ID numérico del token BOS (Beginning of Sequence)."""
        self._check_trained("bos_id")
        return self._tokenizer.token_to_id(SPECIAL_TOKENS.BOS)

    @property
    def eos_id(self) -> int:
        """ID numérico del token EOS (End of Sequence).

        Se usa durante la inferencia: cuando el modelo genera este token,
        PARA de generar. Es la señal de "ya terminé".
        """
        self._check_trained("eos_id")
        return self._tokenizer.token_to_id(SPECIAL_TOKENS.EOS)

    def token_to_id(self, token: str) -> int | None:
        """Convierte un token (string) a su ID numérico.

        Útil para buscar IDs de special tokens específicos.

        Returns:
            El ID del token, o None si no existe en el vocabulario.
        """
        self._check_trained("token_to_id")
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, id: int) -> str | None:
        """Convierte un ID numérico al token (string) correspondiente.

        Útil para debug — para ver qué token tiene un ID específico.
        """
        self._check_trained("id_to_token")
        return self._tokenizer.id_to_token(id)

    # ===================================================================
    # INTERNOS
    # ===================================================================

    def _check_trained(self, method: str) -> None:
        """Verifica que el tokenizer esté entrenado antes de usarlo.

        Si alguien intenta encode/decode sin haber entrenado o cargado
        el tokenizer, este método lanza un error claro explicando qué hacer.
        """
        if not self._trained:
            raise RuntimeError(
                f"No puedes llamar a {method}() sin entrenar el tokenizer primero. "
                "Usa tokenizer.train(files=[...]) o NanoLogicTokenizer.load(dir)."
            )

    def __repr__(self) -> str:
        """Representación legible del tokenizer para debug."""
        if self._trained:
            return f"NanoLogicTokenizer(vocab_size={self.vocab_size}, trained=True)"
        return "NanoLogicTokenizer(trained=False)"
