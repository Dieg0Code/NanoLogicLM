"""
NanoLogicDataset — carga, tokeniza y empaqueta datos para entrenamiento.

Este archivo es el "cocinero" que transforma archivos JSONL crudos en
tensores listos para alimentar al modelo. Implementa TODAS las tecnicas
de eficiencia de datos del state of the art + underground.

Flujo completo:
    Archivo JSONL → Dataset → Collator → DataLoader → Modelo

    1. Cargar JSONL: leer secuencias ya formateadas con special tokens
    2. Pre-tokenizar: convertir texto → IDs UNA VEZ (no repetir cada epoch)
    3. Curriculum: filtrar por complejidad segun el epoch actual
    4. Bucketing: agrupar secuencias de largo similar
    5. Packing: empaquetar multiples ejemplos en una secuencia (zero waste)
    6. Dynamic Padding: rellenar al maximo del batch, no al maximo global
    7. Document Mask: mask que impide cross-attention entre documentos packed

Tricks implementados:
    1. Pre-tokenizacion offline: tokenizar una vez, cachear en memoria.
       Ahorra 3+ seg por epoch. Con 6K datos caben en RAM sin problema.

    2. Dynamic Padding: padear al max del batch, no a max_seq_len.
       Si un batch tiene secuencias de 30 tokens, padea a 30 (no 1024).
       Speedup: 3-5x vs padding fijo.

    3. Length Bucketing: agrupar secuencias de largo similar en el mismo batch.
       Complementa Dynamic Padding: minimiza el padding dentro de cada batch.
       Speedup adicional: 2x sobre Dynamic Padding solo.

    4. Packing: empaquetar multiples ejemplos en una sola secuencia de largo
       max_seq_len. Zero desperdicio de tokens. Eficiencia: 95%+ vs 30% naive.
       Requiere Document Mask para evitar cross-contamination entre documentos.

    5. Curriculum Learning: entrenar primero con ejemplos faciles (Simple),
       luego medios (Intermediate), luego dificiles (Advanced).
       El modelo construye entendimiento de abajo hacia arriba.
       Nuestro dataset ya tiene la columna 'complexity' → gratis.

    6. Document Mask: mascara block-diagonal para packing.
       Cada documento solo puede atender a tokens de SU MISMO documento.
       Impide que "Si llueve me mojo" atienda a "El cielo es azul".
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from src.model.config import NanoLogicConfig
from src.tokenizer.tokenizer import NanoLogicTokenizer


# =====================================================================
# CONSTANTES DE CURRICULUM
# =====================================================================
# Mapeo de complejidad a "nivel" numerico para curriculum learning.
# El dataset tiene 3 niveles: simple, intermediate, advanced.

COMPLEXITY_LEVELS: dict[str, int] = {
    "simple": 0,
    "intermediate": 1,
    "advanced": 2,
}


# =====================================================================
# DATASET PRINCIPAL
# =====================================================================


class NanoLogicDataset(Dataset):
    """Dataset que carga, tokeniza y cachea ejemplos de NanoLogic.

    Este dataset hace pre-tokenizacion: convierte TODOS los textos a IDs
    al inicio y los guarda en memoria. Esto elimina la retokenizacion
    repetida en cada epoch (con 6K ejemplos, caben en RAM sin problema).

    Soporta curriculum learning: puede filtrar ejemplos por complejidad
    para exponer primero los faciles y luego los dificiles.

    Attributes:
        examples: Lista de dicts con IDs tokenizados y metadata.
        config: Configuracion del modelo (para max_seq_len, pad_token_id).
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: NanoLogicTokenizer,
        config: NanoLogicConfig,
        max_complexity: int = 2,
    ) -> None:
        """Carga y pre-tokeniza el dataset.

        Args:
            jsonl_path: Path al archivo JSONL procesado.
            tokenizer: Tokenizer BPE entrenado.
            config: Configuracion del modelo.
            max_complexity: Nivel maximo de complejidad a incluir.
                0 = solo Simple
                1 = Simple + Intermediate
                2 = Todo (default)
        """
        super().__init__()
        self.config = config

        # ===========================================================
        # PASO 1: Cargar y pre-tokenizar
        # ===========================================================
        # Leemos el JSONL, tokenizamos cada secuencia UNA VEZ,
        # y guardamos los IDs en memoria.
        self.examples: list[dict] = []
        path = Path(jsonl_path)

        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)

                # Curriculum: filtrar por complejidad
                complexity = COMPLEXITY_LEVELS.get(data.get("complexity", "simple"), 0)
                if complexity > max_complexity:
                    continue

                # Tokenizar la secuencia completa (ya tiene special tokens)
                token_ids = tokenizer.encode(data["sequence"])

                # Truncar a max_seq_len si es necesario
                if len(token_ids) > config.max_seq_len:
                    token_ids = token_ids[: config.max_seq_len]

                self.examples.append(
                    {
                        "input_ids": token_ids,
                        "length": len(token_ids),
                        "complexity": complexity,
                    }
                )

        # Ordenar por largo para facilitar bucketing
        self.examples.sort(key=lambda x: x["length"])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]

    def get_length_stats(self) -> dict[str, int | float]:
        """Estadisticas de longitud para debug."""
        lengths = [ex["length"] for ex in self.examples]
        return {
            "count": len(lengths),
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "median": sorted(lengths)[len(lengths) // 2],
        }


# =====================================================================
# PACKING: empaquetar multiples ejemplos en una secuencia
# =====================================================================
# En vez de tener 1 ejemplo por secuencia (con mucho padding),
# empaquetamos varios ejemplos hasta llenar max_seq_len.
#
# Ejemplo:
#   3 ejemplos: [30 tokens], [25 tokens], [40 tokens]
#   max_seq_len = 128
#   → Packed: [30 + 25 + 40 = 95 tokens, + 33 padding] = 1 secuencia
#   → doc_ids: [0,0,...,0, 1,1,...,1, 2,2,...,2, PAD,PAD,...,PAD]
#
# Los doc_ids se usan para generar la Document Mask en el collator.


def pack_examples(
    examples: list[dict],
    max_seq_len: int,
    pad_token_id: int,
) -> list[dict]:
    """Empacueta multiples ejemplos en secuencias de largo fijo.

    Algoritmo greedy: recorre los ejemplos en orden y va llenando
    la secuencia actual. Cuando no cabe el siguiente ejemplo, cierra
    la secuencia y empieza una nueva.

    Args:
        examples: Lista de dicts con 'input_ids' y 'length'.
        max_seq_len: Largo maximo de cada secuencia empaquetada.
        pad_token_id: ID del token de padding.

    Returns:
        Lista de dicts empaquetados, cada uno con:
            - input_ids: secuencia empaquetada (largo = max_seq_len)
            - doc_ids: ID de documento para cada token (para document mask)
            - n_docs: cantidad de documentos en esta secuencia
            - n_real_tokens: cantidad de tokens reales (sin padding)
    """
    packed: list[dict] = []
    current_ids: list[int] = []
    current_doc_ids: list[int] = []
    current_doc_idx = 0

    for example in examples:
        ids = example["input_ids"]
        length = len(ids)

        # ¿Cabe en la secuencia actual?
        if len(current_ids) + length <= max_seq_len:
            # Si cabe, agregar
            current_ids.extend(ids)
            current_doc_ids.extend([current_doc_idx] * length)
            current_doc_idx += 1
        else:
            # No cabe: cerrar secuencia actual (si tiene algo)
            if current_ids:
                n_real = len(current_ids)
                # Pad hasta max_seq_len
                n_pad = max_seq_len - n_real
                current_ids.extend([pad_token_id] * n_pad)
                # doc_id del padding = -1 (se ignora en la mask)
                current_doc_ids.extend([-1] * n_pad)

                packed.append(
                    {
                        "input_ids": current_ids,
                        "doc_ids": current_doc_ids,
                        "n_docs": current_doc_idx,
                        "n_real_tokens": n_real,
                    }
                )

            # Empezar nueva secuencia con este ejemplo
            current_ids = list(ids)
            current_doc_ids = [0] * length
            current_doc_idx = 1

    # Cerrar la ultima secuencia
    if current_ids:
        n_real = len(current_ids)
        n_pad = max_seq_len - n_real
        current_ids.extend([pad_token_id] * n_pad)
        current_doc_ids.extend([-1] * n_pad)

        packed.append(
            {
                "input_ids": current_ids,
                "doc_ids": current_doc_ids,
                "n_docs": current_doc_idx,
                "n_real_tokens": n_real,
            }
        )

    return packed


# =====================================================================
# DOCUMENT MASK: mascara block-diagonal para packing
# =====================================================================
# Cuando empaquetamos multiples documentos en una secuencia,
# necesitamos una mascara que:
# 1. Sea CAUSAL dentro de cada documento (token N ve tokens 0..N)
# 2. BLOQUEE atencion entre documentos diferentes
# 3. BLOQUEE atencion hacia/desde padding
#
# Resultado: una mascara block-diagonal causal.
#
# Ejemplo con 3 docs de 3, 2, 2 tokens + 1 pad (max_seq_len=8):
#
#   doc_ids = [0, 0, 0, 1, 1, 2, 2, -1]
#
#   Mask (0 = permitido, -inf = bloqueado):
#   pos:  0    1    2    3    4    5    6    7
#    0 [  0,  -∞,  -∞,  -∞,  -∞,  -∞,  -∞,  -∞ ]  doc 0
#    1 [  0,   0,  -∞,  -∞,  -∞,  -∞,  -∞,  -∞ ]  doc 0
#    2 [  0,   0,   0,  -∞,  -∞,  -∞,  -∞,  -∞ ]  doc 0
#    3 [ -∞,  -∞,  -∞,   0,  -∞,  -∞,  -∞,  -∞ ]  doc 1
#    4 [ -∞,  -∞,  -∞,   0,   0,  -∞,  -∞,  -∞ ]  doc 1
#    5 [ -∞,  -∞,  -∞,  -∞,  -∞,   0,  -∞,  -∞ ]  doc 2
#    6 [ -∞,  -∞,  -∞,  -∞,  -∞,   0,   0,  -∞ ]  doc 2
#    7 [ -∞,  -∞,  -∞,  -∞,  -∞,  -∞,  -∞,  -∞ ]  pad


def build_document_mask(
    doc_ids: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Construye la mascara block-diagonal causal para packing.

    Args:
        doc_ids: Tensor de IDs de documento por token. Shape: (seq_len,).
                 -1 marca padding.
        dtype: Tipo de dato para la mascara.

    Returns:
        Mascara de forma (seq_len, seq_len).
        0.0 = permitido, -inf = bloqueado.
    """
    seq_len = doc_ids.size(0)

    # Paso 1: Mascara de mismo documento
    # same_doc[i][j] = True si token i y token j pertenecen al mismo doc
    # (y ninguno es padding, es decir doc_id != -1)
    doc_ids_row = doc_ids.unsqueeze(1)  # (seq_len, 1)
    doc_ids_col = doc_ids.unsqueeze(0)  # (1, seq_len)
    same_doc = (doc_ids_row == doc_ids_col) & (doc_ids_row != -1) & (doc_ids_col != -1)

    # Paso 2: Mascara causal (triangular inferior)
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=doc_ids.device))

    # Paso 3: Combinar — permitido SOLO si mismo doc Y causal
    allowed = same_doc & causal

    # Paso 4: Convertir a formato de mascara (-inf donde bloqueado)
    mask = torch.where(
        allowed, torch.tensor(0.0, dtype=dtype), torch.tensor(float("-inf"), dtype=dtype)
    )

    return mask


# =====================================================================
# COLLATOR: Dynamic Padding + Document Mask
# =====================================================================
# El collator se ejecuta en el DataLoader para cada batch.
# Recibe una lista de ejemplos y los convierte en tensores.
#
# Dos modos:
# - packed=True:  los ejemplos ya estan empaquetados → solo tensorizar
# - packed=False: padding dinamico al max del batch → mas simple


class NanoLogicCollator:
    """Collator que prepara batches con Dynamic Padding y Document Mask.

    Responsabilidades:
    1. Convertir listas de IDs a tensores
    2. Construir input_ids y targets (shifted right)
    3. Construir attention_mask o document_mask segun modo
    4. Dynamic Padding: padear al max del batch, no al max global

    Targets (shifted right):
        input:  [BOS, Si, llueve, me, mojo, EOS]
        target: [Si,  llueve, me, mojo, EOS, PAD]

        El modelo predice el SIGUIENTE token en cada posicion.
        La ultima posicion no tiene target util → se ignora con -100.
    """

    def __init__(self, config: NanoLogicConfig, packed: bool = False) -> None:
        self.config = config
        self.packed = packed

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        if self.packed:
            return self._collate_packed(batch)
        return self._collate_dynamic(batch)

    def _collate_dynamic(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Dynamic Padding: padea al maximo del batch actual.

        Si el batch tiene secuencias de [30, 25, 28, 32] tokens,
        padea todo a 32 (no a 1024).
        """
        # Encontrar el largo maximo en ESTE batch
        max_len = max(ex["length"] for ex in batch)

        # Limitar a max_seq_len (por seguridad)
        max_len = min(max_len, self.config.max_seq_len)

        all_input_ids = []
        all_targets = []
        all_masks = []

        for ex in batch:
            ids = ex["input_ids"][:max_len]
            length = len(ids)

            # Padding
            n_pad = max_len - length
            padded_ids = ids + [self.config.pad_token_id] * n_pad

            # Targets: shifted right
            # input:  [BOS, A, B, C, EOS, PAD, PAD]
            # target: [A,   B, C, EOS, -100, -100, -100]
            # -100 = ignore_index en CrossEntropyLoss
            targets = padded_ids[1:] + [-100]
            # Enmascarar targets de padding
            for i in range(length - 1, max_len):
                targets[i] = -100

            # Attention mask: 1 = real, 0 = padding
            attn_mask = [1] * length + [0] * n_pad

            all_input_ids.append(padded_ids)
            all_targets.append(targets)
            all_masks.append(attn_mask)

        return {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
            "targets": torch.tensor(all_targets, dtype=torch.long),
            "attention_mask": torch.tensor(all_masks, dtype=torch.long),
        }

    def _collate_packed(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Empaquetado: ejemplos ya packed, solo tensorizar + document mask."""
        all_input_ids = []
        all_targets = []
        all_doc_masks = []

        for ex in batch:
            ids = ex["input_ids"]
            doc_ids = ex["doc_ids"]
            n_real = ex["n_real_tokens"]

            # Targets: shifted right (misma logica)
            targets = ids[1:] + [self.config.pad_token_id]

            # Enmascarar targets:
            # 1. Padding → -100
            # 2. Ultimo token de cada documento → -100
            #    (porque su "siguiente" token es el BOS de otro doc)
            targets_masked = list(targets)
            for i in range(len(targets_masked)):
                if i >= n_real:
                    # Padding
                    targets_masked[i] = -100
                elif i < len(doc_ids) - 1 and doc_ids[i] != doc_ids[i + 1]:
                    # Frontera entre documentos: el target seria el BOS
                    # del siguiente doc, que no tiene sentido predecir
                    targets_masked[i] = -100

            # Document mask
            doc_ids_tensor = torch.tensor(doc_ids, dtype=torch.long)
            doc_mask = build_document_mask(doc_ids_tensor)

            all_input_ids.append(ids)
            all_targets.append(targets_masked)
            all_doc_masks.append(doc_mask)

        return {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
            "targets": torch.tensor(all_targets, dtype=torch.long),
            "document_mask": torch.stack(all_doc_masks).unsqueeze(1),
        }


# =====================================================================
# BUCKET SAMPLER: agrupar secuencias similares
# =====================================================================
# En vez de shufflear completamente (lo que mezcla secuencias de 20
# tokens con secuencias de 300 tokens en el mismo batch), agrupamos
# secuencias por largo y shuffleamos DENTRO de cada grupo.
#
# Esto minimiza el padding en cada batch.


class BucketBatchSampler:
    """Sampler que agrupa secuencias de largo similar en batches.

    Algoritmo:
    1. Ordenar indices por largo de secuencia
    2. Crear "mega-batches" de tamaño bucket_size
    3. Dentro de cada mega-batch, shufflear
    4. Particionar en batches de tamaño batch_size
    5. Shufflear el orden de los batches

    Esto logra:
    - Secuencias dentro de cada batch tienen largo similar → poco padding
    - Orden de batches es aleatorio → no hay sesgo sistematico
    """

    def __init__(
        self,
        dataset: NanoLogicDataset,
        batch_size: int,
        bucket_size_multiplier: int = 10,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        """
        Args:
            dataset: El dataset (ya ordenado por largo).
            batch_size: Tamaño de cada batch.
            bucket_size_multiplier: Cuantos batches caben en un bucket.
                Con batch_size=8 y multiplier=10: bucket = 80 ejemplos.
                Dentro de esos 80, se shufflea y se hacen batches de 8.
            shuffle: Si True, shufflear dentro de buckets y orden de batches.
            drop_last: Si True, descartar el ultimo batch si es incompleto.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_size_multiplier
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        # Paso 1: Indices ordenados por largo (dataset ya esta ordenado)
        indices = list(range(len(self.dataset)))

        # Paso 2: Crear buckets
        buckets = [
            indices[i : i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)
        ]

        # Paso 3: Shufflear dentro de cada bucket
        if self.shuffle:
            for bucket in buckets:
                random.shuffle(bucket)

        # Paso 4: Aplanar y particionar en batches
        flat_indices = [idx for bucket in buckets for idx in bucket]
        batches = [
            flat_indices[i : i + self.batch_size]
            for i in range(0, len(flat_indices), self.batch_size)
        ]

        # Drop last batch si es incompleto
        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]

        # Paso 5: Shufflear orden de batches
        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        n = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size != 0:
            n += 1
        return n


# =====================================================================
# FABRICA DE DATALOADERS
# =====================================================================
# Funcion de conveniencia que ensambla todo: Dataset + Collator +
# Sampler + DataLoader.


def create_dataloader(
    jsonl_path: str | Path,
    tokenizer: NanoLogicTokenizer,
    config: NanoLogicConfig,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    max_complexity: int = 2,
    use_packing: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Crea un DataLoader listo para entrenar.

    Esta es la funcion que se llama desde el training loop.
    Ensambla todo el pipeline de datos.

    Args:
        jsonl_path: Path al archivo JSONL.
        tokenizer: Tokenizer entrenado.
        config: Configuracion del modelo.
        batch_size: Ejemplos por batch.
        shuffle: Si shufflear los datos.
        num_workers: Workers paralelos para cargar datos.
        max_complexity: Nivel max de complejidad (curriculum learning).
        use_packing: Si True, empaquetar multiples ejemplos por secuencia.
        drop_last: Si True, descartar ultimo batch incompleto.

    Returns:
        DataLoader listo para iterar.

    Ejemplo:
        # Sin packing (Dynamic Padding + Bucketing):
        loader = create_dataloader("train.jsonl", tokenizer, config, batch_size=8)

        # Con packing (maximo eficiencia):
        loader = create_dataloader("train.jsonl", tokenizer, config, use_packing=True)

        # Curriculum (solo ejemplos simples):
        loader = create_dataloader("train.jsonl", tokenizer, config, max_complexity=0)
    """
    # Paso 1: Crear dataset (pre-tokeniza todo)
    dataset = NanoLogicDataset(
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        config=config,
        max_complexity=max_complexity,
    )

    if use_packing:
        # ===========================================================
        # MODO PACKING: empaquetar y luego DataLoader simple
        # ===========================================================
        # Shufflear antes de empaquetar (para variedad)
        examples = list(dataset.examples)
        if shuffle:
            random.shuffle(examples)

        # Empaquetar
        packed = pack_examples(
            examples=examples,
            max_seq_len=config.max_seq_len,
            pad_token_id=config.pad_token_id,
        )

        # Crear dataset falso con los packed examples
        packed_dataset = _PackedDataset(packed)

        return DataLoader(
            packed_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=NanoLogicCollator(config, packed=True),
            drop_last=drop_last,
            pin_memory=True,
        )
    else:
        # ===========================================================
        # MODO DYNAMIC PADDING + BUCKETING
        # ===========================================================
        sampler = BucketBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=NanoLogicCollator(config, packed=False),
            pin_memory=True,
        )


class _PackedDataset(Dataset):
    """Wrapper trivial para packed examples."""

    def __init__(self, packed: list[dict]) -> None:
        self.packed = packed

    def __len__(self) -> int:
        return len(self.packed)

    def __getitem__(self, idx: int) -> dict:
        return self.packed[idx]
