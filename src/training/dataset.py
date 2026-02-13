"""
NanoLogicDataset — El encargado de alimentar al modelo.

Este archivo implementa un dataset inteligente con varios TRICKS PRO:

1. On-the-Fly Tokenization:
   No guardamos tensores gigantes en disco. Guardamos texto (JSONL) y
   tokenizamos al vuelo. Esto permite data augmentation dinámica en el futuro
   y ahorra espacio en disco.

2. Smart Masking (Loss Focusing):
   No queremos que el modelo aprenda a predecir el PADDING (es fácil e inútil).
   Tampoco queremos que aprenda a copiar el PROMPT (ya se lo damos).
   Solo queremos que aprenda a generar la RESPUESTA (Reasoning + Formula).

   Implementación:
   - Input:  [BOS] P R O M P T [OUTPUT] R E S P U E S T A [EOS] [PAD] ...
   - Target: [-100] ... [-100] [OUTPUT] R E S P U E S T A [EOS] [-100] ...

   Todo lo que es [-100] es IGNORADO por la loss function (CrossEntropy).
   El gradiente solo se enfoca en la respuesta. Esto acelera el aprendizaje
   y mejora la calidad de la generación.

3. Dynamic Padding (Collator):
   En vez de rellenar todo a max_seq_len (1024), rellenamos al tamaño
   de la secuencia MAS LARGA DEL BATCH.
   Si un batch tiene oraciones cortas (max 40 tokens), el tensor será (B, 40).
   Esto ahorra ~50-80% de cómputo en entrenamiento y es 100% equivalente matemáticas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.tokenizer.special_tokens import SPECIAL_TOKENS
from src.tokenizer.tokenizer import NanoLogicTokenizer


class NanoLogicDataset(Dataset):
    """Dataset que carga ejemplos JSONL y los tokeniza al vuelo.

    Formatos esperados en JSONL:
    {
        "sequence": "<|bos|><|input|> ... <|output|> ... <|eos|>"
    }
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: NanoLogicTokenizer,
        max_seq_len: int = 1024,
    ) -> None:
        """
        Args:
            data_path: Path al archivo .jsonl (train, val o test).
            tokenizer: Instancia de NanoLogicTokenizer ya entrenada.
            max_seq_len: Largo máximo para truncar (si es necesario).
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Cargar datos en memoria (son pocos, ~10MB, cabe sobrado en RAM).
        # Para datasets gigantes (GB/TB), usaríamos un IterableDataset con streaming.
        self.examples = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

        print(f"Dataset cargado: {len(self.examples)} ejemplos desde {self.data_path.name}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Obtiene un ejemplo, lo tokeniza y prepara los targets.

        Returns:
            Dict con:
            - input_ids: Lista de IDs de tokens.
            - labels: Lista de IDs para loss (con -100 en prompt y padding).
            - text: Texto original (para debug).
        """
        example = self.examples[idx]
        text = example["sequence"]

        # 1. Tokenizar (texto -> [ids])
        # encode devuelve una lista de python ints, no tensores todavía
        input_ids = self.tokenizer.encode(text)

        # 2. Truncar si es necesario (rara vez pasa con nuestros datos)
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]

        # 3. Crear LABELS (targets para la loss)
        # Inicialmente son una copia de los inputs
        labels = input_ids.copy()

        # 4. SMART MASKING: Ignorar el Prompt en la Loss
        # Encontrar dónde empieza la respuesta (token <|output|>)
        # Todo lo ANTERIOR a <|output|> se marca con -100 (ignorar).
        output_token_id = self.tokenizer.token_to_id(SPECIAL_TOKENS.OUTPUT)

        try:
            # Buscar el índice del token separator
            split_idx = labels.index(output_token_id)
            # Enmascarar todo hasta el separator (inclusive)
            # Queremos predecir LO QUE SIGUE al separator, no el separator mismo.
            # (Aunque a veces se deja que aprenda el separator, aquí somos estrictos).
            for i in range(split_idx + 1):
                labels[i] = -100
        except ValueError:
            # Si no encuentra <|output|>, algo raro pasa (ejemplo mal formado).
            # Enmascaramos todo para no contaminar el modelo (loss=0 para este ejemplo).
            labels = [-100] * len(labels)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "text": text,  # Útil para visualizar en wandb/tensorboard
        }


class DynamicCollateCallback:
    """Collate function (callable) para DataLoader con Dynamic Padding.

    Esta clase hace la magia de convertir una lista de ejemplos (de __getitem__)
    en un batch (tensores apilados), rellenando SOLO lo necesario.
    """

    def __init__(self, pad_token_id: int, ignore_index: int = -100) -> None:
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Args:
            batch: Lista de dicts que devuelve el Dataset.__getitem__

        Returns:
            Dict de tensores listos para la GPU.
        """
        # Extraer listas de input_ids y labels
        input_ids_list = [item["input_ids"] for item in batch]
        labels_list = [item["labels"] for item in batch]

        # Calcular el largo máximo EN ESTE BATCH (no en todo el dataset)
        max_len = max(len(ids) for ids in input_ids_list)

        # Preparar tensores de salida
        batch_size = len(batch)

        # Llenar inputs con PAD_TOKEN_ID
        input_ids_tensor = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)

        # Llenar labels con IGNORE_INDEX (-100)
        labels_tensor = torch.full((batch_size, max_len), self.ignore_index, dtype=torch.long)

        # Copiar los datos reales a los tensores
        for i, (ids, labs) in enumerate(zip(input_ids_list, labels_list)):
            # Copiar hasta el largo de este ejemplo
            curr_len = len(ids)
            input_ids_tensor[i, :curr_len] = torch.tensor(ids, dtype=torch.long)
            labels_tensor[i, :curr_len] = torch.tensor(labs, dtype=torch.long)

        # Attention Mask: 1 donde hay datos, 0 donde hay padding
        # Esto le dice al mecanismo de Attention qué parte ignorar.
        attention_mask = (input_ids_tensor != self.pad_token_id).long()

        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
            "attention_mask": attention_mask,
            # "texts": [item["text"] for item in batch] # Opcional, si queremos loggear
        }
