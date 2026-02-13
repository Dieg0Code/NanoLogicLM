"""
NanoLogicTransformer — el modelo completo, ensamblado.

Este es el archivo FINAL que une todos los componentes en un Transformer
Decoder-Only funcional. Es el "cerebro" completo de NanoLogic.

Flujo completo:
    Token IDs [45, 892, 12, 567]
         |
    ┌─ Embedding ────────────────┐  ID → vector de 512 dims
    │  * sqrt(d_model)           │  escalar para estabilidad
    │  + Dropout                 │  regularizacion
    └──────────┬─────────────────┘
               |
         (8 veces)
    ┌─ TransformerBlock ─────────┐  Attention + FFN + Deep Norm
    └──────────┬─────────────────┘
               |
    ┌─ RMSNorm final ────────────┐  normalizar antes del head
    └──────────┬─────────────────┘
               |
    ┌─ LM Head ──────────────────┐  512 → 8000 (logits)
    │  * head_scale              │  calibracion aprendible
    │  tanh soft-capping         │  anti-overconfidence
    └──────────┬─────────────────┘
               |
         logits → loss / token predicho

Tricks implementados en este archivo:
    1. Weight Tying: Embedding y LM Head comparten pesos (-4.1M params).
    2. Embed Scale: multiplicar embeddings por sqrt(d_model) para estabilidad.
    3. Deep Norm beta init: escalar inicializacion de sublayers.
    4. Output Soft-Capping: limitar logits finales con tanh (Gemma 2).
    5. Head Scale: factor de escala aprendible para logits de salida.
    6. Z-Loss: flag para regularizacion en el training loop (PaLM).

Weight Tying — por que funciona:
    Embedding (8000 x 512) convierte token_id → vector (buscar significado).
    LM Head (512 x 8000) convierte vector → token_id (predecir token).
    Son operaciones INVERSAS. Compartir pesos fuerza consistencia:
    "el significado de una palabra y su prediccion deben ser lo mismo".

    Sin tying:  4.1M + 4.1M = 8.2M params
    Con tying:  4.1M compartidos  (50% ahorro!)
    Bonus: actua como regularizacion — mejor generalizacion.
    Lo usan: GPT-2, LLaMA, Mistral, Gemma — TODOS.

Embed Scale:
    Los embeddings se inicializan con valores ~N(0, 1).
    El modelo espera magnitudes ~sqrt(d_model) ≈ 22.6.
    Sin escalar, los embeddings empiezan demasiado chicos comparados
    con las activaciones de capas posteriores.
    Paper original: "Attention Is All You Need" (Vaswani et al., 2017).

Output Soft-Capping:
    logits = cap * tanh(logits / cap)
    Con cap=30, los logits se limitan suavemente a [-30, 30].
    Previene overconfidence: el modelo no puede estar 100% seguro.
    Lo usa Gemma 2. Complementario a z-loss.

Head Scale:
    logits = lm_head(x) * head_scale
    head_scale es un parametro aprendible que empieza en 1.0.
    El modelo puede "calibrar" su confianza:
    - head_scale < 1 → distribuciones mas suaves (cauteloso)
    - head_scale > 1 → distribuciones mas picudas (seguro)
    Un solo parametro, zero riesgo.

Deep Norm beta init:
    Los pesos de sublayers (attention, FFN) se inicializan escalados por:
    beta = (8 * n_layers) ** -0.25 = 0.354 con 8 capas.
    Esto asegura que al inicio del entrenamiento, los residuales
    dominen y los sublayers contribuyan poco. El modelo aprende
    gradualmente a "abrir" los sublayers.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.model.attention import precompute_rope_frequencies
from src.model.block import TransformerBlock
from src.model.config import NanoLogicConfig
from src.model.rmsnorm import RMSNorm


class NanoLogicTransformer(nn.Module):
    """Decoder-Only Transformer para NanoLogic.

    Parametros totales (~26M):
        Embedding:      8000 x 512 = 4,096,000  (compartido con LM Head)
        8 x Attention:  8 x ~1.3M  = 10,485,760
        8 x FFN:        8 x ~2.1M  = 16,773,120
        17 x RMSNorm:   17 x 512   =     8,704
        head_scale:                         1
        Total:                      ~25,267,585

    Args:
        config: NanoLogicConfig con todos los hiperparametros.
    """

    def __init__(self, config: NanoLogicConfig) -> None:
        super().__init__()
        self.config = config

        # ===============================================================
        # CAPA DE EMBEDDING
        # ===============================================================
        # Tabla de lookup: cada token_id (0-7999) tiene un vector de 512 dims.
        # Es la primera "traduccion": numero → significado.
        # padding_idx: el token de padding no genera gradientes
        # (no queremos que el modelo "aprenda" sobre el padding).
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )

        # Dropout para embeddings (regularizacion)
        self.embed_dropout = nn.Dropout(config.dropout)

        # Factor de escala para embeddings
        # sqrt(512) ≈ 22.6 — amplifica embeddings a la magnitud esperada
        self.embed_scale = math.sqrt(config.d_model) if config.embed_scale else 1.0

        # ===============================================================
        # BLOQUES DEL TRANSFORMER
        # ===============================================================
        # nn.ModuleList registra los bloques como submodulos de PyTorch.
        # Cada bloque tiene sus propios pesos (no compartidos).
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)]
        )

        # ===============================================================
        # NORMALIZACION FINAL
        # ===============================================================
        # Un ultimo RMSNorm antes del LM Head.
        # Sin esto, la magnitud de las activaciones despues de 8 capas
        # puede ser inestable y los logits serian erraticos.
        self.final_norm = RMSNorm(config.d_model)

        # ===============================================================
        # LM HEAD (Language Model Head)
        # ===============================================================
        # Proyeccion final: d_model (512) → vocab_size (8000)
        # Cada posicion produce un vector de 8000 logits.
        # El logit mas alto indica el token mas probable.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # ===============================================================
        # WEIGHT TYING: compartir pesos entre Embedding y LM Head
        # ===============================================================
        # Embedding.weight tiene forma (vocab_size, d_model)
        # lm_head.weight tiene forma (vocab_size, d_model)
        # SON LA MISMA MATRIZ. Esto:
        # 1. Ahorra 4.1M params (16% del modelo)
        # 2. Fuerza consistencia semantica
        # 3. Actua como regularizacion
        self.lm_head.weight = self.token_embedding.weight

        # ===============================================================
        # HEAD SCALE: calibracion aprendible de logits
        # ===============================================================
        if config.head_scale:
            # Empieza en 1.0 → sin efecto al inicio.
            # El modelo aprende a subir/bajar la "temperatura" de la salida.
            self.head_scale_param = nn.Parameter(torch.tensor(1.0))

        # ===============================================================
        # RoPE: Pre-calcular frecuencias de rotacion
        # ===============================================================
        # Estas NO son parametros entrenables — son constantes matematicas.
        # register_buffer las guarda como parte del modelo (se mueven a GPU
        # automaticamente) pero NO se incluyen en los gradientes.
        rope_freqs = precompute_rope_frequencies(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

        # ===============================================================
        # INICIALIZACION DE PESOS
        # ===============================================================
        self._init_weights()

    def _init_weights(self) -> None:
        """Inicializar pesos con Deep Norm beta scaling.

        La inicializacion es CRITICA. Sin inicializacion adecuada:
        - Los gradientes explotan o se desvanecen desde el paso 1
        - El entrenamiento diverge inmediatamente
        - Se pierden horas de GPU buscando el problema

        Esquema (Deep Norm paper):
        - Embeddings: N(0, 1) — inicializacion estandar
        - Sublayers (attn, FFN): xavier_uniform * beta
          beta = (8 * n_layers) ** -0.25 = 0.354
          Esto asegura que al inicio los sublayers contribuyan POCO
          y los residuales dominen. El modelo gradualmente aprende
          a "abrir" los sublayers.
        - Biases: 0 (no usamos bias, pero por si acaso)
        """
        beta = self.config.deep_norm_beta

        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform: distribucion uniforme escalada por fan_in/fan_out.
                # Es mejor que N(0, 1) porque considera el tamaño de la capa.
                nn.init.xavier_uniform_(module.weight)

                # Deep Norm beta scaling: encoger pesos de sublayers.
                # No aplicar a lm_head (esta tied con embedding).
                if module is not self.lm_head and self.config.deep_norm:
                    module.weight.data *= beta

                # Bias (por si alguna capa lo tiene)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                # Embeddings: N(0, 1) es el estandar.
                nn.init.normal_(module.weight, mean=0.0, std=1.0)

                # Padding token: forzar a cero (no tiene significado)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass completo del modelo.

        Args:
            input_ids: Token IDs de forma (batch, seq_len)
            targets: Token IDs objetivo para calcular loss (opcional).
                     Si se proveen, retorna loss ademas de logits.

        Returns:
            Dict con:
                "logits": (batch, seq_len, vocab_size) — predicciones
                "loss": escalar — cross entropy loss (solo si targets)
                "z_loss": escalar — z-loss regularization (solo si targets)

        Ejemplo:
            input:  [BOS, "Si", "llueve", "me"]
            target: ["Si", "llueve", "me", "mojo"]

            El modelo predice el SIGUIENTE token en cada posicion.
            BOS → deberia predecir "Si"
            "Si" → deberia predecir "llueve"
            "llueve" → deberia predecir "me"
            "me" → deberia predecir "mojo"
        """
        # ============================================================
        # PASO 1: Embedding + Scale
        # ============================================================
        # (batch, seq_len) → (batch, seq_len, d_model)
        x = self.token_embedding(input_ids) * self.embed_scale

        # Dropout sobre embeddings
        x = self.embed_dropout(x)

        # ============================================================
        # PASO 2: Pasar por los 8 bloques
        # ============================================================
        for layer in self.layers:
            x = layer(x, self.rope_freqs)

        # ============================================================
        # PASO 3: Normalizacion final
        # ============================================================
        x = self.final_norm(x)

        # ============================================================
        # PASO 4: LM Head → logits
        # ============================================================
        # (batch, seq_len, d_model) → (batch, seq_len, vocab_size)
        logits = self.lm_head(x)

        # Head Scale: calibracion aprendible
        if self.config.head_scale:
            logits = logits * self.head_scale_param

        # Output Soft-Capping: limitar logits
        if self.config.output_cap > 0:
            logits = self.config.output_cap * torch.tanh(logits / self.config.output_cap)

        # ============================================================
        # PASO 5: Loss (si hay targets)
        # ============================================================
        result: dict[str, torch.Tensor] = {"logits": logits}

        if targets is not None:
            # Cross Entropy Loss
            # Reshape: (batch*seq, vocab_size) vs (batch*seq,)
            # ignore_index: ignorar posiciones con padding (-100 es convencion)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

            # Z-Loss (PaLM): penalizar logits grandes
            if self.config.z_loss_weight > 0:
                # logsumexp(logits)^2 → penaliza si los logits son grandes
                # Complementa output_cap: cap limita hard, z_loss enseña soft
                z_loss = self.config.z_loss_weight * torch.mean(
                    torch.logsumexp(logits.view(-1, logits.size(-1)), dim=-1) ** 2
                )
                result["z_loss"] = z_loss
                result["loss"] = loss + z_loss

        return result

    def count_parameters(self) -> dict[str, int]:
        """Contar parametros del modelo por componente.

        Returns:
            Dict con el conteo de parametros por componente.
        """
        embedding = sum(p.numel() for n, p in self.named_parameters() if "token_embedding" in n)
        attention = sum(
            p.numel()
            for n, p in self.named_parameters()
            if any(
                k in n
                for k in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm", "diff_lambda")
            )
        )
        ffn = sum(
            p.numel()
            for n, p in self.named_parameters()
            if any(k in n for k in ("gate_proj", "up_proj", "down_proj", "gate_residual"))
        )
        norms = sum(
            p.numel()
            for n, p in self.named_parameters()
            if "norm" in n and "q_norm" not in n and "k_norm" not in n
        )
        other = sum(p.numel() for n, p in self.named_parameters() if "head_scale" in n)

        total = sum(p.numel() for p in self.parameters())
        # Nota: lm_head comparte pesos con embedding (weight tying)
        # por eso no se cuenta por separado.

        return {
            "embedding (shared w/ lm_head)": embedding,
            "attention (8 layers)": attention,
            "ffn (8 layers)": ffn,
            "norms": norms,
            "other (head_scale, etc)": other,
            "total": total,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Generar tokens autregresivamente.

        El modelo genera UN token a la vez:
        1. Forward pass con la secuencia actual
        2. Tomar los logits de la ULTIMA posicion
        3. Aplicar temperature y top-k sampling
        4. Agregar el token generado a la secuencia
        5. Repetir

        Args:
            input_ids: Tokens iniciales (batch, seq_len)
            max_new_tokens: Cuantos tokens generar
            temperature: Controla creatividad (0.0=greedy, 1.0=mas random)
            top_k: Solo considerar los k tokens mas probables

        Returns:
            Secuencia completa (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Truncar al max_seq_len si la secuencia se hace muy larga
            idx_cond = input_ids
            if input_ids.size(1) > self.config.max_seq_len:
                idx_cond = input_ids[:, -self.config.max_seq_len :]

            # Forward pass
            outputs = self(idx_cond)
            logits = outputs["logits"]

            # Solo nos interesan los logits de la ULTIMA posicion
            logits = logits[:, -1, :]  # (batch, vocab_size)

            # Temperature scaling
            if temperature > 0:
                logits = logits / temperature

                # Top-k: solo considerar los k tokens mas probables
                if top_k > 0:
                    # Poner -inf en todo excepto los top_k
                    top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    min_top_k = top_k_vals[:, -1].unsqueeze(-1)
                    logits = torch.where(
                        logits < min_top_k,
                        torch.full_like(logits, float("-inf")),
                        logits,
                    )

                # Sampling: elegir segun probabilidades
                probs = nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy: elegir el mas probable
                next_token = logits.argmax(dim=-1, keepdim=True)

            # Agregar el token generado a la secuencia
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Early stop si generamos EOS
            if next_token.item() == self.config.eos_token_id:
                break

        return input_ids
