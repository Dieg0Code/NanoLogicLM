"""
TransformerBlock — la unidad LEGO que se repite en el modelo.

Nuestro modelo tiene 8 bloques identicos apilados. Cada bloque:
1. Normaliza con RMSNorm
2. Aplica Attention (con todos los tricks)
3. Suma el residual (con Deep Norm alpha)
4. Normaliza con RMSNorm otra vez
5. Aplica FFN SwiGLU
6. Suma el residual (con Deep Norm alpha)

Cada capa aprende relaciones diferentes:
- Capas tempranas (1-3): relaciones simples (sintaxis, palabras cercanas)
- Capas medias (4-6): relaciones semanticas (significado, roles)
- Capas tardias (7-8): relaciones abstractas (logica, implicaciones)

Pre-Norm (estilo LLaMA):
    El RMSNorm va ANTES del sublayer, no despues.
    Esto deja la conexion residual como un camino limpio sin obstaculos.

    Post-Norm (GPT-2):   output = Norm(x + sublayer(x))
    Pre-Norm (LLaMA):    output = x + sublayer(Norm(x))

    Pre-Norm entrena mas establemente, especialmente con modelos profundos.

Deep Norm:
    Escalar el residual por alpha > 1 para que la señal original
    llegue fuerte a las capas profundas:
    output = x * alpha + sublayer(Norm(x))

    alpha = (2 * n_layers) ** 0.25 = 2.0 con 8 capas.

Stochastic Depth:
    Durante entrenamiento, saltarse bloques completos al azar.
    Las capas profundas se saltean mas que las tempranas.
    En inferencia, todos los bloques se ejecutan.
    Funciona como regularizacion — previene overfitting con pocos datos.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.attention import CausalAttention
from src.model.config import NanoLogicConfig
from src.model.ffn import SwiGLU
from src.model.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """Un bloque del Transformer Decoder.

    Flujo:
        input
          |
          |---- [RMSNorm] -> [Attention] -----|
          |                                    |
          |--- (* alpha) -------------------- (+) -- residual 1
                                               |
          |---- [RMSNorm] -> [SwiGLU FFN] ----|
          |                                    |
          |--- (* alpha) -------------------- (+) -- residual 2
                                               |
                                            output

    Args:
        config: Configuracion del modelo.
        layer_idx: Indice de la capa (0-7). Se usa para:
            - Stochastic Depth (capas profundas se dropean mas)
            - Debugging/logging
    """

    def __init__(self, config: NanoLogicConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # ---------------------------------------------------------------
        # Componentes del bloque
        # ---------------------------------------------------------------

        # Pre-norm para Attention
        # Se aplica ANTES de attention (Pre-Norm style)
        self.attn_norm = RMSNorm(config.d_model)

        # Attention con todos los tricks (GQA, RoPE, QK-Norm, etc.)
        self.attention = CausalAttention(config)

        # Pre-norm para FFN
        self.ffn_norm = RMSNorm(config.d_model)

        # FFN SwiGLU con Gate Residual
        self.ffn = SwiGLU(config)

        # ---------------------------------------------------------------
        # Deep Norm: factor de escalamiento para residuales
        # ---------------------------------------------------------------
        # alpha > 1 amplifica la señal original en cada conexion residual.
        # Con 8 capas: alpha = 2.0
        # Si deep_norm esta desactivado: alpha = 1.0 (sin efecto)
        self.deep_norm_alpha = config.deep_norm_alpha

        # ---------------------------------------------------------------
        # Stochastic Depth: probabilidad de skip para esta capa
        # ---------------------------------------------------------------
        # Las capas tempranas (idx=0) nunca se skipean.
        # Las capas tardias (idx=7) se skipean con mayor probabilidad.
        # Progresion lineal: p = rate * (idx / (n_layers - 1))
        if config.n_layers > 1:
            self.drop_prob = config.stochastic_depth_rate * (layer_idx / (config.n_layers - 1))
        else:
            self.drop_prob = 0.0

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass del bloque.

        Args:
            x: Tensor de forma (batch, seq_len, d_model)
            rope_freqs: Frecuencias RoPE pre-calculadas
            mask: Mascara causal (opcional)

        Returns:
            Tensor de forma (batch, seq_len, d_model)
        """
        # ============================================================
        # SUB-BLOQUE 1: Attention
        # ============================================================
        # Pre-Norm: normalizar ANTES del sublayer
        attn_out = self.attention(self.attn_norm(x), rope_freqs, mask)

        # Stochastic Depth: durante entrenamiento, a veces skipear
        attn_out = self._maybe_drop(attn_out)

        # Residual con Deep Norm: x * alpha + sublayer_output
        # alpha=2.0 amplifica la señal original para que no se pierda
        x = x * self.deep_norm_alpha + attn_out

        # ============================================================
        # SUB-BLOQUE 2: FFN (SwiGLU)
        # ============================================================
        # Pre-Norm: normalizar ANTES del sublayer
        ffn_out = self.ffn(self.ffn_norm(x))

        # Stochastic Depth
        ffn_out = self._maybe_drop(ffn_out)

        # Residual con Deep Norm
        x = x * self.deep_norm_alpha + ffn_out

        return x

    def _maybe_drop(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica Stochastic Depth: dropear la salida del sublayer.

        Durante entrenamiento:
            - Con probabilidad drop_prob: retornar 0 (skip)
            - Con probabilidad (1-drop_prob): retornar x / (1-drop_prob) (escalar)

        Durante inferencia:
            - Siempre retornar x (sin cambios)

        El scaling por 1/(1-p) compensa el hecho de que a veces dropeamos.
        Es el mismo principio que Dropout:
            Entrenamiento: algunos valores = 0, el resto escalado
            Inferencia:    todos los valores, sin scaling

        Ejemplo con drop_prob=0.1:
            Entrenamiento: 10% de las veces output=0, 90% output=x/0.9
            Inferencia:    100% output=x
            Promedio en ambos: E[output] = 0.9 * x/0.9 = x  (match!)
        """
        if not self.training or self.drop_prob == 0.0:
            return x

        # Generar decision aleatoria: ¿skipear este bloque?
        # shape (1,1,1) para que aplique a todo el batch por igual
        keep_prob = 1.0 - self.drop_prob
        random_tensor = torch.rand(1, 1, 1, device=x.device, dtype=x.dtype)

        if random_tensor.item() < self.drop_prob:
            # SKIP: retornar ceros (este bloque no contribuye nada)
            return torch.zeros_like(x)
        else:
            # KEEP: escalar para compensar los skips
            return x / keep_prob
