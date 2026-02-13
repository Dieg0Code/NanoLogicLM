"""
SwiGLU Feed-Forward Network (FFN) — la "memoria interna" del Transformer.

¿Que hace la FFN?
    Si attention es "a quien presto atencion", la FFN es
    "ahora que ya se que es relevante, que hago con esa informacion?"

    Attention RECOPILA informacion de otros tokens.
    FFN PROCESA esa informacion → "piensa".

    Estudios (Geva et al., 2021) demostraron que las FFN funcionan como
    una MEMORIA ASOCIATIVA: cada neurona almacena un patron (key) y
    una respuesta (value). Cuando el input coincide con un patron,
    la neurona se activa y aporta su respuesta.

¿Que es SwiGLU?
    Es una version mejorada de la FFN clasica que usa una "compuerta"
    (gate) para filtrar selectivamente que informacion dejar pasar.

    FFN clasica (GPT-2):
        x = Linear(512 -> 2048)   # expandir
        x = ReLU(x)               # no-linealidad
        x = Linear(2048 -> 512)   # comprimir

    SwiGLU (LLaMA/Mistral/Gemma):
        gate = Linear(512 -> 1365)     # la compuerta
        up   = Linear(512 -> 1365)     # el contenido
        x    = SiLU(gate) * up         # compuerta filtra el contenido
        x    = Linear(1365 -> 512)     # comprimir

    ¿Por que es mejor?
    - ReLU es un filtro binario: pasa o no pasa (0 o x)
    - SiLU es un filtro suave: puede dejar pasar "un poco"
    - La COMPUERTA aprende QUE dimensiones bloquear para cada token
    - Ejemplo: para "llueve", deja pasar dims de clima, bloquea dims de sintaxis

    Costo: 3 matrices en vez de 2 → se compensa reduciendo d_ff:
        FFN clasica:  d_ff = 4 x d_model = 2048
        FFN SwiGLU:   d_ff = (8/3) x d_model = 1365
        Mismos parametros totales, mejor rendimiento.

    Lo usan: LLaMA 1/2/3, Mistral, Gemma, PaLM — todos desde 2023.

Gate Residual (underground trick):
    Agregar un bypass dentro de la compuerta:
        output = SiLU(gate) * up + alpha * up

    Si la compuerta se equivoca y bloquea algo importante,
    el termino alpha * up permite que pase igualmente.
    Alpha empieza en 0 (sin efecto) y el modelo aprende si necesita usarlo.

    Costo: 1 parametro escalar extra. Riesgo: cero.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import NanoLogicConfig


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network con Gate Residual opcional.

    Arquitectura:
        input (batch, seq, d_model=512)
            |
            v
        [gate_proj]  →  SiLU(gate)  ─── *  ── (+alpha * up si gate_residual) ──┐
        [up_proj]    →  up  ────────────┘                                       │
                                                                                v
                                                                          [down_proj]
                                                                                │
                                                                                v
                                                                    output (batch, seq, d_model=512)

    Parametros (con d_model=512, d_ff=1365):
        gate_proj:  512 x 1365 =  698,880
        up_proj:    512 x 1365 =  698,880
        down_proj:  1365 x 512 =  698,880
        Total:                   2,096,640 por capa
        x 8 capas:             16,773,120 (~64% del modelo total)

    La FFN es por lejos el componente MAS caro en parametros.
    Por eso la eleccion de SwiGLU (que es mas eficiente) importa tanto.
    """

    def __init__(self, config: NanoLogicConfig) -> None:
        super().__init__()

        # ---------------------------------------------------------------
        # Las 3 proyecciones de SwiGLU
        # ---------------------------------------------------------------
        # gate_proj: genera los valores para la compuerta.
        # SiLU(gate_proj(x)) produce valores entre -0.28 y +inf.
        # Estos valores se multiplican con up_proj(x) para filtrar.
        # bias=False: como en attention, los modelos modernos no usan bias.
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)

        # up_proj: genera el "contenido" que la compuerta filtra.
        # Este es el camino principal de la informacion.
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)

        # down_proj: comprime el resultado de vuelta a d_model.
        # d_ff (1365) → d_model (512)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

        # ---------------------------------------------------------------
        # Gate Residual (underground trick)
        # ---------------------------------------------------------------
        self.use_gate_residual = config.gate_residual
        if self.use_gate_residual:
            # Alpha empieza en 0.0 → al inicio, no tiene efecto.
            # El modelo puede aprender a "abrirlo" durante el entrenamiento
            # si necesita dejar pasar informacion sin filtrar.
            self.gate_residual_alpha = nn.Parameter(torch.tensor(0.0))

        # Dropout para regularizacion
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass de la FFN SwiGLU.

        Args:
            x: Tensor de forma (batch, seq_len, d_model)

        Returns:
            Tensor de forma (batch, seq_len, d_model)

        Paso a paso (valores ejemplo para un solo token):
            x = [0.5, -1.0, 2.0, ...]  (512 dims)

            1. gate = gate_proj(x)  → [1.2, -0.5, 0.8, ...]  (1365 dims)
            2. up   = up_proj(x)    → [0.3,  0.7, -0.2, ...] (1365 dims)
            3. SiLU(gate)           → [0.85, -0.19, 0.54, ...]
               SiLU(x) = x * sigmoid(x) — una curva suave que:
               - Valores positivos grandes → pasan casi intactos
               - Valores cercanos a 0 → se reducen
               - Valores negativos → se aplastan hacia 0 (pero suave)
            4. gate * up            → [0.26, -0.13, -0.11, ...]
               (la compuerta filtro el contenido)
            5. [Si gate_residual] + alpha * up  → bypass opcional
            6. down_proj(...)       → [0.1, -0.3, 0.7, ...]  (512 dims)
        """
        # Calcular gate y up en paralelo
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # SwiGLU: SiLU(gate) * up
        # F.silu = x * sigmoid(x), tambien conocido como "swish"
        hidden = F.silu(gate) * up

        # Gate Residual: safety net
        if self.use_gate_residual:
            # alpha * up permite que la informacion pase sin filtrar
            # si la compuerta se equivoca. Alpha empieza en 0.
            hidden = hidden + self.gate_residual_alpha * up

        # Comprimir de vuelta a d_model y aplicar dropout
        return self.dropout(self.down_proj(hidden))
