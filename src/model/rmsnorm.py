"""
RMSNorm — Root Mean Square Layer Normalization.

¿Qué hace?
    Normaliza los valores dentro de cada vector para mantenerlos
    en un rango controlado. Sin normalización, los valores crecen
    o se achican capa tras capa y el modelo no puede entrenar.

¿Cómo funciona?
    1. Calcular el RMS (Root Mean Square) del vector:
       rms = sqrt(mean(x^2))

    2. Dividir cada elemento por el RMS:
       x_norm = x / rms

    3. Escalar por un parámetro aprendible (gamma):
       output = x_norm * gamma

    Gamma empieza en 1.0 y el modelo lo ajusta durante el entrenamiento.
    Esto le permite "des-normalizar" si necesita valores más grandes.

¿Por qué RMSNorm y no LayerNorm?
    LayerNorm hace: (x - mean(x)) / std(x) * gamma + beta
    RMSNorm hace:   x / rms(x) * gamma

    Diferencias:
    - RMSNorm NO resta la media → 1 operación menos
    - RMSNorm NO tiene bias (beta) → 1 parámetro menos
    - Paper: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
      demostró que la media aporta muy poco — eliminarla no cambia la calidad

    Resultado: misma calidad, ~10-15% más rápido.
    Lo usan: LLaMA, LLaMA 2, Mistral, Gemma, GPT-NeoX.

¿Dónde se usa en nuestro modelo?
    Se usa 17 veces:
    - 2 por capa (antes de attention + antes de FFN) x 8 capas = 16
    - 1 final antes del head de salida = 1
    Total: 17 instancias de RMSNorm

    Parámetros: solo gamma (d_model=512 valores por instancia)
    Total: 17 x 512 = 8,704 parámetros (0.03% del modelo)
    Es baratísimo en parámetros pero CRÍTICO para la estabilidad.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Args:
        dim: Dimensión del vector a normalizar (d_model en nuestro caso).
        eps: Epsilon para evitar división por cero. Default: 1e-6.
             Sin eps, si todos los valores son 0, dividimos por 0 → NaN.
             1e-6 es el estándar de LLaMA/Mistral.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()

        # eps: constante pequeña para estabilidad numérica.
        # Se suma al denominador para evitar dividir por exactamente 0.
        self.eps = eps

        # gamma (weight): parámetro aprendible de escala.
        # Empieza en 1.0 para todos los elementos.
        # nn.Parameter le dice a PyTorch: "esto es un parámetro entrenable,
        # calcula gradientes y actualízalo con el optimizador".
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica RMSNorm al tensor de entrada.

        Args:
            x: Tensor de forma (..., dim). El último eje es el que se normaliza.
               Típicamente (batch, seq_len, d_model).

        Returns:
            Tensor normalizado de la misma forma.

        Paso a paso:
            x = [0.5, -1.0, 2.0, 0.0]  (ejemplo con dim=4)

            1. x^2 = [0.25, 1.0, 4.0, 0.0]
            2. mean(x^2) = 1.3125
            3. rms = sqrt(1.3125 + eps) = 1.1456
            4. x_norm = x / rms = [0.436, -0.873, 1.746, 0.0]
            5. output = x_norm * gamma  (gamma empieza en [1,1,1,1])
        """
        # --- Paso 1-3: Calcular RMS ---
        # x.float(): convertir a float32 para precisión numérica.
        # Aunque entrenemos en float16/bfloat16, la normalización debe ser
        # en float32 para evitar overflow/underflow.
        # Es un truco estándar que usan TODOS los frameworks modernos.
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # rsqrt = 1/sqrt — es más rápido que calcular sqrt y luego dividir.
        # pow(2).mean() = mean(x^2)
        # keepdim=True mantiene la dimensión para broadcasting.
        # Resultado: rms tiene forma (..., 1) para poder multiplicar con x.

        # --- Paso 4-5: Normalizar y escalar ---
        # Multiplicamos por x (no x_float) para mantener el dtype original.
        # El .type_as(x) asegura que gamma esté en el mismo dtype que x.
        return (x_float * rms).type_as(x) * self.weight
