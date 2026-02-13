"""
Attention — El corazon del Transformer.

Este archivo implementa Multi-Head Attention con TODOS los tricks "grado militar":

1. RoPE (Rotary Position Embeddings) — posicion relativa por rotacion de vectores
2. GQA (Grouped Query Attention) — compartir K,V entre heads para eficiencia
3. QK-Norm — normalizar Q,K antes de attention para estabilidad
4. Flash Attention — implementacion fusionada via PyTorch scaled_dot_product_attention
5. Logit Soft-Capping — limitar logits de atencion con tanh (Gemma 2)
6. Differential Attention — restar dos patrones de atencion para cancelar ruido

¿Que hace la atencion?
    Permite que cada token "mire" a todos los tokens anteriores y decida
    cuales son relevantes. Es el mecanismo que da al Transformer su poder.

    Ejemplo: "Si llueve me mojo"
    Cuando el modelo procesa "mojo", la atencion le dice:
    - "llueve" es MUY relevante (causa)
    - "Si" es relevante (conector)
    - "me" es poco relevante (gramatical)

¿Como funciona (simplificado)?
    1. Cada token genera 3 vectores: Query (Q), Key (K), Value (V)
       - Q = "¿que estoy buscando?"
       - K = "¿que tengo para ofrecer?"
       - V = "¿que informacion tengo?"

    2. Calcular scores: Q * K^T / sqrt(d)
       Scores altos = tokens relevantes entre si

    3. Softmax: convertir scores a probabilidades (suman 1)

    4. Multiplicar por V: extraer informacion de tokens relevantes

    5. Resultado: cada token ahora contiene info de sus tokens relevantes

Causal Masking:
    En un Decoder-Only, cada token solo puede "ver" tokens ANTERIORES.
    Token 5 puede ver tokens 1-4 pero NO tokens 6-10.
    Esto se logra poniendo -inf en las posiciones futuras antes del softmax.
    El softmax convierte -inf en 0 (probabilidad cero de atender al futuro).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import NanoLogicConfig
from src.model.rmsnorm import RMSNorm

# =====================================================================
# RoPE — Rotary Position Embeddings
# =====================================================================
# RoPE codifica la posicion de cada token rotando sus vectores Q y K.
#
# Intuicion: imagina que cada par de dimensiones del embedding forma
# un punto en un plano 2D. RoPE ROTA ese punto un angulo proporcional
# a la posicion del token. Token 0 no rota, token 1 rota theta,
# token 2 rota 2*theta, etc.
#
# ¿Por que rotaciones? Porque el producto punto Q*K (que mide similitud)
# entre dos tokens rotados depende SOLO de la diferencia de posiciones,
# no de las posiciones absolutas. Esto es "posicion relativa".
#
# Frecuencias: usamos multiples frecuencias (como las manecillas de un reloj).
# Las dimensiones bajas rotan lento (capturan relaciones lejanas).
# Las dimensiones altas rotan rapido (capturan relaciones cercanas).
#
# Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)


def precompute_rope_frequencies(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Pre-calcula las frecuencias de rotacion para RoPE.

    Estas frecuencias se calculan UNA VEZ y se reusan en cada forward pass.
    No son parametros entrenables — son constantes matematicas.

    Args:
        dim: dimension por cabeza (head_dim). Debe ser par.
        max_seq_len: largo maximo de secuencia.
        theta: base para las frecuencias. 10000 es el valor del paper original.
               Valores mas grandes = frecuencias mas bajas = mejor para secuencias largas.

    Returns:
        Tensor de forma (max_seq_len, dim//2, 2) con [cos, sin] para cada posicion.
    """
    # Frecuencias: theta^(-2i/d) para i = 0, 1, ..., dim//2 - 1
    # Las frecuencias decrecen exponencialmente:
    #   i=0: theta^0 = 1 (rotacion rapida)
    #   i=dim//2-1: theta^(-1) = 0.0001 (rotacion muy lenta)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # Posiciones: 0, 1, 2, ..., max_seq_len-1
    positions = torch.arange(max_seq_len).float()

    # Angulos: posicion * frecuencia (outer product)
    # Forma: (max_seq_len, dim//2)
    angles = torch.outer(positions, freqs)

    # Pre-calcular cos y sin (mas eficiente que calcularlos cada vez)
    # Forma: (max_seq_len, dim//2, 2) donde [..., 0]=cos, [..., 1]=sin
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


def apply_rope(x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
    """Aplica RoPE a un tensor de queries o keys.

    La rotacion se aplica a pares de dimensiones:
    Para cada par (x0, x1):
        x0_new = x0 * cos(theta) - x1 * sin(theta)
        x1_new = x0 * sin(theta) + x1 * cos(theta)

    Es literalmente una rotacion 2D aplicada a cada par de dimensiones.

    Args:
        x: Tensor de forma (batch, n_heads, seq_len, head_dim)
        rope_freqs: Frecuencias pre-calculadas (seq_len, head_dim//2, 2)

    Returns:
        Tensor rotado de la misma forma.
    """
    # Reshape x para trabajar con pares: (..., head_dim) -> (..., head_dim//2, 2)
    x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)

    # Extraer cos y sin para las posiciones relevantes
    # rope_freqs tiene forma (max_seq_len, head_dim//2, 2)
    # Necesitamos solo las primeras seq_len posiciones
    seq_len = x.shape[2]
    cos_vals = rope_freqs[:seq_len, :, 0]  # (seq_len, head_dim//2)
    sin_vals = rope_freqs[:seq_len, :, 1]  # (seq_len, head_dim//2)

    # Expandir para broadcasting: (1, 1, seq_len, head_dim//2)
    cos_vals = cos_vals.unsqueeze(0).unsqueeze(0)
    sin_vals = sin_vals.unsqueeze(0).unsqueeze(0)

    # Aplicar rotacion 2D a cada par
    x0 = x_pairs[..., 0]  # primero de cada par
    x1 = x_pairs[..., 1]  # segundo de cada par

    x_out_0 = x0 * cos_vals - x1 * sin_vals
    x_out_1 = x0 * sin_vals + x1 * cos_vals

    # Recombinar pares: (..., head_dim//2, 2) -> (..., head_dim)
    x_out = torch.stack([x_out_0, x_out_1], dim=-1)
    return x_out.reshape(*x.shape).type_as(x)


# =====================================================================
# MODULO DE ATENCION
# =====================================================================


class CausalAttention(nn.Module):
    """Multi-Head Causal Attention con todos los tricks.

    Flujo interno:
        input (batch, seq, d_model)
            |
            v
        Proyecciones lineales -> Q, K, V
            |
            v
        [Opcional] QK-Norm: normalizar Q y K
            |
            v
        RoPE: rotar Q y K segun posicion
            |
            v
        GQA: expandir K,V para que coincidan con Q
            |
            v
        [Si Differential] Partir Q,K en dos mitades -> dos scores
            |
            v
        Attention scores: Q * K^T / sqrt(d)
            |
            v
        [Opcional] Soft-capping: tanh(scores/cap) * cap
            |
            v
        Causal mask: -inf para posiciones futuras
            |
            v
        Softmax -> probabilidades
            |
            v
        [Si Differential] Restar: softmax1 - lambda * softmax2
            |
            v
        Multiplicar por V -> output
            |
            v
        Proyeccion de salida -> (batch, seq, d_model)
    """

    def __init__(self, config: NanoLogicConfig) -> None:
        super().__init__()
        self.config = config

        # Guardar valores clave
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_kv_groups = config.n_kv_groups

        # ---------------------------------------------------------------
        # Proyecciones lineales: convierten input -> Q, K, V
        # ---------------------------------------------------------------
        # Q tiene n_heads cabezas (cada una busca algo diferente)
        # K y V tienen n_kv_heads cabezas (compartidas entre grupos de Q)
        #
        # bias=False: los modelos modernos NO usan bias en attention.
        # LLaMA, Mistral, Gemma — ninguno usa bias aqui.
        # Ahorra parametros sin afectar calidad.

        if config.differential_attention:
            # Differential Attention: Q y K se duplican (Q1,Q2 y K1,K2)
            # Cada "mitad" calcula un patron de atencion diferente
            # Para Q: n_heads cabezas, pero cada una tiene 2*head_dim
            # Para K: n_kv_heads cabezas, cada una con 2*head_dim
            self.q_proj = nn.Linear(
                config.d_model, config.n_heads * config.head_dim * 2, bias=False
            )
            self.k_proj = nn.Linear(
                config.d_model, config.n_kv_heads * config.head_dim * 2, bias=False
            )
        else:
            self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
            self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)

        # V siempre es igual (no se duplica en Differential)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)

        # Proyeccion de salida: combina todas las cabezas -> d_model
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

        # ---------------------------------------------------------------
        # QK-Norm (opcional)
        # ---------------------------------------------------------------
        if config.qk_norm:
            self.q_norm = RMSNorm(config.head_dim)
            self.k_norm = RMSNorm(config.head_dim)

        # ---------------------------------------------------------------
        # Differential Attention: lambda parametro aprendible
        # ---------------------------------------------------------------
        if config.differential_attention:
            # lambda controla cuanto peso darle al segundo patron de atencion
            # que se resta del primero. Empieza en 0.8 (valor del paper).
            # El modelo aprende el valor optimo durante el entrenamiento.
            self.diff_lambda = nn.Parameter(torch.tensor(0.8))

        # Dropout para regularizacion
        self.attn_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass del modulo de atencion.

        Args:
            x: Input de forma (batch, seq_len, d_model)
            rope_freqs: Frecuencias RoPE pre-calculadas
            mask: Mascara causal (opcional, se puede generar internamente)

        Returns:
            Output de forma (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # ============================================================
        # PASO 1: Proyecciones Q, K, V
        # ============================================================
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.config.differential_attention:
            # Reshape para Differential: cada head tiene 2*head_dim
            # (batch, seq, n_heads * head_dim * 2) -> (batch, n_heads, seq, head_dim * 2)
            q = q.view(batch, seq_len, self.n_heads, self.head_dim * 2).transpose(1, 2)
            k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim * 2).transpose(1, 2)

            # Partir en dos mitades: Q1, Q2 y K1, K2
            q1, q2 = q.chunk(2, dim=-1)  # cada una: (batch, n_heads, seq, head_dim)
            k1, k2 = k.chunk(2, dim=-1)  # cada una: (batch, n_kv_heads, seq, head_dim)
        else:
            # Reshape normal: (batch, seq, n_heads * head_dim) -> (batch, n_heads, seq, head_dim)
            q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # V siempre tiene el mismo reshape
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # ============================================================
        # PASO 2: QK-Norm (opcional)
        # ============================================================
        if self.config.qk_norm:
            if self.config.differential_attention:
                q1 = self.q_norm(q1)
                q2 = self.q_norm(q2)
                k1 = self.k_norm(k1)
                k2 = self.k_norm(k2)
            else:
                q = self.q_norm(q)
                k = self.k_norm(k)

        # ============================================================
        # PASO 3: RoPE — rotar Q y K segun posicion
        # ============================================================
        if self.config.differential_attention:
            q1 = apply_rope(q1, rope_freqs)
            q2 = apply_rope(q2, rope_freqs)
            k1 = apply_rope(k1, rope_freqs)
            k2 = apply_rope(k2, rope_freqs)
        else:
            q = apply_rope(q, rope_freqs)
            k = apply_rope(k, rope_freqs)

        # ============================================================
        # PASO 4: GQA — expandir K,V para que coincidan con Q
        # ============================================================
        # Si n_kv_heads < n_heads, necesitamos repetir K,V.
        # Con n_heads=8, n_kv_heads=2: repetimos cada K,V 4 veces.
        if self.n_kv_groups > 1:
            if self.config.differential_attention:
                k1 = self._repeat_kv(k1)
                k2 = self._repeat_kv(k2)
            else:
                k = self._repeat_kv(k)
            v = self._repeat_kv(v)

        # ============================================================
        # PASO 5: Calcular attention
        # ============================================================
        if self.config.differential_attention:
            output = self._differential_attention(q1, q2, k1, k2, v, mask)
        else:
            output = self._standard_attention(q, k, v, mask)

        # ============================================================
        # PASO 6: Combinar cabezas y proyectar salida
        # ============================================================
        # (batch, n_heads, seq, head_dim) -> (batch, seq, n_heads * head_dim)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        # Proyeccion final: d_model -> d_model
        return self.o_proj(output)

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Atencion estandar con Flash Attention y soft-capping."""

        if self.config.attention_cap > 0:
            # Con soft-capping no podemos usar Flash Attention directo
            # porque necesitamos acceso a los logits intermedios.
            # Hacemos la atencion manualmente.
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Soft-capping: limitar scores con tanh
            cap = self.config.attention_cap
            scores = cap * torch.tanh(scores / cap)

            # Causal mask
            if mask is None:
                mask = self._make_causal_mask(q.shape[2], q.device, q.dtype)
            scores = scores + mask

            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
            attn_weights = self.attn_dropout(attn_weights)
            return torch.matmul(attn_weights, v)
        else:
            # Sin soft-capping: usar Flash Attention (mucho mas rapido)
            # PyTorch 2.0+ lo activa automaticamente con esta funcion.
            # Flash Attention fusiona Q*K, softmax, y *V en un solo kernel CUDA.
            # Resultado: 2-4x mas rapido, 5-20x menos memoria.
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=mask is None,  # genera causal mask internamente
            )

    def _differential_attention(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        k1: torch.Tensor,
        k2: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Differential Attention: resta dos patrones para cancelar ruido.

        attn = softmax(Q1*K1/sqrt(d)) - lambda * softmax(Q2*K2/sqrt(d))

        Los tokens irrelevantes generan scores similares en ambos patrones.
        Al restarlos, el ruido se cancela y solo queda la senal real.
        """
        scale = 1.0 / math.sqrt(self.head_dim)

        # Calcular scores para ambos patrones
        scores1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        scores2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale

        # Soft-capping (opcional)
        if self.config.attention_cap > 0:
            cap = self.config.attention_cap
            scores1 = cap * torch.tanh(scores1 / cap)
            scores2 = cap * torch.tanh(scores2 / cap)

        # Causal mask
        if mask is None:
            mask = self._make_causal_mask(q1.shape[2], q1.device, q1.dtype)
        scores1 = scores1 + mask
        scores2 = scores2 + mask

        # Softmax por separado (en float32 para estabilidad)
        attn1 = F.softmax(scores1, dim=-1, dtype=torch.float32).type_as(q1)
        attn2 = F.softmax(scores2, dim=-1, dtype=torch.float32).type_as(q2)

        # Differential: restar patron 2 de patron 1
        # lambda es aprendible — el modelo decide cuanto ruido cancelar
        diff_attn = attn1 - self.diff_lambda * attn2

        # Dropout sobre la atencion diferencial
        diff_attn = self.attn_dropout(diff_attn)

        # Multiplicar por V
        return torch.matmul(diff_attn, v)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repite K o V para GQA.

        Si tenemos 2 KV heads y 8 Q heads, cada KV head se repite 4 veces
        para alinearse con las Q heads correspondientes.

        (batch, n_kv_heads, seq, head_dim) -> (batch, n_heads, seq, head_dim)
        """
        if self.n_kv_groups == 1:
            return x

        batch, _, seq_len, head_dim = x.shape
        # Expandir: insertar dimension para repeticion
        x = x.unsqueeze(2)  # (batch, n_kv_heads, 1, seq, head_dim)
        # Repetir en la nueva dimension
        x = x.expand(-1, -1, self.n_kv_groups, -1, -1)
        # Reshape: fusionar n_kv_heads * n_kv_groups = n_heads
        return x.reshape(batch, self.n_heads, seq_len, head_dim)

    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Genera la mascara causal.

        Crea una matriz triangular superior llena de -inf:
        [[ 0,   -inf, -inf, -inf],
         [ 0,    0,   -inf, -inf],
         [ 0,    0,    0,   -inf],
         [ 0,    0,    0,    0  ]]

        Cuando se suma a los scores ANTES del softmax:
        - Posiciones permitidas (0) no cambian
        - Posiciones futuras (-inf) se convierten en 0 despues del softmax

        Resultado: cada token solo atiende a si mismo y a tokens anteriores.
        """
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask
