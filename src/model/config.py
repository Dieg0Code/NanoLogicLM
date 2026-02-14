"""
NanoLogicConfig — Todos los hiperparámetros del modelo en un solo lugar.

¿Qué es un config?
    Es un dataclass que centraliza TODOS los hiperparámetros del modelo.
    En vez de tener números mágicos desperdigados por el código, todo
    está aquí. Si quieres cambiar algo, lo cambias en UN SOLO lugar.

¿Por qué frozen=True?
    Para que nadie modifique los hiperparámetros después de crear el config.
    Si quieres un config diferente, crea una nueva instancia.
    Esto previene bugs sutiles donde alguien cambia d_model a mitad
    del entrenamiento y todo explota.

¿Cómo se usa?
    config = NanoLogicConfig()                    # usar defaults
    config = NanoLogicConfig(n_layers=12)         # override un valor
    config = NanoLogicConfig.from_json("cfg.json") # cargar de archivo

    model = NanoLogicTransformer(config)          # pasar al modelo
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class NanoLogicConfig:
    """Configuración completa del modelo NanoLogic.

    Cada campo tiene un valor default que corresponde a nuestra config
    "grado militar" — optimizada para máximo rendimiento con ~20M params.
    """

    # =================================================================
    # DIMENSIONES PRINCIPALES
    # =================================================================
    # Estos 4 valores definen el "tamaño" del modelo.
    # Cambiar cualquiera de estos cambia el número de parámetros.

    # d_model: dimensión de los embeddings y de todas las capas internas.
    # Es el "ancho" del modelo — cada token se representa como un vector
    # de d_model dimensiones. Más grande = más capacidad, pero más lento.
    #
    # Referencia: GPT-2 small=768, LLaMA 7B=4096, nuestro=512
    d_model: int = 512

    # n_layers: cantidad de Decoder Blocks apilados.
    # Es la "profundidad" del modelo — más capas = mejor razonamiento
    # porque la información pasa por más transformaciones no-lineales.
    # Cada capa refina la representación del texto.
    #
    # 8 capas es un buen balance para ~20M params.
    # Con 6 el razonamiento era más limitado (probamos conceptualmente).
    # Con 12+ necesitaríamos más datos para no hacer overfit.
    n_layers: int = 8

    # n_heads: cantidad de cabezas de atención en paralelo.
    # Cada cabeza "mira" una relación diferente entre tokens.
    # Una cabeza puede enfocarse en sintaxis, otra en semántica, etc.
    #
    # d_model debe ser divisible por n_heads.
    # 512 / 8 = 64 dimensiones por cabeza (estándar en la industria).
    n_heads: int = 8

    # n_kv_heads: cantidad de cabezas para Key y Value en GQA.
    # ¿Qué es GQA? Grouped Query Attention.
    #
    # En Multi-Head Attention normal, cada cabeza tiene su propio Q, K, V:
    #   8 heads × (Q, K, V) = 24 matrices de proyección  # noqa: RUF003
    #
    # En GQA, varias cabezas de Q comparten las mismas K y V:
    #   8 Q heads + 2 K heads + 2 V heads = 12 matrices
    #
    # Beneficios:
    # - 50% menos parámetros en K,V → modelo más eficiente
    # - Inferencia más rápida (menos memoria para KV cache)
    # - Calidad similar o igual (paper de Mistral lo demuestra)
    #
    # n_heads debe ser divisible por n_kv_heads.
    # 8 Q heads / 2 KV heads = cada 4 Q heads comparten 1 K,V.
    n_kv_heads: int = 2

    # d_ff: dimensión interna del Feed-Forward Network (SwiGLU).
    # El FFN expande el vector de d_model a d_ff y lo contrae de vuelta.
    # Es donde el modelo "piensa" — transforma las representaciones.
    #
    # Con SwiGLU, la regla es d_ff = (8/3) * d_model:
    #   (8/3) × 512 = 1365.33... → redondeamos a 1365  # noqa: RUF003
    #
    # ¿Por qué (8/3)? Porque SwiGLU usa 3 matrices (W1, W_gate, W2)
    # en vez de 2, así que se compensa reduciendo d_ff para mantener
    # el mismo número de parámetros que un FFN con 4×d_model.  # noqa: RUF003
    d_ff: int = 1365

    # =================================================================
    # VOCABULARIO Y SECUENCIA
    # =================================================================

    # vocab_size: tamaño del vocabulario del tokenizer BPE.
    # Debe coincidir con el tokenizer entrenado.
    # Define el tamaño de la embedding layer: nn.Embedding(vocab_size, d_model)
    vocab_size: int = 8000

    # max_seq_len: largo máximo de secuencia que el modelo puede procesar.
    # Secuencias más largas que esto se truncan.
    # Debe ser suficiente para cubrir nuestros ejemplos más largos.
    # El P99 de nuestros datos es mucho menor que 1024, así que sobra.
    max_seq_len: int = 1024

    # pad_token_id: ID del token de padding en el vocabulario.
    # Se usa para: (1) padding_idx en nn.Embedding (no genera gradientes),
    # (2) ignorar posiciones de padding en la loss.
    # Debe coincidir con el ID real asignado por el tokenizer.
    # Default: 0 (primer token en SpecialTokens.as_list()).
    pad_token_id: int = 0

    # eos_token_id: ID del token de fin de secuencia.
    # Se usa para saber cuando PARAR la generacion en inferencia.
    # Default: 2 (tercer token en SpecialTokens.as_list()).
    eos_token_id: int = 2

    # =================================================================
    # REGULARIZACIÓN
    # =================================================================

    # dropout: probabilidad de "apagar" neuronas al azar durante entrenamiento.
    # Evita el overfitting (que el modelo memorice en vez de aprender).
    # 0.1 = apaga el 10% de las neuronas aleatoriamente.
    # En inferencia (producción), el dropout se desactiva automáticamente.
    dropout: float = 0.1

    # =================================================================
    # TRICKS "GRADO MILITAR"
    # =================================================================
    # Cada uno de estos cuesta casi 0 en computación pero mejora el modelo.

    # rope_theta: base para la frecuencia de Rotary Embeddings.
    # Controla cómo rota el espacio vectorial para codificar posiciones.
    # 10000.0 es el estándar (RoFormer, GPT-NeoX, PaLM).
    # Para contextos muy largos (>32k), se usa 500000.0 (CodeLlama).
    rope_theta: float = 10000.0

    # weight_tying: compartir pesos entre la embedding layer y el linear head.
    # La embedding convierte IDs → vectores.
    # El head convierte vectores → probabilidades sobre IDs.
    # Son operaciones inversas — tiene sentido que usen los mismos pesos.
    # Ahorra ~4M parámetros y a menudo MEJORA el rendimiento.
    # Lo usan: GPT-2, LLaMA, T5.
    weight_tying: bool = True

    # pre_norm: normalizar ANTES de attention/FFN (estilo LLaMA).
    # Post-Norm (viejo):  x → Attention → Add → Norm
    # Pre-Norm (moderno): x → Norm → Attention → Add
    #
    # Pre-Norm entrena más estable, especialmente en modelos chicos.
    # TODOS los modelos modernos (LLaMA, Mistral, Gemma) usan Pre-Norm.
    pre_norm: bool = True

    # qk_norm: normalizar Query y Key antes de calcular attention scores.
    # Sin esto, los valores de Q·K pueden crecer mucho con secuencias largas,
    # causando gradientes inestables o attention scores que saturan en softmax.
    # Gemma lo usa. Es literalmente 2 líneas de código.
    qk_norm: bool = True

    # embed_scale: escalar los embeddings por √d_model.
    # Los embeddings se inicializan con valores pequeños (~0.02).
    # Multiplicar por √512 ≈ 22.6 los lleva a una escala compatible
    # con las capas de attention. Trick del paper original "Attention Is All You Need".
    embed_scale: bool = True

    # attention_cap: valor máximo para los logits de atención (soft-capping).
    # Previene que los attention scores exploten a infinito.
    # Funciona así: logits = cap * tanh(logits / cap)
    # Con cap=50, los valores se limitan suavemente al rango [-50, 50].
    # Lo usa Gemma 2. Si es 0.0, se desactiva.
    attention_cap: float = 50.0

    # deep_norm: escalar las conexiones residuales para estabilidad profunda.
    # Trick de Microsoft Research ("DeepNet: Scaling Transformers to 1,000 Layers").
    #
    # Sin Deep Norm:   output = x + sublayer(norm(x))
    # Con Deep Norm:   output = x * alpha + sublayer(norm(x))
    #
    # alpha = (2 * n_layers) ** 0.25
    # Con 8 capas: alpha = (2*8)^0.25 = 16^0.25 = 2.0
    #
    # ¿Por qué funciona? Las capas profundas tienden a "olvidar" la señal
    # original porque pasa por demasiadas transformaciones. Multiplicar
    # el residual por alpha > 1 asegura que la señal original llegue fuerte
    # hasta las últimas capas.
    #
    # El paper también recomienda escalar la inicialización de los pesos
    # por β = (8 * n_layers) ** -0.25, pero eso lo hacemos en transformer.py.
    deep_norm: bool = True

    # differential_attention: usar Differential Attention (Microsoft, 2024).
    # Cada head calcula DOS patrones de atencion y los resta:
    #   attn = softmax(Q1*K1) - lambda * softmax(Q2*K2)
    #
    # ¿Por que funciona? Los attention scores normales tienen mucho "ruido":
    # tokens irrelevantes reciben atencion no-cero por el softmax.
    # Al restar dos patrones, el ruido (que es similar en ambos) se cancela
    # y solo queda la senal real: los tokens que realmente importan.
    #
    # Costo: duplica Q y K por head (pero con GQA, K ya esta compartido).
    # Beneficio: paper muestra mejoras consistentes en benchmarks.
    # Paper: "Differential Transformer" (Microsoft Research, 2024)
    differential_attention: bool = True

    # gate_residual: agregar un bypass dentro de la FFN SwiGLU.
    # Normalmente SwiGLU hace: output = SiLU(gate) * up
    # Con gate residual:       output = SiLU(gate) * up + alpha * up
    #
    # alpha es un parametro aprendible que empieza en 0 (sin efecto).
    # El modelo puede aprender a "dejar pasar" informacion sin filtrar
    # si la compuerta SiLU se equivoca. Es un "safety net" dentro de la FFN.
    # Truco underground — bajo costo (1 parametro extra), bajo riesgo.
    gate_residual: bool = True

    # stochastic_depth_rate: probabilidad MAXIMA de saltarse un bloque.
    # Durante entrenamiento, cada bloque se salta con probabilidad:
    #   p = rate * (layer_idx / (n_layers - 1))
    #
    # Las capas TEMPRANAS se saltean poco (son criticas).
    # Las capas TARDIAS se saltean mas (contienen info mas redundante).
    #
    # Con rate=0.1 y 8 capas:
    #   Capa 0: p=0.000 (nunca se salta)
    #   Capa 3: p=0.043
    #   Capa 7: p=0.100 (10% chance de skip)
    #
    # En INFERENCIA: todos los bloques se ejecutan (pero escalados).
    # Es como dropout pero a nivel de bloque completo.
    # Paper: "Deep Networks with Stochastic Depth" (Huang et al., 2016)
    stochastic_depth_rate: float = 0.1

    # output_cap: soft-capping para los logits FINALES del LM head.
    # Igual que attention_cap pero para la salida del modelo.
    # Previene overconfidence: el modelo no puede estar 100% seguro.
    # logits = cap * tanh(logits / cap)
    # Con cap=30, logits se limitan a [-30, 30].
    # Lo usa Gemma 2. Si es 0.0, se desactiva.
    output_cap: float = 30.0

    # head_scale: factor de escala aprendible para los logits de salida.
    # logits = lm_head(x) * head_scale
    # El modelo puede aprender a "calibrar" su confianza:
    # - head_scale < 1 → distribuciones mas suaves (menos seguro)
    # - head_scale > 1 → distribuciones mas picudas (mas seguro)
    # Empieza en 1.0 (sin efecto). Un solo parametro, zero riesgo.
    head_scale: bool = True

    # z_loss_weight: peso del Z-Loss (regularizacion de PaLM).
    # Penaliza logits grandes: z_loss = weight * mean(logsumexp(logits)^2)
    # Esto enseña al modelo a mantener logits en rango razonable.
    # Complementario a output_cap: cap limita "a la fuerza",
    # z_loss enseña a no necesitar limitacion.
    # Se aplica en el training loop, no en el modelo.
    # Si es 0.0, se desactiva.
    z_loss_weight: float = 1e-4

    # =================================================================
    # PROPIEDADES DERIVADAS
    # =================================================================
    # Estos valores se calculan automáticamente a partir de los anteriores.

    @property
    def head_dim(self) -> int:
        """Dimensión de cada cabeza de atención.

        d_model / n_heads = dimensiones por cabeza.
        Cada cabeza trabaja con un subespacio de head_dim dimensiones.
        """
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) debe ser divisible por n_heads ({self.n_heads})"
        )
        return self.d_model // self.n_heads

    @property
    def kv_head_dim(self) -> int:
        """Dimensión de cada cabeza K,V (en GQA, puede ser diferente)."""
        return self.head_dim

    @property
    def n_kv_groups(self) -> int:
        """Cuántas Q heads comparten cada K,V head.

        Con n_heads=8 y n_kv_heads=2: cada 4 Q heads comparten 1 K,V.
        """
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) debe ser divisible por n_kv_heads ({self.n_kv_heads})"
        )
        return self.n_heads // self.n_kv_heads

    @property
    def embed_scale_factor(self) -> float:
        """Factor de escalamiento para embeddings: sqrt(d_model)."""
        return math.sqrt(self.d_model) if self.embed_scale else 1.0

    @property
    def deep_norm_alpha(self) -> float:
        """Factor de escalamiento residual para Deep Norm.

        alpha = (2 * n_layers) ** 0.25
        Con 8 capas: (2*8)^0.25 = 16^0.25 = 2.0
        Retorna 1.0 si deep_norm está desactivado (sin escalamiento).
        """
        if not self.deep_norm:
            return 1.0
        return (2.0 * self.n_layers) ** 0.25

    @property
    def deep_norm_beta(self) -> float:
        """Factor de escalamiento para inicialización de pesos con Deep Norm.

        beta = (8 * n_layers) ** -0.25
        Con 8 capas: (8*8)^-0.25 = 64^-0.25 = 0.354
        Se usa para escalar la inicialización Xavier/He de sublayers.
        Retorna 1.0 si deep_norm está desactivado.
        """
        if not self.deep_norm:
            return 1.0
        return (8.0 * self.n_layers) ** -0.25

    # =================================================================
    # GUARDAR Y CARGAR
    # =================================================================

    def to_json(self, path: str) -> None:
        """Guarda el config como JSON para reproducibilidad.

        Siempre guarda el config junto al modelo entrenado.
        Así, cuando cargues el modelo después, sabes exactamente
        qué hiperparámetros se usaron.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> NanoLogicConfig:
        """Carga un config desde un archivo JSON.

        Uso:
            config = NanoLogicConfig.from_json("models/config.json")
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def __post_init__(self) -> None:
        """Validaciones que se ejecutan al crear el config.

        Si alguien pasa valores inválidos (ej: d_model=513 con n_heads=8),
        esto lanza un error INMEDIATO con un mensaje claro, en vez de
        fallar misteriosamente 3 capas más adelante.
        """
        # --- d_model debe ser divisible por n_heads ---
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) debe ser divisible por n_heads ({self.n_heads}). "
                f"Prueba d_model={self.n_heads * (self.d_model // self.n_heads)}"
            )

        # --- n_heads debe ser divisible por n_kv_heads (para GQA) ---
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) debe ser divisible por n_kv_heads ({self.n_kv_heads}). "
                f"Valores válidos de n_kv_heads: {[i for i in range(1, self.n_heads + 1) if self.n_heads % i == 0]}"
            )

        # --- vocab_size debe ser positivo ---
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size debe ser positivo, recibí {self.vocab_size}")

        # --- max_seq_len debe ser positivo ---
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len debe ser positivo, recibí {self.max_seq_len}")

    def count_params_estimate(self) -> dict[str, int]:
        """Estima el número de parámetros por componente.

        Útil para entender dónde están los parámetros del modelo
        y verificar que el total esté en el rango esperado.

        Returns:
            Dict con el conteo por componente y el total.
        """
        # Embedding: vocab_size × d_model
        embedding = self.vocab_size * self.d_model

        # Attention por capa:
        #   Q: d_model × (n_heads × head_dim) = d_model × d_model
        #   K: d_model × (n_kv_heads × head_dim)
        #   V: d_model × (n_kv_heads × head_dim)
        #   Output: d_model × d_model
        q_params = self.d_model * self.d_model
        k_params = self.d_model * (self.n_kv_heads * self.head_dim)
        v_params = self.d_model * (self.n_kv_heads * self.head_dim)
        o_params = self.d_model * self.d_model
        attention_per_layer = q_params + k_params + v_params + o_params

        # SwiGLU FFN por capa: 3 matrices de d_model × d_ff
        ffn_per_layer = 3 * self.d_model * self.d_ff

        # RMSNorm por capa: 2 × d_model (una para attention, otra para FFN)
        norm_per_layer = 2 * self.d_model

        # Total por capa
        per_layer = attention_per_layer + ffn_per_layer + norm_per_layer

        # Head: d_model × vocab_size (0 si weight_tying)
        head = 0 if self.weight_tying else self.vocab_size * self.d_model

        # RMSNorm final
        final_norm = self.d_model

        total = embedding + (per_layer * self.n_layers) + head + final_norm

        return {
            "embedding": embedding,
            "attention_per_layer": attention_per_layer,
            "ffn_per_layer": ffn_per_layer,
            "norm_per_layer": norm_per_layer,
            "total_per_layer": per_layer,
            "all_layers": per_layer * self.n_layers,
            "head": head,
            "final_norm": final_norm,
            "total": total,
            "total_millions": round(total / 1_000_000, 1),
        }

    def __repr__(self) -> str:
        """Representación legible del config."""
        params = self.count_params_estimate()
        return (
            f"NanoLogicConfig(\n"
            f"  d_model={self.d_model}, n_layers={self.n_layers}, "
            f"n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads},\n"
            f"  d_ff={self.d_ff}, vocab_size={self.vocab_size}, "
            f"max_seq_len={self.max_seq_len},\n"
            f"  ~{params['total_millions']}M params\n"
            f")"
        )
