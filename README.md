# 🧠 Nano Language Model — Apuntes

Modelo que transforma lenguaje natural (español) en fórmulas de lógica proposicional.

**Ejemplo**:
```
Input:  "Si llueve y no llevo paraguas, entonces me mojo"
Output: (p ∧ ¬q) → r
```

---

## Pipeline completo (el camino del dato al modelo)

```
dataset.json (2,500 raw de DeepSeek)
    ↓  clean.py
dataset_clean.json (~2,100)
    ↓  verify.py ($0.50)
dataset_verified.json (~2,000)
    ↓  augment.py
dataset_augmented.json (~5,500+)
    ↓  preprocess.py
train.jsonl / val.jsonl / test.jsonl
    ↓  train tokenizer (BPE)
tokenizer.json
    ↓  train.py
model.pt 🎉
```

---

## Fundamentos de Deep Learning (TLDR)

### Tensores — la unidad básica

Un tensor es un contenedor de números con forma (shape). Todo en deep learning son tensores:

```
Escalar (0D):     5                       → shape: ()        → un solo número
Vector (1D):      [1, 2, 3]               → shape: (3,)      → una lista
Matriz (2D):      [[1,2], [3,4]]          → shape: (2, 2)    → una tabla
Tensor 3D:        [[[...], [...]], ...]    → shape: (8, 10, 512)  → un cubo
```

¿Por qué tensores y no listas de Python? Porque los tensores viven en la GPU
y las operaciones son **miles de veces más rápidas**. Una multiplicación de
matrices 512x512 en CPU tarda segundos; en GPU con tensores, microsegundos.

En PyTorch:
```python
import torch
x = torch.tensor([1.0, 2.0, 3.0])    # vector
x = torch.randn(8, 10, 512)           # tensor 3D con valores aleatorios
# 8 = batch_size (cuántos ejemplos procesar a la vez)
# 10 = seq_len (largo de la secuencia)
# 512 = d_model (dimensión de representación de cada token)
```

### Neuronas y capas

Una neurona hace: `inputs x pesos -> suma -> activacion -> output`

```
inputs  = [0.5, -1.0, 2.0]
pesos   = [0.3,  0.7, 0.1]   <-- estos se APRENDEN
suma    = 0.5*0.3 + (-1.0)*0.7 + 2.0*0.1 = -0.35
output  = activacion(-0.35)   <-- funcion no lineal (ReLU, SiLU, etc.)
```

Muchas neuronas juntas = una **capa (layer)**.
Muchas capas apiladas = una **red neuronal**.

En PyTorch, una capa lineal es:
```python
capa = nn.Linear(512, 256)   # 512 entradas, 256 salidas
# internamente: output = input @ weight.T + bias
# weight tiene shape (256, 512) = 131,072 parametros aprendibles
```

### Multiplicacion de matrices — LA operacion del deep learning

Todo en un Transformer se reduce a multiplicaciones de matrices:

```
(batch, seq, 512) @ (512, 256) = (batch, seq, 256)
     input            weight         output

Regla: el ultimo eje de A debe coincidir con el penultimo de B:
  (8, 10, 512) @ (512, 256) -> (8, 10, 256)  OK
  (8, 10, 512) @ (256, 512) -> ERROR          los ejes no coinciden
```

### El ciclo de aprendizaje (TODO el deep learning son estos 4 pasos)

```
1. FORWARD:    input -> modelo -> prediccion
2. LOSS:       comparar prediccion vs respuesta correcta -> numero de error
3. BACKWARD:   calcular gradientes (cuanto contribuyo cada peso al error?)
4. UPDATE:     ajustar pesos para reducir el error

Repetir miles de veces -> el modelo "aprende"
```

En codigo:
```python
for batch in dataloader:
    pred = model(batch.input)             # 1. forward pass
    loss = loss_fn(pred, batch.target)     # 2. calcular error
    loss.backward()                        # 3. calcular gradientes (automatico!)
    optimizer.step()                       # 4. actualizar pesos
    optimizer.zero_grad()                  # limpiar para la siguiente iteracion
```

### Gradientes — como "aprende" el modelo

El gradiente de un peso dice: "si muevo este peso un poquito, el error sube o baja?"

```
peso=0.5, gradiente=-0.1 -> "si subo peso, error BAJA" -> subir
peso=0.5, gradiente=+0.3 -> "si subo peso, error SUBE" -> bajar
```

PyTorch calcula TODOS los gradientes automaticamente con `loss.backward()`.
No necesitas hacer calculo a mano. Esto se llama **autograd** y es la magia
central de PyTorch.

### Loss function — como medir el error

Para modelos de lenguaje, se usa **Cross-Entropy Loss**:

```
prediccion del modelo:  [0.1, 0.05, 0.7, 0.15]   (probabilidades para cada token)
respuesta correcta:     [0,   0,    1,   0   ]    (el token correcto es el 3ro)

cross_entropy = -log(0.7) = 0.36    (bajo = bueno, la prediccion era buena)

si hubiera predicho:    [0.6, 0.2, 0.1, 0.1]
cross_entropy = -log(0.1) = 2.30    (alto = malo, fallo la prediccion)
```

El modelo minimiza este numero. Cuando loss baja, el modelo esta aprendiendo.

### nn.Parameter vs nn.Module — el sistema de PyTorch

```python
nn.Parameter:  un tensor que PyTorch sabe que debe entrenar
               (calcula gradientes y actualiza con el optimizador)
               Ejemplo: los pesos gamma de RMSNorm

nn.Module:     una "pieza" del modelo. Puede contener Parameters y otros Modules.
               Tiene un metodo forward() que define como transforma el input.
               Ejemplo: RMSNorm, Linear, nuestro Transformer completo

nn.Linear:     Module predefinido. Hace: output = input @ weight + bias
nn.Embedding:  Module predefinido. Es una tabla: ID -> vector de d_model dimensiones
nn.Dropout:    Module predefinido. Apaga neuronas al azar (regularizacion)
```

### Conexion residual — el truco que permite redes profundas

Sin residuales, con 8+ capas la señal se degrada (se pierde informacion):
```
input -> capa1 -> capa2 -> ... -> capa8 -> output  (la señal original se perdio)
```

Con residuales, sumamos el input original al output de cada capa:
```
input -> capa1 -> (+input) -> capa2 -> (+) -> ... -> output  (señal preservada)
```

Esto crea un "atajo" para que la informacion fluya directo de las primeras capas
a las ultimas. Sin esto, seria imposible entrenar Transformers profundos.

### La intuicion completa para NanoLogic

```
"Si llueve me mojo"
     | tokenizer
[45, 892, 12, 567]              <-- IDs numericos
     | embedding (tabla de lookup)
[[0.1, -0.3, ...],             <-- cada ID se convierte en un vector de 512 numeros
 [0.5,  0.2, ...],                 que "representan" el significado de la palabra
 [0.8, -0.1, ...],
 [0.2,  0.6, ...]]
     | 8 capas de transformer (attention + FFN)
[[...],                         <-- los vectores se transforman capa a capa
 [...],                             cada capa "entiende" mas contexto
 [...],                             la capa 1 ve palabras individuales
 [...]]                             la capa 8 entiende relaciones logicas
     | linear head
[[0.01, 0.02, ..., 0.95],      <-- probabilidades sobre los 8000 tokens
 [...],                             del vocabulario
 [...],
 [...]]                             "¿cual es el siguiente token mas probable?"
     | argmax (tomar el mas probable)
"p" -> "→" -> "q"              <-- la formula generada token a token
```

---

## Paso 1: Limpieza (`data/scripts/clean.py`)

**¿Qué hace?** Filtra los ejemplos basura que generó DeepSeek.

**¿Cómo?** Usa un **Recursive Descent Parser** — un parser que tiene una función
por cada regla de la gramática de lógica proposicional:

```
fórmula     → bicondicional
bicondicional → implicación (↔ implicación)*
implicación → disyunción (→ disyunción)*
disyunción  → conjunción (∨ conjunción)*
conjunción  → unario (∧ unario)*
unario      → ¬ unario | primario
primario    → átomo | ( fórmula )
```

Cada nivel = una precedencia de operador. Los de arriba se evalúan último (↔),
los de abajo primero (¬). Como en matemáticas: × antes que +.

**Sub-pasos:**
1. Validar sintaxis con el parser (si no parsea → fuera)
2. Verificar paréntesis balanceados
3. Verificar que los átomos declarados coincidan con los de la fórmula
4. Eliminar duplicados exactos (mismo input + misma fórmula)
5. Subsampling de fórmulas triviales (demasiados `p → q` simples)
6. Normalizar formato (espacios, paréntesis consistentes)

**Resultado**: ~2,100 ejemplos limpios (de 2,500 originales).

---

## Paso 2: Verificación con API (`data/scripts/verify.py`)

**¿Qué hace?** Le pregunta a DeepSeek: "¿esta fórmula es correcta para este enunciado?"

**¿Por qué?** Porque DeepSeek generó los datos Y los puede verificar (como peer review).
Un ejemplo puede pasar la validación sintáctica pero ser semánticamente incorrecto:

```
Input:   "Si llueve O nieva, llevo abrigo"
Fórmula: p ∧ q → r    ← SINTÁCTICAMENTE VÁLIDA, pero debería ser ∨, no ∧
```

El parser no detecta eso. La API sí.

**Costo**: ~$0.50 para 2,000 ejemplos (temperatura baja, respuestas cortas).

**Resultado**: ~2,000 ejemplos verificados.

---

## Paso 3: Data Augmentation (`data/scripts/augment.py`)

**¿Qué hace?** Crea ejemplos nuevos a partir de los existentes usando equivalencias lógicas.

**Técnicas (Python puro, $0):**

| Técnica | Ejemplo |
|---------|---------|
| Equivalencia de implicación | `p → q` se convierte en `¬p ∨ q` |
| De Morgan (AND) | `¬(p ∧ q)` se convierte en `¬p ∨ ¬q` |
| De Morgan (OR) | `¬(p ∨ q)` se convierte en `¬p ∧ ¬q` |
| Doble negación | `¬¬p` se convierte en `p` (y viceversa) |
| Conmutatividad | `p ∧ q` se convierte en `q ∧ p` |
| Composición | Combina 2 ejemplos simples en 1 avanzado |

**Importante**: la frase en lenguaje natural queda igual, pero la fórmula cambia
a su equivalente. El modelo así aprende que hay múltiples representaciones correctas.

**Resultado**: ~5,500+ ejemplos.

---

## Paso 4: Preprocesamiento (`data/scripts/preprocess.py`)

**¿Qué hace?** Prepara los datos en el formato exacto que el modelo necesita.

### 4a. Balanceo
- Por complejidad: ~33% simple, 33% intermediate, 33% advanced
- Por bloque/dominio: ninguno domina más del 25%
- Reporte de distribución de conectores y átomos

### 4b. Formato con special tokens

Todo ejemplo se convierte en una secuencia con tokens especiales que le dicen
al modelo dónde empieza cada sección:

**Fase 1 (con Chain-of-Thought):**
```
<|input|> Si el server crashea y no hay backup, se pierden los datos
<|output|>
<|thought|> "si...entonces" indica implicación, "y" indica conjunción, "no" indica negación
<|atoms|> p: el server crashea | q: hay backup | r: se pierden los datos
<|connectors|> ∧: y | ¬: no | →: si...entonces
<|formula|> (p ∧ ¬q) → r
<|end|>
```

**Fase 2 (sin thought — para Chain-of-Thought Distillation):**
```
<|input|> Si el server crashea y no hay backup, se pierden los datos
<|output|>
<|atoms|> p: el server crashea | q: hay backup | r: se pierden los datos
<|connectors|> ∧: y | ¬: no | →: si...entonces
<|formula|> (p ∧ ¬q) → r
<|end|>
```

### 4c. Split (80/10/10)
- Split por **patrón lógico**, no random
- Ejemplo: si `(X ∧ X) → X` aparece en train, no aparece en test
- Así medimos si el modelo generaliza a estructuras nuevas

### 4d. Curriculum ordering
- Train se ordena: simple → intermediate → advanced
- El modelo aprende lo fácil primero (como un humano)

**Resultado**: `train.jsonl`, `val.jsonl`, `test.jsonl`

---

## Paso 5: Entrenar tokenizer (BPE)

**¿Qué es BPE?** Byte-Pair Encoding — un algoritmo que aprende a dividir texto
en sub-palabras basándose en frecuencia. Ejemplo:

```
"firewall" → ["fire", "wall"]     (palabras comunes se mantienen)
"bypassear" → ["by", "pass", "ear"]  (palabras raras se dividen)
"∧" → ["∧"]                       (símbolos lógicos = 1 token)
```

Se entrena sobre TODO el corpus (frases + fórmulas). Así el vocab cubre
tanto español como los símbolos lógicos.

---

## Paso 6: Construir el modelo (Transformer Decoder-Only)

### Decisión: Decoder-Only (no Encoder-Decoder)

**Encoder-Decoder** = 2 transformers (uno lee, otro genera). Usado por T5, BART.
**Decoder-Only** = 1 transformer que lee y genera en flujo continuo. Usado por GPT, LLaMA, Mistral.

**¿Por qué Decoder-Only?**
1. El formato del dataset es secuencial (`<|input|>...<|output|>...<|formula|>...`) — es exactamente cómo funciona Decoder-Only
2. CoT Distillation funciona natural: en Fase 2 solo quitás tokens de la secuencia
3. Un solo transformer = menos parámetros, cabe holgado en T4
4. Todas las técnicas avanzadas (Progressive Pruning, Contrastive Examples) fueron diseñadas para Decoder-Only

**¿Y no se pierde calidad?** No. Las capas que procesan `<|input|>` funcionan
como un "encoder implícito". Cuando el modelo llega a `<|output|>`, sus hidden
states ya contienen toda la comprensión. No necesitás encoder separado.

### Especificaciones (~15M parámetros, T4)

```
TODO: definir juntos cuando lleguemos aquí
- d_model: ?
- n_heads: ?
- n_layers: ?
- d_ff: ?
- max_seq_len: ?
- vocab_size: ? (lo determina el BPE)
```

### Componentes state of the art:

#### ✅ Causal Masking — "No ver el futuro"

Durante el entrenamiento, el modelo procesa TODA la secuencia de una vez (en paralelo).
Pero cada token solo debería poder ver los tokens **anteriores**, no los que vienen después.

La causal mask es una matriz triangular que bloquea el futuro:

```
Tokens:    A    B    C    D
A          ✅   ❌   ❌   ❌     ← A solo se ve a sí mismo
B          ✅   ✅   ❌   ❌     ← B ve A y a sí mismo
C          ✅   ✅   ✅   ❌     ← C ve A, B y a sí mismo
D          ✅   ✅   ✅   ✅     ← D ve todo
```

Se implementa poniendo **-infinito** en las posiciones futuras antes del softmax:

```python
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
mask = mask.masked_fill(mask == 1, float('-inf'))
# Softmax convierte -inf → 0.00 (atención cero al futuro)
```

**¿Por qué importa?** Sin esto, el modelo haría trampa durante entrenamiento
(vería las respuestas). Con la mask, cada posición es un "mini-examen" independiente.
Esto permite entrenar en paralelo ("teacher forcing") pero simular generación secuencial.

Durante **inferencia** (cuando generás), la mask no hace falta porque literalmente
no hay futuro — el modelo solo ve lo que ya generó.

---

#### ✅ RoPE (Rotary Position Embeddings) — "¿Dónde estoy?"

**Problema**: El Transformer es ciego a posiciones. "gato come pez" y "pez come gato"
le parecen iguales — mismos tokens, diferente orden.

**Solución vieja (2017)**: sumar un vector de posición al embedding. Funciona, pero
codifica posiciones **absolutas** — "estoy en la posición 5". Lo que importa en lenguaje
es la distancia **relativa** — "estoy 3 tokens después de esa palabra".

**RoPE (2021)**: en vez de sumar, **rota** el vector del embedding:

```
Posición 0:  →      (sin rotar)
Posición 1:  ↗      (rotada 10°)
Posición 2:  ↑      (rotada 20°)
Posición 3:  ↖      (rotada 30°)
```

**El truco matemático**: cuando dos tokens calculan atención (dot product de sus
vectores rotados), el resultado solo depende de la **diferencia de posiciones**:

```
dot(pos_5, pos_8) depende de 8-5 = 3
dot(pos_2, pos_5) depende de 5-2 = 3  ← ¡mismo resultado!
```

**En dimensiones altas** (d_model=256): Se agrupan en pares y cada par rota con
frecuencia diferente — como tener un segundero (detalle fino), minutero (medio)
y horario (largo plazo) para cubrir todas las escalas de distancia.

**Se aplica solo a Q y K** (query/key), NO a V (value):
- Q × K (con RoPE) = "**a quién** presto atención" (necesita posiciones)
- V (sin RoPE) = "**qué información** recojo" (no necesita posiciones)

**Lo usan**: LLaMA, Mistral, Qwen, Gemma, y casi todos los modelos modernos.

---

#### ✅ ALiBi (Attention with Linear Biases) — alternativa a RoPE

En vez de modificar embeddings, **penaliza directamente los scores de atención**
según la distancia entre tokens:

```
Score final = Q·K − m × distancia

Head 1: pendiente m = 1/2   (penaliza suave → ve "lejos")
Head 2: pendiente m = 1/4
Head 3: pendiente m = 1/8
Head 4: pendiente m = 1/16  (penaliza fuerte → se enfoca en lo "cercano")
```

Ventaja: **extrapola muy bien** a secuencias más largas que las de entrenamiento.
Usado por BLOOM, MPT, Falcon.

**¿Por qué NO lo usamos?** Nuestras secuencias son cortas (~100-200 tokens),
así que la extrapolación no importa. RoPE captura relaciones posicionales más ricas
que una simple resta lineal, y todo el ecosistema moderno (LLaMA, Mistral) usa RoPE.

---

#### ✅ SwiGLU — activación con compuerta inteligente

**Evolución de las activaciones:**

```
ReLU (2012):  max(0, x)              — simple pero neuronas "mueren" con valores negativos
GELU (2016):  x · Φ(x)              — más suave, no mueren, pero no filtra
SwiGLU (2020): Swish(gate) × content — FILTRA selectivamente
```

**FFN clásico** (con ReLU/GELU): una sola rama
```
input → W₁ → activación → W₂ → output
```

**FFN con SwiGLU**: dos ramas, una actúa como COMPUERTA de la otra
```
         input
        /     \
    W₁(x)   W_gate(x)
      |         |
      |      Swish()
      |         |
      ×─────────×     ← multiplicación: la gate decide qué pasa
      |
    W₂(x)
      |
    output
```

**¿Por qué es mejor?** La compuerta aprende a **bloquear selectivamente**
información irrelevante para cada token. ReLU/GELU aplican la misma transformación
a todas las dimensiones — SwiGLU puede decir "para este token, deja pasar
las dimensiones de polaridad y bloquea las de sustantivo".

**Costo**: 3 matrices en vez de 2. Se compensa reduciendo d_ff:
```
FFN clásico:  d_ff = 4 × d_model
FFN SwiGLU:   d_ff = (8/3) × d_model ≈ 2.67 × d_model
```
Mismos parámetros, mejor rendimiento.

**Lo usan**: LLaMA 1/2/3, Mistral, Gemma, PaLM — todos desde 2023.

---

#### ✅ RMSNorm — normalización más eficiente que LayerNorm

LayerNorm hace: `(x - mean) / std * gamma + beta` (4 operaciones).
RMSNorm hace: `x / rms(x) * gamma` (2 operaciones).
El paper demostró que restar la media aporta casi nada. Resultado: misma calidad, ~10-15% más rápido.
Lo usan: LLaMA 1/2/3, Mistral, Gemma.

Se usa 17 veces en nuestro modelo:
- 2 por capa (antes de attention + antes de FFN) x 8 capas = 16
- 1 final antes del head = 1
- Total params: 17 x 512 = 8,704 (0.03% del modelo, baratísimo pero crítico)

Detalles de implementación:
- `rsqrt` en vez de `sqrt` + división (más rápido)
- Upcast a float32 para la normalización (evita overflow en float16/bfloat16)
- Solo parámetro `gamma` (sin bias), empieza en 1.0

**Deep Norm** (Microsoft Research — "DeepNet: Scaling Transformers to 1,000 Layers"):
Escalar la conexión residual por alpha para que la señal original llegue fuerte
a las capas profundas: `output = x * alpha + sublayer(norm(x))`
- alpha = (2 * n_layers)^0.25 = 2.0 con 8 capas
- beta = (8 * n_layers)^-0.25 = 0.354 para inicialización de pesos

---

#### ✅ Attention Deep Dive — como funciona de verdad

##### ¿Qué es un tensor realmente?

Un tensor es una caja de números con dimensiones (shape).
Una **matriz** es un tensor de 2 dimensiones. Un tensor puede tener cualquier cantidad.

En nuestro modelo, el tensor principal tiene esta forma:
```
(8, 100, 512)
 │   │    └── 512 números que "describen" cada token
 │   └── 100 tokens en la secuencia
 └── 8 secuencias procesadas al mismo tiempo (batch)
```

¿Que significan esos 512 numeros por token? Cada uno captura una "propiedad" aprendida:
```
"llueve" = [0.8, -0.3, 0.1, 0.9, ..., -0.2]
            │     │     │    │
            │     │     │    └── ¿es una accion? (0.9 = muy si)
            │     │     └── ¿es positivo? (0.1 = neutro)
            │     └── ¿es un sustantivo? (-0.3 = no)
            └── ¿tiene que ver con clima? (0.8 = mucho)
```

En realidad las dimensiones no tienen nombres claros — el modelo aprende
que poner en cada una. Pero palabras similares terminan con vectores similares:
```
"llueve"   = [0.8, -0.3, 0.1, ...]  ─┐ similares
"llovizna" = [0.7, -0.2, 0.1, ...]  ─┘ (clima)
"mesa"     = [-0.5, 0.8, -0.3, ...]    completamente diferente
```

##### ¿Por que importa la posicion?

Sin posicion, el modelo ve las palabras como bolsa desordenada:
```
"El perro muerde al gato" = {perro, gato, muerde, el, al}
"El gato muerde al perro" = {perro, gato, muerde, el, al}  <-- MISMA BOLSA!
```

El modelo no sabria quien muerde a quien. RoPE rota cada vector segun su posicion:
- Posición 0: "El" → vector sin rotar
- Posición 1: "perro" → vector rotado 1 paso
- Posición 2: "muerde" → vector rotado 2 pasos

La rotacion hace que la similitud Q*K dependa de la **diferencia** de posiciones
(relativa), no de las posiciones absolutas.

##### El problema que Attention resuelve

El modelo procesa: "Si llueve entonces me ____"

Para predecir "mojo", necesita saber que:
- "llueve" es la CAUSA (super relevante)
- "entonces" indica CONSECUENCIA (relevante)
- "me" indica A QUIÉN (algo relevante)
- "Si" indica CONDICIONAL (relevante para la estructura)

¿Como decide que es relevante? Con **Q, K, V**.

##### Q, K, V — la analogia de la biblioteca

```
Q (Query)  = Tu pregunta: "Necesito info sobre la CAUSA"
K (Key)    = La etiqueta de cada libro: "Yo hablo sobre clima"
V (Value)  = El contenido del libro: la informacion real

Q y K se comparan para decidir que libros abrir.
V es lo que lees de los libros que decidiste abrir.
```

Cada token genera SU PROPIO Q, K y V multiplicando su vector por 3 matrices
de peso (W_Q, W_K, W_V). Estas matrices son los parametros que el modelo aprende.

##### Ejemplo paso a paso con numeros

4 tokens con vectores de 3 dimensiones (simplificado):
```
Tokens: ["Si", "llueve", "me", "mojo"]

Paso 1: Cada token genera Q, K, V
  "Si"     → Q=[1,0,0]  K=[0,1,0]  V=[0.1, 0.2, 0.3]
  "llueve" → Q=[0,1,1]  K=[1,1,0]  V=[0.8, 0.1, 0.5]
  "me"     → Q=[0,0,1]  K=[0,0,1]  V=[0.2, 0.3, 0.1]
  "mojo"   → Q=[1,1,0]  K=[1,0,1]  V=[0.4, 0.6, 0.2]
```

Para "mojo": ¿a quien deberia prestar atencion?
```
Paso 2: Producto punto de Q_mojo con K de cada token anterior

  Q_mojo = [1, 1, 0]

  Q_mojo · K_si     = 1*0 + 1*1 + 0*0 = 1    (algo relevante)
  Q_mojo · K_llueve = 1*1 + 1*1 + 0*0 = 2    (MUY relevante!)
  Q_mojo · K_me     = 1*0 + 1*0 + 0*1 = 0    (irrelevante)
  Q_mojo · K_mojo   = 1*1 + 1*0 + 0*1 = 1    (algo relevante)

  Scores = [1, 2, 0, 1]
```

El **producto punto mide similitud**. Q de "mojo" y K de "llueve"
apuntan en la misma direccion → score alto → atiende a "llueve".
```
Paso 3: Softmax — convertir scores a probabilidades

  softmax([1, 2, 0, 1]) = [0.18, 0.49, 0.07, 0.26]
                            │     │      │      │
                            │     49%!   │      └── "mojo" 26%
                            │  "llueve"  └── "me" 7%
                            └── "Si" 18%
```

"llueve" gana con 49% de la atencion.
```
Paso 4: Promediar los V ponderados por las probabilidades

  output = 0.18*V_si + 0.49*V_llueve + 0.07*V_me + 0.26*V_mojo
         = [0.52, 0.27, 0.37]
```

Este vector de salida ahora "sabe" que "mojo" esta relacionado con "llueve".
El vector original de "mojo" no tenia esa informacion. Despues de attention, si.

##### ¿Por que Multi-Head?

Una sola cabeza solo captura UNA relacion. Las palabras tienen muchas:
```
Cabeza 1: relaciones de CAUSA    → "mojo" atiende a "llueve"
Cabeza 2: relaciones de SINTAXIS → "mojo" atiende a "me" (sujeto)
Cabeza 3: relaciones de POSICION → "mojo" atiende a "entonces"
... (8 cabezas en nuestro modelo)
```

Cada cabeza tiene sus propias matrices W_Q, W_K, W_V → aprende a buscar
un tipo diferente de relacion. Despues se concatenan los resultados.

##### Causal Mask — prohibir ver el futuro

En entrenamiento procesamos TODA la oracion a la vez, pero cada token solo
puede ver los anteriores (sino haria trampa):
```
         Si  llueve  me  mojo
Si     [ OK   -inf  -inf  -inf ]   ← solo se ve a si mismo
llueve [ OK    OK   -inf  -inf ]   ← ve Si y a si mismo
me     [ OK    OK    OK   -inf ]   ← ve los 3 anteriores
mojo   [ OK    OK    OK    OK  ]   ← ve todo
```

Los -inf se eliminan con softmax(-inf) = 0.

##### Differential Attention — cancelar ruido (Microsoft 2024)

El softmax normal asigna atencion no-cero a TODOS los tokens, incluso
los irrelevantes. Differential Attention calcula DOS patrones y los resta:
```
attn = softmax(Q1*K1) - lambda * softmax(Q2*K2)

El ruido (similar en ambos patrones) se cancela.
Solo queda la señal real: los tokens que realmente importan.
```

Lambda es un parametro aprendible — el modelo decide cuanto ruido cancelar.

---

#### ✅ FFN Deep Dive — la "memoria interna" del Transformer

##### ¿Que hace la FFN?

Si attention es "a quien presto atencion", la FFN es
"ahora que ya se que es relevante, que hago con esa informacion?"

```
Attention: RECOPILAR informacion de otros tokens
FFN:       PROCESAR esa informacion → "pensar"
```

Estudios (Geva et al., 2021) demostraron que las FFN funcionan como
MEMORIA ASOCIATIVA: cada neurona almacena un patron (key) y una respuesta (value).
Es literalmente otra forma de attention pero sobre "memorias" aprendidas.

##### ¿Por que SwiGLU y no FFN clasica?

```
FFN clasica (GPT-2):
  x = Linear(512 -> 2048)   # expandir 4x
  x = ReLU(x)               # filtro binario: pasa o no pasa
  x = Linear(2048 -> 512)   # comprimir

SwiGLU (LLaMA/Mistral/Gemma):
  gate = Linear(512 -> 1365)    # la compuerta
  up   = Linear(512 -> 1365)    # el contenido  
  x    = SiLU(gate) * up        # compuerta FILTRA SELECTIVAMENTE
  x    = Linear(1365 -> 512)    # comprimir
```

ReLU es binario (si/no). SwiGLU tiene una compuerta suave que aprende
QUE dimensiones bloquear para cada token. Puede decir "para llueve,
deja pasar las dimensiones de clima y bloquea las de sintaxis".

Costo: 3 matrices en vez de 2 → se compensa con d_ff mas chico:
- FFN clasica: d_ff = 4 x d_model = 2048
- SwiGLU: d_ff = (8/3) x d_model = 1365
- Mismos parametros, mejor rendimiento.

##### Gate Residual (truco underground)

Agregar un bypass aprendible dentro de la compuerta:
```
Normal:          output = SiLU(gate) * up
Gate Residual:   output = SiLU(gate) * up + alpha * up
```

Si la compuerta se equivoca y bloquea algo importante,
`alpha * up` permite que pase igualmente.
Alpha empieza en 0 (sin efecto) y el modelo aprende si necesita usarlo.
Costo: 1 parametro escalar extra. Riesgo: cero.

##### Parametros de la FFN

```
gate_proj:  512 x 1365 =  698,880
up_proj:    512 x 1365 =  698,880
down_proj:  1365 x 512 =  698,880
Total:                   2,096,640 por capa
x 8 capas:             16,773,120 (~64% del modelo)
```

La FFN es por lejos el componente MAS caro en parametros.

---

#### 💡 MoE (Mixture of Experts) — por que NO lo usamos (aun)

MoE = N FFNs pequeñas especializadas + un router que elige cuales activar:
```
Token "llueve" → Router → Experto 2 (clima) + Experto 4 (semantica)
                          Los otros no se activan → ahorra computo
```

¿Por que no ahora?
- **Data insuficiente**: con 6,080 ejemplos, cada experto veria ~3,000.
  No hay suficiente diversidad para que se especialicen de verdad.
- **Riesgo de colapso**: los expertos terminan haciendo todos lo mismo.
- **Complejidad**: router, load balancing, auxiliary loss.

¿Cuando si? Con 20K+ ejemplos Y multiples tareas (NL→logica, logica→NL,
verificacion, simplificacion). Ahi cada experto podria especializarse.

La FFN esta diseñada modular para poder swapear a MoE en el futuro.

---

#### ✅ TransformerBlock — la pieza LEGO que se repite

El bloque es la unidad que se repite 8 veces. Cada bloque ensambla todo:
```
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
```

Cada capa aprende relaciones diferentes:
- Capas tempranas (1-3): relaciones simples (sintaxis, palabras cercanas)
- Capas medias (4-6): relaciones semanticas (significado, roles)
- Capas tardias (7-8): relaciones abstractas (logica, implicaciones)

##### Pre-Norm vs Post-Norm

```
Post-Norm (GPT-2):   output = Norm(x + sublayer(x))
Pre-Norm (LLaMA):    output = x + sublayer(Norm(x))     <-- usamos esta
```

Pre-Norm deja la conexion residual como camino limpio sin obstaculos.
Post-Norm pone la normalizacion EN el camino del residual, lo que puede
amortiguar señales importantes en redes profundas.

##### Stochastic Depth — regularizacion a nivel de bloque

Durante entrenamiento, saltarse bloques al azar con probabilidad creciente:
```
Capa 0: p=0.000 (nunca se salta — es critica)
Capa 3: p=0.043
Capa 7: p=0.100 (10% de chance de skip)
```

¿Por que funciona?
- Obliga al modelo a no depender de una sola capa
- Las capas profundas son mas "redundantes" → se saltean mas
- En inferencia: todos los bloques se ejecutan (escalados por 1/(1-p))
- Zero parametros extra, zero overhead en inferencia
- Con solo 6K datos, toda regularizacion extra ayuda

##### ¿Por que NO Parallel Attention + FFN?

PaLM/GPT-J ejecutan attention y FFN en paralelo:
```
Paralelo:    output = x + attention(norm(x)) + ffn(norm(x))
Secuencial:  output = x + ffn(norm(x + attention(norm(x))))    <-- usamos esta
```

En modelos grandes no pierde calidad. En modelos chicos (< 1B params),
la secuencialidad importa: la FFN necesita ver el OUTPUT de attention
(ya enriquecido con contexto), no el input crudo.

##### ¿Por que NO Adaptive Depth?

Adaptive Depth deja que el modelo decida cuantas capas usar por token.
No lo usamos porque:
- Necesita un loss auxiliar ("ponder cost") dificil de calibrar
- Con 6K datos, el halting score no aprende bien cuando parar
- Termina usando siempre todas las capas → mismo resultado, mas overhead
- Con 100K+ datos seria viable

---

#### ✅ NanoLogicTransformer — el modelo completo

El transformer ensambla todo en un flujo end-to-end:
```
Token IDs [45, 892, 12, 567]
     |
  Embedding (8000x512) → vectores    (* sqrt(512) para escalar)
     |
  8 x TransformerBlock               (attention + FFN + Deep Norm)
     |
  RMSNorm final                      (estabilizar antes de salida)
     |
  LM Head (512→8000)                 (pesos compartidos con Embedding!)
     |  * head_scale                  (calibracion aprendible)
     |  tanh soft-capping             (anti-overconfidence)
     |
  logits → loss / token predicho
```

##### Weight Tying — compartir pesos

Embedding (8000x512) convierte token_id → vector (buscar significado).
LM Head (512x8000) convierte vector → token_id (predecir token).
Son operaciones INVERSAS → compartir la misma matriz:
- Ahorra 4.1M params (16% del modelo)
- Fuerza consistencia semantica
- Actua como regularizacion
Lo usan: GPT-2, LLaMA, Mistral, Gemma — TODOS.

##### Output Soft-Capping (Gemma 2)

Igual que attention soft-capping pero para los logits finales:
`logits = 30 * tanh(logits / 30)`
Previene overconfidence: el modelo no puede estar 100% seguro.

##### Head Scale — calibracion aprendible

`logits = lm_head(x) * head_scale` — head_scale empieza en 1.0.
El modelo puede aprender a ser mas cauteloso (< 1) o mas seguro (> 1).
Un solo parametro, zero riesgo.

##### Z-Loss (PaLM) — regularizacion suave

`z_loss = 1e-4 * mean(logsumexp(logits)^2)`
Enseña al modelo a mantener logits en rango razonable.
Complementario a soft-capping: cap limita "a la fuerza",
z-loss enseña a no necesitar limitacion.

##### Deep Norm beta init

Los sublayers se inicializan con pesos escalados por:
`beta = (8 * n_layers)^-0.25 = 0.354`
Al inicio los sublayers contribuyen POCO y los residuales dominan.
El modelo gradualmente aprende a "abrir" los sublayers.

##### ¿Por que NO uP (Maximal Update Parameterization)?

uP permite transferir hiperparametros de modelos chicos a grandes.
Pero requiere scaling experiments (entrenar multiples tamaños).
Nosotros tenemos UN solo modelo de UN solo tamaño — no hay nada
que "transferir". Lo que si robamos de uP es la idea de escalar
la inicializacion por el ancho, que ya hacemos con Deep Norm beta.

---

#### ✅ Dataset Pipeline — como alimentar al modelo eficientemente

El dataset (`src/training/dataset.py`) es el "cocinero" que transforma archivos JSONL
en tensores listos para el modelo. Implementa 5 tricks de eficiencia integrados.

**El problema**: los datos están en texto. El modelo solo entiende tensores (matrices de números).

```
Archivo JSONL:
  {"sequence": "<|bos|><|input|> Si llueve me mojo <|output|>... <|formula|> p → q <|eos|>"}
  ...6080 ejemplos más

         ↓  dataset.py  ↓

Tensores para el modelo:
  input_ids:      [[1, 4, 563, 892, ...],    # (batch=8, seq_len=128)
                   [1, 4, 238, 447, ...], ...]
  targets:        [[4, 563, 892, ..., 2],     # shifted right (predecir SIGUIENTE)
                   [4, 238, 447, ..., 2], ...]
  attention_mask: [[1,1,1,...,0,0,0],         # 1=token real, 0=padding
                   [1,1,1,...,0,0,0], ...]
```

3 piezas principales:

```
1. Dataset   → "Acá están los datos" (acceso por índice)
2. Collator  → "Así los empaqueto" (padding, masks, batching)
3. DataLoader → "Así los sirvo" (shuffling, workers, prefetch)
```

##### Trick 1: Pre-tokenización offline

Tokenizar en cada epoch es repetir trabajo.

```
Sin pre-tokenización:
  Epoch 1: tokenizar 6080 ejemplos (3 seg) → entrenar
  Epoch 2: tokenizar 6080 ejemplos (3 seg) → entrenar  ← repetido!
  Epoch 3: tokenizar 6080 ejemplos (3 seg) → entrenar  ← repetido!

Con pre-tokenización:
  Setup:   tokenizar 6080 ejemplos (3 seg) → guardar en memoria
  Epoch 1: entrenar (0 seg tokenización)
  Epoch 2: entrenar (0 seg tokenización)
  Epoch 3: entrenar (0 seg tokenización)
```

Con 6K datos caben en RAM sin problema. Tokenizar una vez y cachear.

##### Trick 2: Dynamic Padding

El approach ingenuo rellena TODO a `max_seq_len=1024`:

```
Naive Padding:
  "Si llueve me mojo"      → [tok, tok, tok, tok, PAD, PAD, ..., PAD]  (1024)
  "Si A entonces B y C"    → [tok, tok, tok, tok, tok, tok, PAD, ..., PAD]  (1024)
                                                            ↑
                                           1018 PADs inútiles por secuencia!
```

Dynamic Padding rellena al **máximo del batch**, no al máximo global:

```
Dynamic Padding (batch de 4 secuencias cortas):
  "Si llueve me mojo"       → [tok, tok, tok, tok, PAD, PAD]  (6)
  "Si A entonces B y C"     → [tok, tok, tok, tok, tok, tok]  (6)
  "Llueve y truena"         → [tok, tok, tok, PAD, PAD, PAD]  (6)
  "A implica B"             → [tok, tok, tok, PAD, PAD, PAD]  (6)

  Total: 4 × 6 = 24 tokens procesados
  Naive: 4 × 1024 = 4096 tokens procesados
  Speedup: 170x menos cómputo para este batch!
```

Nuestras secuencias van de ~20 a ~300 tokens. Rellenar a 1024 sería desperdiciar 95%+ del cómputo.

##### Trick 3: Length Bucketing

Agrupar secuencias de largo similar en el mismo batch, minimizando el padding incluso con Dynamic Padding:

```
Sin bucketing (batch aleatorio):
  Secuencia 1:  30 tokens
  Secuencia 2: 250 tokens  ← fuerza padding a 250 para TODOS
  Secuencia 3:  15 tokens
  Secuencia 4:  22 tokens
  → Padded a 250. Desperdicio: 683 PADs

Con bucketing:
  Batch A: [15, 22, 28, 30]    → padded a 30.  Desperdicio: 45 PADs
  Batch B: [240, 245, 248, 250] → padded a 250. Desperdicio: 17 PADs
  → Total desperdicio: 62 PADs (vs 683!)
```

Algoritmo (`BucketBatchSampler`):
1. Ordenar índices por largo de secuencia
2. Crear mega-buckets de `batch_size × 10` ejemplos
3. Shufflear dentro de cada mega-bucket
4. Particionar en batches de `batch_size`
5. Shufflear el orden de los batches (sin sesgo sistemático)

##### Trick 4: Packing + Document Mask (Underground)

La técnica más underground y efectiva. En vez de 1 ejemplo por secuencia,
empaquetar múltiples ejemplos hasta llenar `max_seq_len`:

```
Sin packing (padding):
  Seq 1: [BOS, ejemplo_1, EOS, PAD, PAD, PAD, PAD]     128 tokens (50 reales)
  Seq 2: [BOS, ejemplo_2, EOS, PAD, PAD, PAD, PAD]     128 tokens (35 reales)
  Seq 3: [BOS, ejemplo_3, EOS, PAD, PAD, PAD, PAD]     128 tokens (40 reales)
  → 384 tokens procesados, 125 reales (32% eficiencia)

Con packing:
  Seq 1: [BOS, ej_1, EOS, BOS, ej_2, EOS, BOS, ej_3, EOS, PAD]  128 tokens (125 reales)
  → 128 tokens procesados, 125 reales (98% eficiencia!)
```

**El problema**: con la causal mask normal, el ejemplo 2 puede VER al ejemplo 1.
Pero son oraciones completamente distintas — mezclarlas genera correlaciones espurias.

**La solución**: Document Mask — mask block-diagonal causal. Cada documento solo
puede atender a tokens de SU MISMO documento:

```
                  ej_1      ej_2      ej_3
ej_1:       [ ✅ ✅ ✅ | ❌ ❌ ❌ | ❌ ❌ ❌ ]
ej_2:       [ ❌ ❌ ❌ | ✅ ✅ ✅ | ❌ ❌ ❌ ]  ← solo ve SU documento
ej_3:       [ ❌ ❌ ❌ | ❌ ❌ ❌ | ✅ ✅ ✅ ]  ← solo ve SU documento
```

Implementación: `build_document_mask()` construye esta mask a partir de
un vector `doc_ids` que indica a qué documento pertenece cada token.

##### Trick 5: Curriculum Learning

Entrenar primero con ejemplos fáciles y luego agregar los difíciles.
El modelo construye entendimiento de abajo hacia arriba (como aprender a caminar antes de correr):

```
Epochs  0-4:  Solo "Simple"       → p → q, p ∧ q
Epochs  5-14: + "Intermediate"    → p ∧ q → r
Epochs 15-30: + "Advanced"        → (p ∧ q) → (r ∨ ¬s)
```

Nuestro dataset ya tiene la columna `complexity` → solo es cuestión de filtrar por epoch.

Componentes principales en `dataset.py`:

| Componente | Función |
|------------|---------|
| `NanoLogicDataset` | Carga + pre-tokeniza + filtra por complejidad |
| `NanoLogicCollator` | Dynamic Padding ó Packing + Document Mask |
| `BucketBatchSampler` | Agrupa por largo similar |
| `pack_examples()` | Empaqueta múltiples docs en una secuencia |
| `build_document_mask()` | Mask block-diagonal causal |
| `create_dataloader()` | Fábrica que ensambla todo el pipeline |

---

#### ✅ LightningModule — el director del entrenamiento

El LightningModule (`src/training/lit_module.py`) es el "director de orquesta"
que coordina todo el entrenamiento. Con Lightning, defines QUÉ hacer y él se
encarga del CÓMO (GPU, mixed precision, checkpoints, logging, etc.).

```
Sin Lightning (manual ~200 líneas):       Con Lightning (lit_module.py):
┌────────────────────────────────┐       ┌────────────────────────────────┐
│ for epoch in range(100):        │       │ class LitNanoLogic:             │
│   for batch in loader:          │       │                                 │
│     optimizer.zero_grad()       │       │   training_step(batch):         │
│     outputs = model(batch)      │       │     return loss                 │
│     loss.backward()             │       │                                 │
│     clip_gradients(model)       │       │   configure_optimizers():       │
│     optimizer.step()            │       │     return AdamW(...)           │
│     scheduler.step()            │       └────────────────────────────────┘
│     if step % 100: log(...)     │       Lightning se encarga de:
│     if step % 1000: save(...)   │       ✅ GPU/multi-GPU    ✅ Logging
│     # manejar GPU/fp16/crash... │       ✅ Mixed precision  ✅ Checkpoints
└────────────────────────────────┘       ✅ Gradient clipping ✅ Resume
```

4 piezas del LitModule:

```
1. training_step()         → "Así entreno un batch"
2. validation_step()       → "Así evalúo un batch"
3. configure_optimizers()  → "Qué optimizer y scheduler usar"
4. train_dataloader()      → "De dónde vienen los datos"
```

##### Trick 1: Schedule-Free AdamW (Facebook Research, 2024)

El descubrimiento más importante de optimización reciente. Elimina el scheduler.

```
Approach clásico:
  lr = warmup → cosine decay → 0
  Problemas:
    - Elegir steps de warmup, cuándo decae, etc.
    - Si entrenas más de lo planeado, el lr ya está en 0
    - 3+ hiperparámetros extra que tunear

Schedule-Free:
  lr = constante TODO el entrenamiento
  El optimizer interpola internamente entre dos sequences de pesos
  → Converge igual o MEJOR que cosine decay
  → Zero hiperparámetros de scheduling
```

Incluye fallback a AdamW + Cosine Decay si `schedulefree` no está instalado.

##### Trick 2: Gradient Noise Injection (Underground, Google)

Agrega ruido gaussiano decreciente a los gradientes:

```python
noise = sqrt(eta / (1 + t)^gamma) * N(0, 1)
grad = grad + noise
```

El ruido ayuda a escapar mínimos locales malos. Es como sacudir una pelota
en un valle para que caiga a un valle más profundo. Con pocos datos (6K),
el landscape del loss es más irregular → más mínimos locales → más beneficio.

El ruido decae con el tiempo: al inicio explora mucho, al final se estabiliza
(como simulated annealing). Implementado en el hook `on_after_backward()`.

##### Trick 3: EMA de pesos (Exponential Moving Average)

Mantiene una copia "suavizada" de los pesos del modelo:

```
ema_weight = 0.999 × ema_weight + 0.001 × current_weight
```

Los pesos actuales oscilan durante entrenamiento. El EMA elimina las oscilaciones:

```
Paso 100: weight = 0.5    ema = 0.50
Paso 200: weight = 0.8    ema = 0.65
Paso 300: weight = 0.3    ema = 0.55  ← más estable que 0.3
Paso 400: weight = 0.7    ema = 0.60
```

En validación e inferencia se usan los pesos EMA → predicciones más consistentes.
Implementado con `swap_to_ema()` / `swap_from_ema()`.

##### Trick 4: Label Smoothing

En vez de target 100% seguro ("mojo" = probabilidad 1.0), suavizar para evitar
overconfidence:

```
Sin smoothing:  target = [0, 0, 0, 1.0, 0, 0, 0]    → overconfident
Con smoothing:  target = [0.02, 0.02, 0.02, 0.88, 0.02, 0.02, 0.02]  → humilde
```

Usando `cross_entropy(label_smoothing=0.1)` de PyTorch.

##### Trick 5: Gradient Clipping por norma global

Si los gradientes se hacen muy grandes, el entrenamiento diverge:

```
Sin clipping:  grad = [1000, -2000, 500]  → paso gigante → loss = NaN
Con clipping:  grad = [0.5, -1.0, 0.25]   → paso controlado → estable
```

La norma global (vs per-parameter) mantiene la DIRECCIÓN del gradiente intacta,
solo escala la magnitud. `max_norm=1.0` es el estándar.

##### Trick 6: Mixed Precision (bf16)

Entrenar con números de 16 bits en vez de 32:

```
fp32: 32 bits → más preciso, 2x más lento, 2x más memoria
bf16: 16 bits → menos preciso, 2x más rápido, 2x menos memoria
```

bf16 es mejor que fp16 porque tiene el mismo rango que fp32 (solo pierde
precisión). No necesita loss scaling. Se configura en el Trainer de Lightning
con `precision="bf16-mixed"`.

##### Trick 7: Gradient Accumulation

Simular batch sizes grandes sin explotar la memoria:

```
GPU tiene 8GB → caben 4 ejemplos por batch
Queremos batch efectivo de 32

Sin accumulation:  batch=4, actualizar cada 4 ejemplos     (ruidoso)
Con accumulation:  batch=4 × 8 micro-steps = 32 efectivo   (estable)
```

Acumular gradientes de N micro-batches antes de `optimizer.step()`.
Se configura en Lightning con una línea: `accumulate_grad_batches=8`.

**Weight Decay selectivo**: no se aplica a biases, norms, ni embeddings.
Solo a pesos de capas lineales. Esto es estándar en todos los LLMs modernos.

**TrainingConfig**: todos los hiperparámetros con defaults razonables:

| Parámetro | Default | Función |
|-----------|---------|---------|
| `lr` | 1e-3 | Learning rate |
| `weight_decay` | 0.1 | Regularización L2 selectiva |
| `batch_size` | 8 | Ejemplos por micro-batch |
| `accumulate_grad_batches` | 4 | Batch efectivo = 32 |
| `label_smoothing` | 0.1 | Anti-overconfidence |
| `gradient_clip_norm` | 1.0 | Max norma de gradientes |
| `gradient_noise_eta` | 0.1 | Escala inicial del ruido |
| `ema_decay` | 0.999 | Factor de suavizado EMA |
| `curriculum_schedule` | {0:0, 5:1, 15:2} | Simple → Inter → Advanced |

---

#### ✅ Train Entry Point — el botón START

El entry point (`train.py`) es el archivo que ejecutas para arrancar el entrenamiento.
Ensambla todas las piezas: tokenizer, modelo, datos, callbacks, y Trainer.

```bash
python train.py                                    # defaults (todos los tricks ON)
python train.py --lr 5e-4 --batch-size 16          # override hiperparámetros
python train.py --debug --fast-dev-run              # test rápido con anomaly detection
python train.py --compile                           # 1.5-2x speedup con torch.compile
python train.py --resume models/checkpoints/last.ckpt  # resumir entrenamiento
```

##### Trick 1: Smart Checkpointing

No guardar TODOS los checkpoints (llenan disco). Solo guardar:
- Los **top-K** modelos por `val/loss` (K=3)
- El **último** checkpoint (para resumir si crashea)

```
Naive:  epoch_1.ckpt, epoch_2.ckpt, ..., epoch_30.ckpt  → 30 × 80MB = 2.4GB
Smart:  best_1.ckpt, best_2.ckpt, best_3.ckpt, last.ckpt → 4 × 80MB = 320MB
```

##### Trick 2: Auto-detect Precision

Detecta automáticamente qué precisión soporta la GPU:

```
GPU A100/H100 (Ampere+):  bf16-mixed  (mejor opción)
GPU T4/V100 (Turing):     16-mixed    (fp16 con loss scaling)
GPU antigua / CPU:         32          (sin aceleración)
```

No hardcodear — funciona en Colab (T4) y en GPUs mejores sin cambiar código.

##### Trick 3: Seed Everything

Fijar TODAS las semillas aleatorias: PyTorch, NumPy, Python, CUDA.
Si corres el mismo script dos veces, obtienes el mismo resultado exacto.
Crucial para debugging y reproducibilidad.

##### Trick 4: Anomaly Detection (modo debug)

PyTorch detecta operaciones que producen NaN o Inf y dice EXACTAMENTE qué
operación lo causó. Es lento (solo para debug con `--debug`), pero te salva
horas de búsqueda cuando algo falla.

##### Trick 5: torch.compile (Underground, PyTorch 2.0+)

Compila el modelo a un grafo optimizado: fusiona operaciones, elimina
redundancias, usa kernels CUDA optimizados.

```
Sin compile:  matmul → ReLU → matmul → softmax → matmul  (5 kernel launches)
Con compile:  [matmul+ReLU+matmul] → [softmax+matmul]     (2 kernel launches)
Speedup: 1.5-2x gratis
```

El primer paso es lento (compilación). Después vuela. Se activa con `--compile`.

##### Trick 6: Gradient Checkpointing

Recalcular activaciones en backward en vez de guardarlas en memoria.
-50% memoria, +20% tiempo. Solo si hay OOM. Se activa con `--grad-ckpt`.

##### Trick 7: CLI con argumentos

Override de cualquier hiperparámetro sin tocar el código:

```bash
python train.py --lr 5e-4 --max-epochs 50 --batch-size 16
python train.py --no-schedule-free --no-ema    # desactivar tricks
python train.py --curriculum "0:0,10:1,20:2"   # curriculum custom
```

Resumen de ejecución al iniciar:

```
🚀 INICIANDO ENTRENAMIENTO
   Modelo:        21,000,000 params
   Batch size:    8 × 4 = 32 efectivo
   LR:            0.001
   Precision:     bf16-mixed
   Packing:       ✅
   Schedule-Free: ✅
   EMA:           ✅
   Curriculum:    {0: 0, 5: 1, 15: 2}
```

---

---

#### ✅ Evaluación — ¿realmente funciona el modelo?

El módulo de evaluación (`src/evaluation/`) mide si el modelo genera fórmulas
correctas. El `val/loss` dice qué tan bien predice tokens, pero NO dice si la
fórmula resultante es lógicamente correcta.

3 piezas de evaluación:

```
1. metrics.py      → "¿Cuántas fórmulas acertó?"
2. truth_table.py  → "¿Son lógicamente equivalentes?"
3. benchmark.py    → "¿Cómo le va en cada categoría?"
```

##### Trick 1: Equivalencia Semántica por Tabla de Verdad

La métrica más importante. Dos fórmulas son equivalentes si tienen la misma
tabla de verdad, incluso si el texto es diferente:

```
p ∧ q → r   vs   q ∧ p → r

p | q | r | p∧q→r | q∧p→r
0 | 0 | 0 |   1   |   1
...
1 | 1 | 0 |   0   |   0     ← mismos valores en TODAS las filas
1 | 1 | 1 |   1   |   1

Resultado: EQUIVALENTES ✅
```

##### Trick 2: Compositional Metrics (Evaluation Layers)

Desglosa la evaluación en 4 niveles de composición para diagnosticar el error exacto:

```
Nivel 1 — Átomos:          ¿identificó p, q, r?
Nivel 2 — Sub-fórmulas:    ¿armó p∧q, ¬s, r∨¬s correctamente?
Nivel 3 — Conector raíz:   ¿eligió → como conector principal?
Nivel 4 — Fórmula total:   ¿es equivalente?
```

##### Trick 3: Analysis Dimensions (Benchmark)

Microscopio completo del rendimiento:

- **Por Complejidad**: Simple 95% → Inter 82% → Advanced 43%
- **Por Conector**: ∧ 93%, ∨ 85%, → 78%, ↔ 52%
- **Por Bloque**: Causal 87%, Temporal 79%, Científico 65%
- **Por Largo**: 1-2 conectores 91% → 5+ conectores 38%

##### Trick 4: Confusion Matrix (Underground)

No solo accuracy, sino ¿CON QUÉ lo confunde?

```
Real: →
Pred: ↔  (15 veces)

Diagnóstico: El modelo confunde implicación con bicondicional.
```

##### Trick 5: Scaling Analysis (Underground)

¿Cómo escala la accuracy con la cantidad de átomos?
- 2 átomos: 94%
- 3 átomos: 85%
- 5 átomos: 42% (techo de composicionalidad)

##### Trick 6: Partial Credit (Tree Edit Distance)

En vez de 0/1, crédito parcial basado en similitud de árboles (AST):
`p ∧ q → r` vs `p ∧ q → s` = 0.8 (solo 1 nodo diferente).

Componentes por archivo:

| Archivo | Función |
|---------|---------|
| `truth_table.py` | Parser, AST, tabla de verdad, equivalencia semántica |
| `metrics.py` | Exact match, partial credit, compositional score, normalización |
| `benchmark.py` | Desglose por complejidad/conector/bloque, confusion matrix |

---

#### ✅ Special Tokens — el protocolo de comunicación del modelo

Son tokens inventados que NO existen en el lenguaje natural. Le dan estructura
a las secuencias para que el modelo sepa qué es qué.

Formato de una secuencia de entrenamiento:
```
<|bos|><|input|> Si llueve me mojo <|output|><|thought|> "si...entonces"
indica implicación <|atoms|> p: llueve | q: me mojo <|connectors|>
→: si...entonces <|formula|> p → q <|eos|>
```

Tokens definidos:

| Token | Nombre | Función |
|-------|--------|--------|
| `<\|pad\|>` | Padding | Relleno para igualar largo en batches |
| `<\|bos\|>` | Begin of Seq | "Aquí empieza todo" |
| `<\|eos\|>` | End of Seq | "Ya terminé" — el modelo para de generar |
| `<\|unk\|>` | Unknown | Caracteres desconocidos (no debería aparecer) |
| `<\|input\|>` | Input | Aquí va el texto en español |
| `<\|output\|>` | Output | Aquí empieza lo que el modelo genera |
| `<\|thought\|>` | Thought | Razonamiento paso a paso (CoT, Fase 1) |
| `<\|atoms\|>` | Atoms | Átomos proposicionales identificados |
| `<\|connectors\|>` | Connectors | Conectores lógicos identificados |
| `<\|formula\|>` | Formula | La fórmula final (salida principal) |

**¿Por qué `<|...|>`?** Estándar de la industria (GPT, LLaMA, Mistral).
Los delimitadores hacen imposible confundir un token especial con texto normal.

**¿Por qué `<|eos|>` y no `<|end|>`?** EOS (End of Sequence) es el nombre estándar
en todos los modelos modernos. Consistencia con el ecosistema.

Archivo: `src/tokenizer/special_tokens.py` — dataclass frozen (inmutable).

---

#### ✅ BPE (Byte Pair Encoding) — cómo el modelo "lee" texto

El modelo trabaja con números, no texto. BPE convierte texto → tokens (números).

**3 enfoques posibles:**
- Por caracteres: `"llueve"` → `["l","l","u","e","v","e"]` — secuencias muy largas
- Por palabras: `"llueve"` → `["llueve"]` — no maneja palabras nuevas (`<UNK>`)
- **BPE**: punto medio — palabras frecuentes enteras, raras en sub-pedazos

**Algoritmo (simplificado):**
```
1. Empezar con caracteres individuales como vocabulario
2. Contar qué PARES de tokens aparecen más seguido
3. Fusionar el par más frecuente en un nuevo token
4. Repetir hasta alcanzar el vocab_size deseado
```

**Ejemplo:**
```
Corpus: "llueve llueve lluvia"

Paso 0: l l u e v e  l l u e v e  l l u v i a
Paso 1: [ll] aparece 3 veces → fusionar → [ll]u e v e ...
Paso 2: [llu] aparece 3 veces → fusionar → [llu] e v e ...
Paso 3: [ev] aparece 2 veces → fusionar → [llu][ev] e ...
...y así hasta tener el vocabulario deseado.
```

**¿Por qué entrenamos NUESTRO propio BPE?**
Nuestro modelo trabaja con dos "idiomas": español + lógica proposicional (→, ∧, ¬).
Un tokenizer genérico no sabría manejar los símbolos lógicos. El nuestro sí,
porque lo entrenamos con nuestros datos procesados (6,080 ejemplos).

**Protección de special tokens:**
```
SIN protección: "<|formula|>" → ["<", "|", "formula", "|", ">"]  ← MAL
CON protección: "<|formula|>" → ["<|formula|>"]                   ← BIEN
```

**Vocab size para NanoLogic: ~4,000-8,000 tokens**
(GPT-2 usa 50K, LLaMA 32K — pero ellos cubren TODO el idioma inglés.
Nosotros solo cubrimos español + lógica, así que con muchos menos alcanza.)

**Resultado del entrenamiento BPE:**

| Propiedad | Valor |
|-----------|-------|
| Vocab size total | 8,000 |
| Special tokens | 10 (IDs 0-9) |
| BPE tokens | 7,990 |
| PAD ID | 0 |
| BOS ID | 1 |
| EOS ID | 2 |

Todos los special tokens se mantienen como 1 solo token (no se parten) ✅
Guardado en `models/tokenizer/tokenizer.json`.

Archivos:
- `src/tokenizer/tokenizer.py` — wrapper NanoLogicTokenizer (encode, decode, save, load)
- `data/scripts/train_tokenizer.py` — script de entrenamiento

**¿Se puede mejorar el tokenizer?** Sí, pero el impacto es marginal (~1-2%).
Alternativas: WordPiece, Unigram, SentencePiece — pero BPE ByteLevel es el estándar
y funciona bien para nuestro caso (español + lógica proposicional).

**Lección pragmática:** El tokenizer importa menos que la arquitectura y los datos.
Papers de DeepMind y Meta muestran que con un tokenizer "decente" y buenos datos,
el modelo aprende bien. Prioridad de impacto en rendimiento:
1. Más datos de calidad (+10-20%)
2. Mejor arquitectura (+5-15%)
3. Mejor training strategy (+5-10%)
4. Mejor tokenizer (+1-2%) ← no vale la pena optimizar primero

**Decisión:** dejamos el tokenizer como está y avanzamos al modelo.
Si después vemos que las fórmulas se tokenizan mal, iteramos.

---

## Paso 6: Arquitectura del modelo

#### ✅ Decoder-Only — por qué este tipo de Transformer

3 tipos de Transformers:
- **Encoder-Only** (BERT): clasificación, no genera texto → ❌
- **Encoder-Decoder** (T5): traducción, pero overhead innecesario → 🟡
- **Decoder-Only** (GPT, LLaMA, Mistral): generación secuencial → ✅

Decoder-Only es ideal porque nuestro problema es generación condicional:
dado texto en español, GENERA la fórmula token por token.

#### ✅ Config "grado militar" — máximo rendimiento por parámetro

Filosofía: cada técnica que cuesta 0 en computación pero da +1% de rendimiento, la incluimos.

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `d_model` | 512 | Balance capacidad vs costo |
| `n_layers` | 8 | Más profundo = mejor razonamiento |
| `n_heads` | 8 | 512/8=64 dim por head (estándar) |
| `n_kv_heads` | 2 | GQA: ahorra 50% en KV cache |
| `d_ff` | 1365 | SwiGLU: (8/3)*512 |
| `vocab_size` | 8000 | Ya entrenado |
| `max_seq_len` | 1024 | Holgura para secuencias largas |
| `dropout` | 0.1 | Regularización estándar |

~20M params total. Inspirado en LLaMA 3 + Gemma 2.

#### ✅ Tricks incluidos (estado del arte)

| Técnica | ¿Qué hace? | Usado por |
|---------|-----------|-----------|
| **Pre-Norm** | Normalizar ANTES de attention/FFN (más estable) | LLaMA, Mistral |
| **GQA** | Varias heads comparten K,V (50% menos KV) | Mistral, LLaMA 2+ |
| **QK-Norm** | Normalizar Q,K antes de attention (evita explosión) | Gemma |
| **Embed Scale** | Escalar embeddings por √d_model | Transformer original |
| **Logit Soft-Capping** | Limitar logits de atención con tanh (estabilidad) | Gemma 2 |
| **Weight Tying** | Embedding y head comparten pesos (ahorra 4M params) | GPT-2, LLaMA |
| **RoPE** | Posiciones relativas por rotación | LLaMA, Mistral |
| **SwiGLU** | FFN con gating (mejor que ReLU/GELU) | LLaMA, Mistral |
| **RMSNorm** | Normalización sin media (más eficiente) | LLaMA |

#### ✅ Scaling Laws — por qué 20M y no 100M

Google Colab T4 tiene 16GB VRAM — aguanta modelos de hasta ~150M params.
Pero más params ≠ mejor. Ley de Chinchilla:

> Para que un modelo aproveche sus parámetros, necesita datos proporcionales.

| Params | Datos necesarios | ¿Tenemos? |
|--------|-----------------|-----------|
| 20M | ~400K tokens | ✅ Sí |
| 50M | ~1M tokens | 🟡 Borderline |
| 100M | ~2M tokens | ❌ No |

Con ~6,080 ejemplos (secuencias largas con CoT), un modelo de 20-25M es el sweet spot.
Si después metemos más datos (Detective, leyes de Chile), escalamos a 30-40M.

#### Estructura de archivos del modelo

```
src/model/
  ├── config.py        ← hiperparámetros (dataclass)
  ├── attention.py     ← Multi-Head Attention + RoPE + GQA + QK-Norm
  ├── ffn.py           ← SwiGLU Feed-Forward Network
  ├── rmsnorm.py       ← RMSNorm
  ├── block.py         ← Decoder Block (attention + ffn + residuals)
  └── transformer.py   ← Modelo completo
```

#### ✅ config.py — centralización de hiperparámetros

Se usa un `dataclass(frozen=True)` para que nadie modifique los hiperparámetros
después de crear el config. Si quieres otro config, creas nueva instancia.

Conteo de parámetros con defaults (~26M brutos, ~22M netos con weight tying):
- Embedding: 8,000 × 512 = 4.1M
- Por capa: Attention (~390K) + SwiGLU FFN (~2.1M) + RMSNorm (~1K) ≈ 2.5M
- 8 capas × 2.5M = 20M
- Head: 0 (weight tying con embedding)

Incluye `count_params_estimate()` para inspeccionar la distribución de parámetros.
Incluye `to_json()` / `from_json()` para guardar junto al modelo (reproducibilidad).
Validaciones en `__post_init__` atrapan errores inmediatamente (ej: d_model no divisible por n_heads).

---

## Paso 7: Entrenamiento (PyTorch Lightning)

TODO: notas sobre el training loop cuando lleguemos ahí.

Temas pendientes:
- Chain-of-Thought Distillation (2 fases)
- Progressive Output Pruning
- Label Smoothing Selectivo
- Contrastive Examples
- Auxiliary Losses

---

## 🔮 Visión futura: NanoLogic Stack (3 modelos)

Una vez el Formalizador esté entrenado, el plan es construir un **stack de 3 capas**:

```
         ┌─────────────────────────┐
Frase ──►│  Modelo 1: FORMALIZADOR │──► fórmula + átomos + conectores
         └───────────┬─────────────┘
                     │
         ┌───────────▼─────────────┐
         │  Modelo 2: VERIFICADOR  │──► VÁLIDO / INVÁLIDO / FALACIA FORMAL
         └───────────┬─────────────┘   (truth tables, Python puro, NO es modelo)
                     │
         ┌───────────▼─────────────┐
         │  Modelo 3: DETECTIVE    │──► FALACIA INFORMAL detectada
         └─────────────────────────┘   + nombre + explicación
```

### Modelo 1: Formalizador (lo que estamos construyendo)
- NL (español) → fórmula de lógica proposicional
- Decoder-Only, ~15M params
- Dataset: ~2,334 ejemplos minados con DeepSeek

### Modelo 2: Verificador (código puro, NO modelo)
- Truth tables + reglas deterministas
- Cero error, verificable, sin incertidumbre
- Detecta falacias FORMALES (ej: afirmación del consecuente)

### Modelo 3: Detective de Falacias Informales (futuro, ~$1.50)
- Detecta lo que la lógica formal NO atrapa:
  - Hombre de paja, ad hominem, falsa causa, pendiente resbaladiza
- Dataset nuevo: ~3,000 ejemplos (~$1.50)

### Entrenamiento Adversarial Cooperativo (self-play)
Los modelos se desafían mutuamente para mejorar:

```
Round 1: Detective genera frases con falacias ocultas
         → Formalizador las formaliza
         → Si la fórmula es "válida" pero hay falacia → hard negative para Formalizador

Round 2: Formalizador genera fórmulas válidas
         → Detective busca falacias
         → Si dice "falacia" cuando no hay → falso positivo para Detective

Round 3: Repeat → ambos mejoran (estilo AlphaZero)
```

**Potencial de paper:**
> *"NanoLogic Stack: Formal and Informal Fallacy Detection through
> Adversarial Cooperation of Specialized Micro-Models"*

---

## 💰 Control de costos

| Concepto | Costo | Estado |
|----------|-------|--------|
| Minado del dataset (2,334 ej.) | ~$1.00 | ✅ Completado |
| Verificación con API (2,210 ej.) | ~$0.50 | 🔄 En progreso |
| Augmentation | $0.00 | ⬜ Pendiente (gratis, Python puro) |
| Preprocessing | $0.00 | ⬜ Pendiente (gratis, Python puro) |
| **Total Formalizador** | **~$1.50** | |
| Dataset Detective (futuro) | ~$1.50 | ⬜ Futuro |
| **Balance actual** | **$3.49** | Sobra de sobra |

---

## Herramientas

| Tool | Para qué |
|------|----------|
| PyTorch | Construir el modelo |
| PyTorch Lightning | Orquestar el entrenamiento |
| HuggingFace Tokenizers | Entrenar el BPE tokenizer |
| Rich | Output bonito en terminal (barras de progreso, tablas) |
| TensorBoard | Visualizar métricas de entrenamiento |
| Google Colab (T4) | GPU gratis para entrenar |
| DeepSeek API | Minado y verificación de datos |

---

## 📋 Progreso

- [x] Minado del dataset (2,334 ejemplos raw)
- [x] Limpieza — `clean.py` (2,210 limpios, 94.7% retención)
- [x] Verificación — `verify.py` (2,118 verificados, 1,924 corregidos, 93 rechazados)
- [x] Augmentation — `augment.py` (6,761 ejemplos — 3x el original, gratis)
- [x] Preprocessing — `preprocess.py` (6,080 balanceados → train/val/test)
- [x] Entrenar tokenizer BPE (vocab=8,000, guardado en models/tokenizer/)
- [ ] Construir modelo (Decoder-Only, ~15M params)
- [ ] Entrenar en Colab (T4)
- [ ] Evaluar y benchmark

### Resultado del pipeline completo:

```
dataset.json (2,334 raw)
    ↓ clean.py     → 2,210 (94.7% retención)
    ↓ verify.py    → 2,118 (1,924 corregidos por la API)
    ↓ augment.py   → 6,761 (equivalencias lógicas + composiciones, gratis)
    ↓ preprocess.py → 6,080 (balanceado por complejidad)
    ✅ LISTO → data/processed/
```

### Split final:

| Split | Ejemplos | Archivo |
|-------|----------|---------|
| Train | 4,864 | `data/processed/train.jsonl` |
| Val | 617 | `data/processed/val.jsonl` |
| Test | 599 | `data/processed/test.jsonl` |

### Distribución del dataset final:

| Complejidad | % |
|-------------|---|
| Simple | 28% |
| Intermediate | 35% |
| Advanced | 37% |

| Conector | Presencia |
|----------|-----------|
| ∧ (AND) | 97.8% |
| ¬ (NOT) | 83.0% |
| → (IMPLICA) | 70.5% |
| ∨ (OR) | 66.0% |
| ↔ (BICOND) | 40.8% |

