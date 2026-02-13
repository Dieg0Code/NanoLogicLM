"""
Generador de dataset sintético v3 — Bloques temáticos con auto-switch.
Genera 2,500 ejemplos en 5 bloques de 500, cada uno con prompt y topics distintos.

Uso:
    python generate_dataset_v3.py
    python generate_dataset_v3.py --total 2500
    python generate_dataset_v3.py --total 2500 --output dataset.json
"""

import json
import os
import random
import pathlib
import argparse
import gc
import time
import re

from dotenv import load_dotenv
from openai import OpenAI

# --- Cargar .env ---
load_dotenv()
assert os.getenv("DEEPSEEK_API_KEY"), "❌ DEEPSEEK_API_KEY no encontrada en .env"

# --- Cliente DeepSeek ---
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# =============================================
# SCHEMA JSON (compartido por todos los bloques)
# =============================================

JSON_SCHEMA = """{
  "natural_language_input": "string - enunciado técnico en español",
  "complexity": "simple | intermediate | advanced",
  "thought": {
    "reasoning_steps": [
      {"step": 1, "explanation": "string - paso de razonamiento"},
      {"step": 2, "explanation": "string - paso de razonamiento"}
      ... (la explicación debe ser detallada, paso a paso, y mostrar cómo se identifican los átomos y conectores) ...
    ],
    "identified_atoms": [
      {"atom": "p", "definition": "string - proposición que representa"},
      {"atom": "q", "definition": "string - proposición que representa"},
      {"atom": "r", "definition": "string - proposición que representa"}
      {"atom": "s", "definition": "string - proposición que representa"},
      {"atom": "t", "definition": "string - proposición que representa"},
      {"atom": "u", "definition": "string - proposición que representa"}
      ... (los átomos deben ser letras minúsculas y su definición debe ser clara y técnica) ...
    ],
    "identified_connectors": [
      {"connector": "→", "natural_language_cue": "string - palabra clave"}
      ... (usa los conectores lógicos correctos: ¬, ∧, ∨, →, ↔ y muestra la palabra o frase en el texto original que indica cada conector) ...
    ]
  },
  "output": {
    "formula": "string - fórmula con símbolos Unicode (¬, ∧, ∨, →, ↔)",
    "formula_ascii": "string - fórmula ASCII (~, &, |, ->, <->)"
  }
}"""

# =============================================
# REGLAS COMPARTIDAS (se inyectan en cada system prompt)
# =============================================

SHARED_RULES = """
REGLAS ESTRICTAS:
1. Los enunciados deben sonar NATURALES, como los diría una persona real en ese contexto.
2. Usa correctamente los conectores lógicos:
   - "y", "además", "siempre que ambos" → ∧ (conjunción)
   - "o", "ya sea", "cualquiera de" → ∨ (disyunción)
   - "si...entonces", "implica", "cuando", "siempre que" → → (implicación)
   - "si y solo si", "equivale a", "únicamente cuando" → ↔ (bicondicional)
   - "no", "no es cierto que", "falla", "no está" → ¬ (negación)
3. Los átomos deben ser letras minúsculas (p, q, r, s, t, u...).
4. Las fórmulas deben usar paréntesis para desambiguar precedencia.
5. El razonamiento (thought) debe ser detallado paso a paso.
6. Proporciona fórmula Unicode (∧, ∨, →, ↔, ¬) Y ASCII (&, |, ->, <->, ~).
7. Enunciados en español, con términos técnicos en inglés donde sea natural.
8. NO generes enunciados genéricos ni aburridos. Cada uno debe ser ÚNICO y realista.
9. VARÍA la estructura: no siempre empieces con "Si...". Usa "Cuando...", "Siempre que...",
   "Para que...", "Es necesario que...", "No es posible que...", etc.

IMPORTANTE: Responde ÚNICAMENTE con JSON válido, sin markdown, sin ```json, sin explicaciones extra."""

# =============================================
# BLOQUES TEMÁTICOS
# =============================================

BLOCKS = [
    # --- BLOQUE 1: Cybersec & Hacking ---
    {
        "name": "🔓 Cybersec & Hacking",
        "system_prompt": f"""Eres un experto en lógica proposicional especializado en ciberseguridad y hacking ético.

Tu tarea es generar ejemplos que transformen enunciados de seguridad informática en fórmulas de lógica proposicional.

DOMINIOS:
- Reglas de firewall, WAF, ACLs, filtrado de paquetes
- Pentesting: SQLi, XSS, RCE, SSRF, LFI, IDOR
- Escalación de privilegios: SUID, capabilities, kernel exploits
- CTF challenges: crypto, reversing, pwn, web exploitation
- Análisis de malware y condiciones de ejecución de payloads
- Game hacking: memory manipulation, packet tampering, anti-cheat bypass

TONO: Jerga de hacker/pentester. Usa términos como "explotar", "bypassear", "pivotear",
"rootear", "exfiltrar", "dumpear", "shellcode", "payload", "reverse shell", etc.
{SHARED_RULES}""",
        "topics": [
            "reglas de firewall y filtrado de paquetes (iptables, nftables, WAF, ACLs)",
            "pentesting web (SQLi, XSS reflejado/stored, RCE, SSRF, IDOR, path traversal)",
            "escalación de privilegios en Linux (SUID, capabilities, cron, kernel exploits)",
            "CTF challenges de crypto (RSA, AES, hashing, padding oracle)",
            "CTF challenges de reversing (binary analysis, patching, anti-debug)",
            "CTF challenges de pwn (buffer overflow, ROP, heap exploitation, format string)",
            "análisis de malware (condiciones de ejecución, sandbox evasion, C2 callbacks)",
            "game hacking (speedhack, wallhack, aimbot, memory manipulation)",
            "movimiento lateral y post-explotación (pivoting, pass-the-hash, mimikatz)",
            "OSINT y reconocimiento (subdomain enumeration, port scanning, fingerprinting)",
            "ingeniería social y phishing (pretexting, spear phishing, watering hole)",
            "exploit de APIs (broken auth, mass assignment, rate limiting bypass)",
            "wireless hacking (WPA2 cracking, evil twin, deauth attacks)",
            "red team operations (initial access, persistence, defense evasion)",
            "bug bounty (scope rules, report conditions, severity classification)",
        ],
    },
    # --- BLOQUE 2: Programación ---
    {
        "name": "💻 Programación",
        "system_prompt": f"""Eres un experto en lógica proposicional especializado en desarrollo de software.

Tu tarea es generar ejemplos que transformen enunciados de programación y desarrollo en fórmulas de lógica proposicional.

DOMINIOS:
- Condiciones en código: if/else, guards, pattern matching, ternarios
- Lógica de negocio: e-commerce, pagos, inventario, suscripciones
- Validaciones: formularios, APIs, schemas, tipos de datos
- Testing: precondiciones, postcondiciones, assertions, edge cases
- Error handling: try/catch, fallbacks, circuit breakers, retry logic
- Arquitectura: microservicios, eventos, colas, cache invalidation

TONO: Jerga de desarrollador. Usa términos como "deployar", "mergear", "commitear",
"refactorizar", "debuggear", "el build falla", "pasa los tests", "está en staging", etc.
{SHARED_RULES}""",
        "topics": [
            "validación de formularios web (email, password strength, campos requeridos)",
            "lógica de e-commerce (descuentos, cupones, stock, carrito, checkout)",
            "flujo de pagos (tarjeta válida, fondos suficientes, 3DS, refunds)",
            "sistema de suscripciones (trial, upgrade, downgrade, cancelación, grace period)",
            "condiciones de if/else complejas en código real (guards, early returns)",
            "error handling y excepciones (try/catch, fallback, retry con backoff)",
            "testing y assertions (precondiciones, postcondiciones, invariantes)",
            "lógica de permisos en apps (roles, scopes, feature flags, A/B testing)",
            "cache invalidation (TTL, dirty flags, write-through vs write-back)",
            "event-driven architecture (pub/sub, dead letter queues, idempotencia)",
            "rate limiting y throttling (token bucket, sliding window, circuit breaker)",
            "migrations y schema changes (backward compatible, rollback conditions)",
            "feature flags y rollout gradual (canary, percentage, user targeting)",
            "concurrencia y race conditions (locks, deadlocks, optimistic locking)",
            "API contracts (request validation, response codes, versioning)",
        ],
    },
    # --- BLOQUE 3: DevOps & Vida Tech ---
    {
        "name": "🚀 DevOps & Vida Tech",
        "system_prompt": f"""Eres un experto en lógica proposicional especializado en DevOps, sysadmin y tecnología cotidiana.

Tu tarea es generar ejemplos que transformen enunciados de infraestructura, DevOps y apps cotidianas en fórmulas de lógica proposicional.

DOMINIOS:
- CI/CD: pipelines, condiciones de deploy, rollbacks, health checks
- Sysadmin: permisos Unix, configuración de servicios, monitoreo, logs
- Redes: VLANs, subnets, VPN, DNS, load balancing
- Cloud: AWS/GCP/Azure, auto-scaling, IAM, billing alerts
- Apps cotidianas: Uber, delivery, streaming, redes sociales
- Smart home / IoT: sensores, automatización, condiciones de activación

TONO: Mezcla de sysadmin y usuario tech. Desde "si el pod crashea y no hay réplicas"
hasta "si te quedas sin datos y no hay WiFi, no puedes ver el stream".
{SHARED_RULES}""",
        "topics": [
            "pipelines CI/CD (GitHub Actions, Jenkins, GitLab CI, condiciones de stage)",
            "condiciones de deploy (branch protection, approvals, tests passing, staging OK)",
            "rollback y recovery (health checks fallidos, error rate, auto-rollback)",
            "Kubernetes (pod scheduling, readiness/liveness probes, HPA, resource limits)",
            "permisos Unix y control de acceso (chmod, chown, sudo, sudoers, SELinux)",
            "monitoreo y alertas (Prometheus, Grafana, PagerDuty, thresholds, escalation)",
            "configuración de redes (VLANs, subnets, VPN, firewall rules, DNS)",
            "cloud (auto-scaling triggers, spot instances, billing alerts, IAM policies)",
            "apps de transporte (Uber/Lyft: surge pricing, driver matching, ETA)",
            "apps de delivery (disponibilidad, radio de entrega, mínimo de pedido, propinas)",
            "streaming (Netflix/Spotify: plan, dispositivos, contenido regional, offline)",
            "redes sociales (moderación, algoritmo de feed, verificación, shadowban)",
            "smart home (sensores de movimiento, temperatura, horarios, escenas)",
            "gaming online (matchmaking, ping, servers, ranked conditions, bans)",
            "backup y disaster recovery (RPO, RTO, snapshots, geo-replication)",
        ],
    },
    # --- BLOQUE 4: Lógica Pura & Académica ---
    {
        "name": "🎓 Lógica Pura & Académica",
        "system_prompt": f"""Eres un experto en lógica proposicional con formación en filosofía, matemáticas y ciencias.

Tu tarea es generar ejemplos que transformen enunciados lógicos, científicos y cotidianos en fórmulas de lógica proposicional.

DOMINIOS:
- Silogismos y razonamiento clásico
- Puzzles lógicos (sombreros, puertas, mentirosos)
- Proposiciones matemáticas y científicas
- Reglas legales y regulaciones
- Condiciones médicas y diagnósticos
- Razonamiento cotidiano (decisiones, planes, condiciones)

TONO: Más formal y preciso. Mezcla de académico con ejemplos prácticos.
Los enunciados deben ser claros y bien construidos gramaticalmente.
{SHARED_RULES}""",
        "topics": [
            "silogismos clásicos (Sócrates, mortales, categorías, universales)",
            "puzzles lógicos (el acertijo de los sombreros, las puertas, los mentirosos)",
            "paradojas lógicas (el barbero, el mentiroso, Russell, Curry)",
            "proposiciones matemáticas (divisibilidad, paridad, primos, desigualdades)",
            "lógica de conjuntos (pertenencia, subconjuntos, intersección, unión)",
            "razonamiento científico (hipótesis, experimentos, control, variables)",
            "diagnóstico médico (síntomas, condiciones, tratamientos, contraindicaciones)",
            "reglas legales (condiciones de contrato, excepciones, cláusulas, jurisdicción)",
            "decisiones financieras (inversión, riesgo, diversificación, liquidez)",
            "planificación de viajes (vuelos, conexiones, visas, seguros, equipaje)",
            "reglas deportivas (faltas, offside, tarjetas, descalificación, desempate)",
            "lógica de votación (mayorías, quórum, veto, empate, segunda vuelta)",
            "condiciones climáticas (lluvia, viento, temperatura, alertas, precauciones)",
            "reglas de tránsito (semáforos, prioridad, velocidad, multas, excepciones)",
            "razonamiento ético (dilemas morales, utilitarismo, deontología, consecuencias)",
        ],
    },
    # --- BLOQUE 5: Mix Difícil ---
    {
        "name": "🧠 Mix Difícil",
        "system_prompt": f"""Eres un experto en lógica proposicional que genera ejemplos COMPLEJOS y desafiantes.

Tu tarea es generar ejemplos avanzados que combinen múltiples conectores y átomos en fórmulas no triviales.

REQUISITOS ESPECIALES PARA ESTE BLOQUE:
- MÍNIMO 4 átomos por fórmula, idealmente 5-7
- Usa TODOS los conectores: ¬, ∧, ∨, →, ↔ (al menos 3 distintos por ejemplo)
- Incluye negaciones dobles, bicondicionales, y paréntesis anidados
- Los enunciados deben ser largos y con múltiples cláusulas
- Incluye tautologías, contradicciones, y equivalencias lógicas cuando sea natural
- Mezcla dominios: cybersec + programación, vida cotidiana + lógica pura, etc.

TONO: Variado. Puede ser técnico o cotidiano, pero la FÓRMULA siempre debe ser compleja.
{SHARED_RULES}""",
        "topics": [
            "condiciones de deploy con múltiples checks de seguridad y rollback automático",
            "reglas de firewall complejas con excepciones, whitelists y condiciones temporales",
            "lógica de negocio de e-commerce con descuentos, impuestos, envío y devoluciones",
            "sistema de alertas con escalación multinivel y condiciones de silenciamiento",
            "flujo de autenticación completo con MFA, sesiones, tokens y revocación",
            "matchmaking de juegos con ranking, ping, región, queue y party restrictions",
            "pipeline CI/CD completo con lint, test, build, staging, approval y prod",
            "diagnóstico de red con múltiples puntos de falla y condiciones de recovery",
            "sistema de recomendaciones con preferencias, historial, popularidad y novedad",
            "reglas de moderación de contenido con reportes, apelaciones y escalación",
            "condiciones de un contrato inteligente (smart contract) con múltiples partes",
            "sistema de backup con RPO, RTO, verificación de integridad y geo-replication",
            "lógica de un compilador (type checking, scope resolution, error recovery)",
            "condiciones electorales (candidatos, votantes, quórum, mayoría, desempate)",
            "tautologías y equivalencias lógicas disfrazadas de lenguaje natural",
        ],
    },
]

COMPLEXITIES = ["simple", "intermediate", "advanced"]

# Bloque 5 usa complejidades más altas
COMPLEXITIES_HARD = ["intermediate", "advanced", "advanced"]


# =============================================
# FUNCIONES
# =============================================


def get_block(example_index: int, block_size: int = 500) -> dict:
    """Determina qué bloque temático usar según el índice del ejemplo."""
    block_idx = min(example_index // block_size, len(BLOCKS) - 1)
    return BLOCKS[block_idx]


def load_progress(output_file: str) -> list[dict]:
    """Carga ejemplos previos del archivo si existe."""
    path = pathlib.Path(output_file)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                data = json.loads(content)
                return data.get("examples", [])
    return []


def save_progress(examples: list[dict], output_file: str):
    """Guarda el progreso actual a disco."""
    dataset = {"examples": examples}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


def extract_json(text: str) -> dict | None:
    """Extrae JSON de la respuesta, limpiando markdown si hay."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def validate_example(example: dict) -> bool:
    """Validación básica de un ejemplo."""
    required = ["natural_language_input", "complexity", "thought", "output"]
    if not all(k in example for k in required):
        return False
    if not isinstance(example.get("thought"), dict):
        return False
    if not isinstance(example.get("output"), dict):
        return False
    if "formula" not in example["output"]:
        return False
    if len(example.get("natural_language_input", "")) < 20:
        return False
    return True


def generate_one(block: dict, topic: str, complexity: str) -> dict | None:
    """Genera un ejemplo usando el system prompt del bloque actual."""
    prompt = f"""Genera exactamente 1 ejemplo de entrenamiento sobre: {topic}.

Complejidad: {complexity}.

El ejemplo debe seguir este schema JSON exacto:
{JSON_SCHEMA}

Responde SOLO con el JSON del ejemplo, nada más."""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": block["system_prompt"]},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        max_tokens=2000,
        timeout=120,
    )

    text = response.choices[0].message.content
    example = extract_json(text)

    if example and validate_example(example):
        # Normalizar complexity
        if example.get("complexity") not in COMPLEXITIES:
            example["complexity"] = complexity
        # Etiquetar el bloque de origen
        example["block"] = block["name"]
        return example
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Genera dataset sintético — 5 bloques temáticos con auto-switch"
    )
    parser.add_argument(
        "--total", type=int, default=2500, help="Total de ejemplos (default: 2500)"
    )
    parser.add_argument(
        "--output", type=str, default="dataset.json", help="Archivo de salida"
    )
    parser.add_argument(
        "--block-size", type=int, default=500, help="Ejemplos por bloque (default: 500)"
    )
    args = parser.parse_args()

    output_file = args.output
    total = args.total
    block_size = args.block_size

    # Cargar progreso previo
    all_examples = load_progress(output_file)
    if all_examples:
        print(f"📂 Cargados {len(all_examples)} ejemplos previos desde {output_file}")

    remaining = total - len(all_examples)
    if remaining <= 0:
        print(f"✅ Ya tienes {len(all_examples)}/{total} ejemplos. Nada que generar.")
        return

    print(f"🚀 Generando {remaining} ejemplos restantes (objetivo: {total})")
    print(f"📁 Guardando en: {output_file}")
    print(f"💰 Costo estimado: ~${remaining * 0.002:.2f} USD")
    print(f"📦 Bloques de {block_size} ejemplos cada uno\n")

    # Mostrar plan de bloques
    for i, block in enumerate(BLOCKS):
        start = i * block_size
        end = min((i + 1) * block_size, total)
        status = (
            "✅"
            if len(all_examples) >= end
            else ("🔄" if len(all_examples) >= start else "⏳")
        )
        print(f"   {status} Bloque {i + 1}: {block['name']} ({start}-{end})")
    print()

    errors = 0
    consecutive_errors = 0
    start_time = time.time()
    current_block_name = None

    # Habilitar colores ANSI en Windows
    if os.name == "nt":
        os.system("")

    BAR_LEN = 30
    # Ancho de terminal para limpiar líneas completas
    try:
        COLS = os.get_terminal_size().columns
    except OSError:
        COLS = 120

    def make_bar(current, total, errs=0, status=""):
        """Genera string de barra de progreso con status opcional."""
        pct = current / total if total > 0 else 0
        filled = int(BAR_LEN * pct)
        bar = "█" * filled + "░" * (BAR_LEN - filled)
        elapsed = time.time() - start_time
        if current > 0 and elapsed > 0:
            eta = (elapsed / current) * (total - current)
            eta_str = f"{eta / 60:.0f}m" if eta > 60 else f"{eta:.0f}s"
        else:
            eta_str = "?"
        cost = current * 0.002
        line = f"  [{bar}] {pct:>6.1%}  ({current}/{total})  ⏱️ ETA: {eta_str}  💰 ~${cost:.2f}  ❌ {errs} err"
        if status:
            line += f"  | {status}"
        return line[: COLS - 1]  # truncar a ancho de terminal

    def show_bar(current, total, errs=0, status=""):
        """Imprime la barra como última línea (se sobreescribe con \\r)."""
        bar = make_bar(current, total, errs=errs, status=status)
        # Limpiar línea completa y escribir la barra
        print(f"\r{' ' * (COLS - 1)}\r{bar}", end="", flush=True)

    def log(msg):
        """Borra la barra, imprime el mensaje, reimprime la barra."""
        print(f"\r{' ' * (COLS - 1)}\r", end="", flush=True)
        print(msg, flush=True)
        show_bar(len(all_examples), total, errs=errors)

    # Mostrar barra inicial
    show_bar(len(all_examples), total, errs=0)

    for i in range(remaining):
        if len(all_examples) >= total:
            break

        idx = len(all_examples)
        block = get_block(idx, block_size)

        # Anunciar cambio de bloque
        if block["name"] != current_block_name:
            current_block_name = block["name"]
            log(f"\n{'=' * 60}")
            log(f"📦 BLOQUE: {block['name']}")
            log(
                f"   Ejemplos {(idx // block_size) * block_size}-{min(((idx // block_size) + 1) * block_size, total)}"
            )
            log(f"{'=' * 60}\n")

        topic = random.choice(block["topics"])
        # Bloque 5 (Mix Difícil) usa complejidades más altas
        complexities = COMPLEXITIES_HARD if "Mix" in block["name"] else COMPLEXITIES
        complexity = random.choice(complexities)

        # Mostrar barra con el status de lo que se está generando
        show_bar(
            len(all_examples),
            total,
            errs=errors,
            status=f"🔄 {topic[:35]}... ({complexity})",
        )

        try:
            example = generate_one(block, topic, complexity)
            if example:
                all_examples.append(example)
                save_progress(all_examples, output_file)
                log(f"✅ [{idx + 1}/{total}] {topic[:45]}... ({complexity})")
                consecutive_errors = 0
            else:
                errors += 1
                consecutive_errors += 1
                log(f"⚠️ [{idx + 1}/{total}] JSON inválido ({consecutive_errors}/10)")

        except Exception as e:
            errors += 1
            consecutive_errors += 1
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate limit" in err_str.lower()
            if is_rate_limit:
                wait = 10 * consecutive_errors
                log(f"⏳ Rate limit, esperando {wait}s...")
                time.sleep(wait)
            else:
                log(f"❌ ({consecutive_errors}/10) {err_str[:80]}")

        if consecutive_errors >= 10:
            log("\n⚠️ 10 errores consecutivos — reiniciando en 30s...")
            time.sleep(30)
            return False  # Señal de que debe reiniciar

        # Liberar memoria cada 20 ejemplos
        if (i + 1) % 20 == 0:
            gc.collect()

        # Pausa entre llamadas
        time.sleep(1)

    # Limpiar la barra antes del resumen
    print(f"\r{' ' * 100}\r", end="", flush=True)

    # Resumen final
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"🎉 Generación completada!")
    print(f"📊 Total: {len(all_examples)} ejemplos")
    print(f"❌ Errores: {errors}")
    print(f"⏱️  Tiempo: {elapsed / 60:.1f} minutos")
    print(f"📁 Guardado en: {output_file}")

    # Estadísticas
    from collections import Counter

    print(f"\n📈 Distribución por complejidad:")
    complexities = Counter(ex.get("complexity", "?") for ex in all_examples)
    for comp, count in complexities.most_common():
        print(f"   {comp}: {count}")

    print(f"\n📦 Distribución por bloque:")
    blocks = Counter(ex.get("block", "?") for ex in all_examples)
    for blk, count in blocks.most_common():
        print(f"   {blk}: {count}")

    if all_examples:
        sample = random.choice(all_examples)
        print(f"\n📝 Ejemplo aleatorio ({sample.get('block', '?')}):")
        print(f"   Input: {sample['natural_language_input']}")
        print(f"   Fórmula: {sample['output']['formula']}")
        print(f"   ASCII:   {sample['output']['formula_ascii']}")

    return True  # Terminó limpio


if __name__ == "__main__":
    restart_count = 0
    while True:
        finished = main()
        if finished:
            break
        restart_count += 1
        print(
            f"\n🔁 Auto-reinicio #{restart_count} — recargando progreso desde disco...\n"
        )
        gc.collect()
