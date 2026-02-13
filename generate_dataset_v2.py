"""
Generador de dataset sintético v2 - Usa OpenAI SDK directo con DeepSeek.
Sin tool calling, sin Pydantic AI. JSON en prompt → parseo manual.

Uso:
    python generate_dataset_v2.py
    python generate_dataset_v2.py --total 500
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

# --- Cliente DeepSeek (compatible con OpenAI SDK) ---
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# =============================================
# SCHEMA DE EJEMPLO (como string para el prompt)
# =============================================

JSON_SCHEMA = """{
  "natural_language_input": "string - enunciado técnico en español",
  "complexity": "simple | intermediate | advanced",
  "thought": {
    "reasoning_steps": [
      {"step": 1, "explanation": "string - paso de razonamiento"}
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
    ]
  },
  "output": {
    "formula": "string - fórmula con símbolos Unicode (¬, ∧, ∨, →, ↔)",
    "formula_ascii": "string - fórmula ASCII (~, &, |, ->, <->)"
  }
}"""

# =============================================
# CONFIGURACIÓN
# =============================================

SYSTEM_PROMPT = """Eres un experto en lógica proposicional especializado en ciberseguridad, desarrollo de software y hacking ético.

Tu tarea es generar ejemplos de entrenamiento que transformen enunciados técnicos en lenguaje natural
a fórmulas de lógica proposicional.

DOMINIOS TEMÁTICOS (varía entre estos):
- 🔓 Ciberseguridad: reglas de firewall, detección de intrusos, análisis de vulnerabilidades, políticas de acceso
- 🐛 Pentesting/CTF: condiciones de exploit, escalación de privilegios, movimiento lateral, exfiltración
- 💻 Programación: validaciones, flujos de control, condiciones de error, lógica de negocio
- 🖥️ Sysadmin: reglas de red, permisos Unix, configuración de servicios, monitoreo
- 🚀 DevOps/CI-CD: pipelines, condiciones de deploy, rollbacks, health checks
- 🎮 Game hacking: manipulación de memoria, bypass de anticheat, condiciones de win/lose

REGLAS:
1. Los enunciados deben sonar como los diría un dev/hacker real, con jerga técnica natural.
   Ejemplo: "Si el puerto 443 está abierto y el certificado SSL ha expirado, entonces el servidor es vulnerable a MITM"
2. Usa correctamente los conectores lógicos:
   - "y", "además", "siempre que ambos" → ∧ (conjunción)
   - "o", "ya sea", "cualquiera de" → ∨ (disyunción)
   - "si...entonces", "implica", "cuando", "siempre que" → → (implicación)
   - "si y solo si", "equivale a", "únicamente cuando" → ↔ (bicondicional)
   - "no", "no es cierto que", "falla", "no está" → ¬ (negación)
3. Los átomos deben ser letras minúsculas (p, q, r, s, t...).
4. Las fórmulas deben usar paréntesis para desambiguar precedencia.
5. Genera una mezcla de complejidades: simple, intermediate y advanced.
6. El razonamiento (thought) debe ser detallado paso a paso, explicando la lógica técnica.
7. Proporciona tanto la fórmula con símbolos Unicode (∧, ∨, →, ↔, ¬) como en ASCII (&, |, ->, <->, ~).
8. Genera los enunciados en español, pero permite términos técnicos en inglés cuando sea natural
   (ej: "firewall", "buffer overflow", "SQL injection", "deploy", "rollback").
9. NO generes enunciados genéricos aburridos. Cada ejemplo debe sentirse como algo que un profesional diría en su día a día.

IMPORTANTE: Responde ÚNICAMENTE con JSON válido, sin markdown, sin ```json, sin explicaciones extra."""

TOPICS = [
    "reglas de firewall y filtrado de paquetes (iptables, WAF, ACLs)",
    "pentesting y explotación de vulnerabilidades (SQLi, XSS, RCE, SSRF)",
    "escalación de privilegios en Linux (SUID, capabilities, kernel exploits)",
    "CTF challenges (crypto, reversing, pwn, web)",
    "validaciones y sanitización de input en APIs REST",
    "flujos de autenticación y autorización (OAuth, JWT, RBAC)",
    "configuración de redes y segmentación (VLANs, subnets, VPN)",
    "pipelines CI/CD y condiciones de deploy (GitHub Actions, Jenkins)",
    "monitoreo y alertas de seguridad (SIEM, IDS/IPS, logs)",
    "game hacking y anti-cheat (memory manipulation, packet tampering)",
    "hardening de servidores y buenas prácticas sysadmin",
    "análisis de malware y condiciones de ejecución de payloads",
    "lógica de negocio en aplicaciones web (e-commerce, banking)",
    "permisos Unix y control de acceso (chmod, chown, sudo, SELinux)",
    "condiciones de error handling y excepciones en código",
]

COMPLEXITIES = ["simple", "intermediate", "advanced"]


# =============================================
# FUNCIONES
# =============================================


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
    # Quitar bloques de código markdown
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Intentar encontrar JSON dentro del texto
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
    return True


def generate_one(topic: str, complexity: str) -> dict | None:
    """Genera un ejemplo llamando a DeepSeek directamente."""
    prompt = f"""Genera exactamente 1 ejemplo de entrenamiento sobre: {topic}.

Complejidad: {complexity}.

El ejemplo debe seguir este schema JSON exacto:
{JSON_SCHEMA}

Responde SOLO con el JSON del ejemplo, nada más."""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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
            return example
        else:
            return None

    except Exception as e:
        raise e


def main():
    parser = argparse.ArgumentParser(
        description="Genera dataset sintético de lógica proposicional (v2 - directo)"
    )
    parser.add_argument(
        "--total", type=int, default=50, help="Total de ejemplos a generar"
    )
    parser.add_argument(
        "--output", type=str, default="dataset.json", help="Archivo de salida"
    )
    args = parser.parse_args()

    output_file = args.output
    total = args.total

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
    print(f"💰 Costo estimado: ~${remaining * 0.002:.2f} USD\n")

    errors = 0
    consecutive_errors = 0
    start_time = time.time()

    for i in range(remaining):
        if len(all_examples) >= total:
            break

        topic = random.choice(TOPICS)
        complexity = random.choice(COMPLEXITIES)

        print(
            f"🔄 [{len(all_examples) + 1}/{total}] {topic[:55]}... ({complexity})",
            end=" ",
            flush=True,
        )

        try:
            example = generate_one(topic, complexity)
            if example:
                all_examples.append(example)
                save_progress(all_examples, output_file)
                print("✅")
                consecutive_errors = 0
            else:
                errors += 1
                consecutive_errors += 1
                print(f"⚠️ JSON inválido ({consecutive_errors}/10)")

        except Exception as e:
            errors += 1
            consecutive_errors += 1
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate limit" in err_str.lower()
            if is_rate_limit:
                wait = 10 * consecutive_errors
                print(f"⏳ Rate limit, esperando {wait}s...")
                time.sleep(wait)
            else:
                print(f"❌ ({consecutive_errors}/10) {err_str[:80]}")

        if consecutive_errors >= 10:
            print("\n⚠️ 10 errores consecutivos, deteniendo.")
            break

        # Liberar memoria cada 20 ejemplos
        if (i + 1) % 20 == 0:
            gc.collect()

        # Pausa mínima entre llamadas
        time.sleep(1)

    # Resumen final
    elapsed = time.time() - start_time
    generated = len(all_examples) - (total - remaining - errors)
    print(f"\n{'=' * 60}")
    print(f"🎉 Generación completada!")
    print(f"📊 Total: {len(all_examples)} ejemplos")
    print(f"❌ Errores: {errors}")
    print(f"⏱️  Tiempo: {elapsed / 60:.1f} minutos")
    print(f"📁 Guardado en: {output_file}")

    # Estadísticas
    from collections import Counter

    complexities = Counter(ex.get("complexity", "?") for ex in all_examples)
    print(f"\n📈 Distribución:")
    for comp, count in complexities.most_common():
        print(f"   {comp}: {count}")

    if all_examples:
        sample = random.choice(all_examples)
        print(f"\n📝 Ejemplo aleatorio:")
        print(f"   Input: {sample['natural_language_input']}")
        print(f"   Fórmula: {sample['output']['formula']}")
        print(f"   ASCII: {sample['output']['formula_ascii']}")


if __name__ == "__main__":
    main()
