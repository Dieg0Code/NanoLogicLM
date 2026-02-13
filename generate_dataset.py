"""
Script standalone para generar el dataset sintético de entrenamiento.
Corre fuera del notebook para evitar OOM del kernel de Jupyter.

Uso:
    python generate_dataset.py
    python generate_dataset.py --total 100 --batch-size 3
"""

import asyncio
import json
import os
import random
import pathlib
import argparse
import gc
import time
from enum import Enum

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent

# --- Cargar .env ---
load_dotenv()
assert os.getenv("DEEPSEEK_API_KEY"), "❌ DEEPSEEK_API_KEY no encontrada en .env"


# =============================================
# MODELOS PYDANTIC (mismos del notebook)
# =============================================


class LogicalConnector(str, Enum):
    NEGATION = "¬"
    CONJUNCTION = "∧"
    DISJUNCTION = "∨"
    IMPLICATION = "→"
    BICONDITIONAL = "↔"


class Complexity(str, Enum):
    SIMPLE = "simple"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ReasoningStep(BaseModel):
    step: int = Field(description="Número secuencial del paso de razonamiento.")
    explanation: str = Field(
        description="Descripción técnica de lo que se está analizando en este punto del razonamiento."
    )


class AtomDefinition(BaseModel):
    atom: str = Field(
        description="La letra que representa la proposición (ejemplo: p, q, r...).",
        pattern=r"^[a-z][0-9]?$",
    )
    definition: str = Field(
        description="La proposición simple en lenguaje natural que representa el átomo."
    )


class ConnectorUsage(BaseModel):
    connector: str = Field(
        description="El conector lógico identificado (¬, ∧, ∨, →, ↔)."
    )
    natural_language_cue: str = Field(
        description="La palabra o frase en el texto original que indica este conector."
    )


class ThoughtBlock(BaseModel):
    reasoning_steps: list[ReasoningStep] = Field(
        description="Secuencia de pasos de razonamiento."
    )
    identified_atoms: list[AtomDefinition] = Field(
        description="Lista de átomos proposicionales identificados."
    )
    identified_connectors: list[ConnectorUsage] = Field(
        description="Lista de conectores lógicos identificados."
    )


class PropositionalFormula(BaseModel):
    formula: str = Field(
        description="La fórmula en lógica proposicional usando símbolos estándar."
    )
    formula_ascii: str = Field(description="La misma fórmula usando notación ASCII.")


class TrainingExample(BaseModel):
    natural_language_input: str = Field(
        description="El enunciado en lenguaje natural que se debe formalizar."
    )
    complexity: Complexity = Field(description="Nivel de complejidad del ejemplo.")
    thought: ThoughtBlock = Field(
        description="Bloque de pensamiento con el razonamiento paso a paso."
    )
    output: PropositionalFormula = Field(
        description="La fórmula proposicional resultante."
    )


class SyntheticDataset(BaseModel):
    examples: list[TrainingExample] = Field(
        description="Lista de ejemplos de entrenamiento."
    )


# =============================================
# CONFIGURACIÓN DEL AGENTE
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
"""

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

EXAMPLES_PER_BATCH = 1

COMPLEXITY_MIXES = [
    "simple",
    "intermediate",
    "advanced",
]


# =============================================
# FUNCIONES DE GENERACIÓN
# =============================================


def load_progress(output_file: str) -> list[TrainingExample]:
    """Carga ejemplos previos del archivo si existe."""
    path = pathlib.Path(output_file)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                dataset = SyntheticDataset.model_validate_json(content)
                return list(dataset.examples)
    return []


def save_progress(examples: list[TrainingExample], output_file: str):
    """Guarda el progreso actual a disco."""
    dataset = SyntheticDataset(examples=examples)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(dataset.model_dump_json(indent=2))


async def generate_one(
    agent: Agent, topic: str, complexity: str
) -> list[TrainingExample]:
    """Genera un ejemplo de entrenamiento con timeout."""
    prompt = f"""Genera exactamente {EXAMPLES_PER_BATCH} ejemplo de entrenamiento sobre: {topic}.

Complejidad: {complexity}.

El ejemplo debe tener:
- Un enunciado que suene como algo que diría un dev o hacker en su día a día
- Razonamiento detallado paso a paso con contexto técnico
- Identificación correcta de átomos y conectores
- La fórmula proposicional correcta en Unicode y ASCII

¡Sé creativo y técnicamente preciso! Usa jerga real del campo."""

    result = await asyncio.wait_for(agent.run(prompt), timeout=180)
    return result.output.examples


async def main():
    parser = argparse.ArgumentParser(
        description="Genera dataset sintético de lógica proposicional"
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

    # Crear agente con DeepSeek V3 (~$0.002/ejemplo)
    agent = Agent(
        "deepseek:deepseek-chat",
        output_type=SyntheticDataset,
        system_prompt=SYSTEM_PROMPT,
        retries=5,
    )

    # Cargar progreso previo
    all_examples = load_progress(output_file)
    if all_examples:
        print(f"📂 Cargados {len(all_examples)} ejemplos previos desde {output_file}")

    remaining = total - len(all_examples)
    if remaining <= 0:
        print(f"✅ Ya tienes {len(all_examples)}/{total} ejemplos. Nada que generar.")
        return

    batches_needed = (remaining + EXAMPLES_PER_BATCH - 1) // EXAMPLES_PER_BATCH
    print(
        f"🚀 Generando ~{remaining} ejemplos restantes en ~{batches_needed} batches de {EXAMPLES_PER_BATCH}"
    )
    print(f"📁 Guardando en: {output_file}\n")

    errors = 0
    consecutive_errors = 0
    for i in range(batches_needed):
        if len(all_examples) >= total:
            break
        topic = random.choice(TOPICS)
        complexity = random.choice(COMPLEXITY_MIXES)

        print(
            f"🔄 Batch {i + 1}/{batches_needed} [{len(all_examples)}/{total}] {topic[:50]}... ({complexity})",
            end=" ",
            flush=True,
        )

        try:
            examples = await generate_one(agent, topic, complexity)
            all_examples.extend(examples)
            save_progress(all_examples, output_file)
            print(f"✅")
            consecutive_errors = 0

            # Liberar memoria cada 10 ejemplos
            if (i + 1) % 10 == 0:
                gc.collect()

            # Pausa entre llamadas para respetar rate limit (30 req/min)
            await asyncio.sleep(2.5)

        except Exception as e:
            errors += 1
            consecutive_errors += 1
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate limit" in err_str.lower()
            if is_rate_limit:
                wait = 15 * consecutive_errors
                print(f"⏳ Rate limit, esperando {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"❌ ({consecutive_errors}/5) {err_str[:100]}")
            if consecutive_errors >= 5:
                print("\n⚠️ 5 errores consecutivos, deteniendo.")
                break
            continue

    # Resumen final
    print(f"\n{'=' * 60}")
    print(f"🎉 Generación completada!")
    print(f"📊 Total: {len(all_examples)} ejemplos")
    print(f"❌ Errores: {errors}")
    print(f"📁 Guardado en: {output_file}")

    # Estadísticas
    from collections import Counter

    complexities = Counter(ex.complexity.value for ex in all_examples)
    print(f"\n📈 Distribución:")
    for comp, count in complexities.most_common():
        print(f"   {comp}: {count}")

    if all_examples:
        sample = random.choice(all_examples)
        print(f"\n📝 Ejemplo aleatorio:")
        print(f"   Input: {sample.natural_language_input}")
        print(f"   Fórmula: {sample.output.formula}")
        print(f"   ASCII: {sample.output.formula_ascii}")


if __name__ == "__main__":
    asyncio.run(main())
