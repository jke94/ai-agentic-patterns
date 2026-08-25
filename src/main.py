"""
═══════════════════════════════════════════════════════════════
 PATRÓN — ORCHESTRATOR-WORKERS
═══════════════════════════════════════════════════════════════

                         ┌──▶ 👷 Worker A ──┐
                         │                   │
 Entrada ──▶ 👔 ────────┼──▶ 👷 Worker B ──┼──▶ 👔 Orchestrator
                         │                   │
                         └──▶ 👷 Worker C ──┘

                  FAN-OUT           FAN-IN
              (trabajo paralelo)  (síntesis)

 Idea clave:
 Un Orchestrator distribuye una tarea entre múltiples Workers
 especializados que trabajan de forma independiente y en paralelo.

 Los Workers reciben la misma entrada, pero cada uno utiliza
 instrucciones diferentes para analizarla desde una perspectiva
 determinada.

 Finalmente, el Orchestrator realiza el FAN-IN: recopila los
 resultados, los contrasta y genera una respuesta final.

 Flujo:
     Entrada → FAN-OUT → Workers → FAN-IN → Resultado

 Implementación:
     FAN-OUT → asyncio.gather(...)
     FAN-IN  → síntesis de los resultados
═══════════════════════════════════════════════════════════════
"""

import os
import asyncio
from typing import TypedDict

from dotenv import load_dotenv
from ollama import AsyncClient


# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════

load_dotenv()

MODEL = os.getenv("MODEL")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")


# ═══════════════════════════════════════════════════════════════
# TIPOS
# ═══════════════════════════════════════════════════════════════

class OrchestrationResult(TypedDict):
    workers: dict[str, str]
    result: str


# ═══════════════════════════════════════════════════════════════
# CLIENTE LLM
# ═══════════════════════════════════════════════════════════════

def make_client() -> AsyncClient:
    """
    Crea el cliente de Ollama utilizando la configuración
    proporcionada mediante variables de entorno.
    """

    return AsyncClient(
        host=OLLAMA_HOST,
        headers={
            "Authorization": f"Bearer {OLLAMA_API_KEY}"
        },
    )


async def chat(
    client: AsyncClient,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """
    Ejecuta una petición al modelo y devuelve únicamente
    el contenido textual de la respuesta.
    """

    response = await client.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    return response["message"]["content"]


# ═══════════════════════════════════════════════════════════════
# WORKER
# ═══════════════════════════════════════════════════════════════

async def run_worker(
    client: AsyncClient,
    name: str,
    instructions: str,
    input_data: str,
) -> tuple[str, str]:
    """
    Ejecuta un Worker.

    Cada Worker recibe:
        - un nombre;
        - sus instrucciones;
        - la entrada común.

    Devuelve:
        (nombre, resultado)
    """

    result = await chat(
        client=client,
        system_prompt=instructions,
        user_prompt=input_data,
    )

    print(f"   ✔ Worker '{name}' ha terminado")

    return name, result


# ═══════════════════════════════════════════════════════════════
# FAN-OUT
# ═══════════════════════════════════════════════════════════════

async def fan_out(
    client: AsyncClient,
    workers: dict[str, str],
    input_data: str,
) -> dict[str, str]:
    """
    Distribuye la entrada entre todos los Workers.

    asyncio.gather() permite ejecutar las llamadas
    concurrentemente en lugar de hacerlo de forma secuencial.
    """

    results = await asyncio.gather(
        *[
            run_worker(
                client=client,
                name=name,
                instructions=instructions,
                input_data=input_data,
            )
            for name, instructions in workers.items()
        ]
    )

    return dict(results)


# ═══════════════════════════════════════════════════════════════
# FAN-IN
# ═══════════════════════════════════════════════════════════════

async def fan_in(
    client: AsyncClient,
    results: dict[str, str],
    instructions: str,
) -> str:
    """
    Recopila los resultados de los Workers y los entrega
    al Orchestrator para generar una respuesta final.

    El Orchestrator debe devolver exclusivamente Markdown.
    """

    context = "\n\n".join(
        f"## Worker: {name}\n\n{result}"
        for name, result in results.items()
    )

    return await chat(
        client=client,
        system_prompt=instructions,
        user_prompt=context,
    )


# ═══════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

async def orchestrate(
    input_data: str,
    workers: dict[str, str],
    orchestrator: str,
    client: AsyncClient | None = None,
) -> OrchestrationResult:
    """
    Implementa el patrón Orchestrator-Workers.

    1. FAN-OUT:
       Distribuye la entrada entre los Workers.

    2. WORKERS:
       Ejecutan sus tareas de forma concurrente.

    3. FAN-IN:
       Recopila los resultados.

    4. ORCHESTRATOR:
       Sintetiza los resultados en una respuesta final.
    """

    client = client or make_client()

    print()
    print("🚀 FAN-OUT")
    print("   Ejecutando workers en paralelo...")
    print()

    results = await fan_out(
        client=client,
        workers=workers,
        input_data=input_data,
    )

    print()
    print("👔 FAN-IN")
    print("   Sintetizando resultados...")
    print()

    result = await fan_in(
        client=client,
        results=results,
        instructions=orchestrator,
    )

    return {
        "workers": results,
        "result": result,
    }


# ═══════════════════════════════════════════════════════════════
# CASO DE USO
# ═══════════════════════════════════════════════════════════════

async def main(
    idea: str,
    team: dict[str, str],
    orchestrator: str,
) -> None:

    result = await orchestrate(
        input_data=idea,
        workers=team,

        orchestrator=orchestrator,
    )

    # ═══════════════════════════════════════════════════════════
    # RESULTADO FINAL
    # ═══════════════════════════════════════════════════════════

    print()
    print("═" * 70)
    print("👔 RESULTADO DEL ORCHESTRATOR")
    print("═" * 70)
    print()

    print(result["result"])

    # ═══════════════════════════════════════════════════════════
    # RESULTADOS INDIVIDUALES
    # ═══════════════════════════════════════════════════════════

    print()
    print("═" * 70)
    print("📋 RESULTADOS INDIVIDUALES DE LOS WORKERS")
    print("═" * 70)

    for name, worker_result in result["workers"].items():

        print()
        print(f"## {name.upper()}")
        print()
        print(worker_result)


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    asyncio.run(main(
        idea="una app que publique tweets de una estación meteorológica",
        team={
            "mercado": (
                "Eres analista de mercado. Di quién compraría esto, "
                "qué competencia existe y cómo destacar. Sé breve."
            ),
            "tecnico": (
                "Eres ingeniero de software senior. Di qué haría falta "
                "para construirlo y cuál es la parte más difícil. Sé breve."
            ),
            "riesgos": (
                "Eres analista de riesgos. Di las dos formas más probables "
                "en que esta idea podría fracasar. Sé breve."
            ),
        },
        orchestrator=(
            "Eres un analista senior. Sintetiza los análisis, "
            "señala oportunidades, viabilidad y riesgos, y da una "
            "recomendación final. Devuelve solo Markdown y sé breve."
        ),
    ))