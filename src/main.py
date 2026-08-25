# ═══════════════════════════════════════════════════════════════
#  PATRÓN — ORCHESTRATOR-WORKERS
# ═══════════════════════════════════════════════════════════════
#
#                         ┌──▶ 👷 Worker A ──┐
#                         │                   │
#   Entrada ──▶ 👔 ───────┼──▶ 👷 Worker B ──┼──▶ 👔 Orchestrator
#                         │                   │
#                         └──▶ 👷 Worker C ──┘
#
#                  FAN-OUT           FAN-IN
#              (trabajo paralelo)  (síntesis)
#
#  Idea clave:
#  Un Orchestrator descompone una tarea en trabajos independientes
#  y los distribuye entre múltiples Workers especializados.
#
#  Los Workers procesan la misma entrada desde perspectivas,
#  capacidades o instrucciones diferentes y devuelven sus resultados.
#
#  Finalmente, el Orchestrator realiza el FAN-IN: recopila, contrasta
#  y sintetiza los resultados para producir una respuesta final.
#
#  Flujo:
#      Entrada → Orchestrator → Workers → Resultados → Orchestrator
#
#  Implementación:
#      FAN-OUT → asyncio.gather(...)
#      FAN-IN  → síntesis de los resultados
# ═══════════════════════════════════════════════════════════════

import os
import asyncio
from typing import TypedDict

from dotenv import load_dotenv
from ollama import AsyncClient

load_dotenv()

MODEL = os.getenv("MODEL")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

class OrchestrationResult(TypedDict):
    workers: dict[str, str]
    result: str


def make_client() -> AsyncClient:
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


async def run_worker(
    client: AsyncClient,
    name: str,
    instructions: str,
    input_data: str,
) -> tuple[str, str]:

    result = await chat(
        client=client,
        system_prompt=instructions,
        user_prompt=input_data,
    )

    print(f"   ✔ Worker '{name}' ha terminado")

    return name, result


async def fan_out(
    client: AsyncClient,
    workers: dict[str, str],
    input_data: str,
) -> dict[str, str]:

    results = await asyncio.gather(
        *[
            run_worker(
                client,
                name,
                instructions,
                input_data,
            )
            for name, instructions in workers.items()
        ]
    )

    return dict(results)


async def fan_in(
    client: AsyncClient,
    results: dict[str, str],
) -> str:

    context = "\n\n".join(
        f"[{name.upper()}]\n{result}"
        for name, result in results.items()
    )

    return await chat(
        client=client,
        system_prompt=(
            "Eres un orquestador. "
            "Integra los resultados proporcionados por los workers. "
            "Identifica coincidencias, discrepancias y conclusiones "
            "relevantes. Produce una respuesta final coherente."
        ),
        user_prompt=context,
    )


async def orchestrate(
    input_data: str,
    workers: dict[str, str],
    client: AsyncClient | None = None,
) -> OrchestrationResult:

    client = client or make_client()

    print("🚀 FAN-OUT: ejecutando workers en paralelo...")

    results = await fan_out(
        client=client,
        workers=workers,
        input_data=input_data,
    )

    print("👔 FAN-IN: sintetizando resultados...")

    result = await fan_in(
        client=client,
        results=results,
    )

    return {
        "workers": results,
        "result": result,
    }

async def main(
    idea:str,
    workers:dict
):

    result = await orchestrate(
        input_data=idea,
        workers=workers,
    )

    print("\n✅ RESULTADO FINAL")
    print(result["result"])

if __name__ == "__main__":

    asyncio.run(main(
        idea=(
            "una empresa qué cree LLMs"
        ),
        workers={
            "mercado": (
                "Analiza la oportunidad de mercado, "
                "los clientes potenciales y la competencia. Se breve."
            ),
            "tecnico": (
                "Analiza la viabilidad técnica, "
                "la arquitectura necesaria y las dificultades. Se breve."
            ),
            "riesgos": (
                "Identifica los principales riesgos, "
                "puntos de fallo y posibles mitigaciones. Se breve."
            ),
        }
    ))