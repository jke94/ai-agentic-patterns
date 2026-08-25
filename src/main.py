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

async def main() -> None:

    result = await orchestrate(
        input_data=(
            "una empresa que crea modelos de lenguaje "
            "especializados para empresas"
        ),

        workers={
            "mercado": (
                """
Analiza la oportunidad de mercado.

Tu respuesta debe estar escrita exclusivamente en Markdown.

Utiliza exactamente esta estructura:

### Oportunidad de mercado

Explica brevemente la oportunidad.

### Clientes potenciales

Enumera los principales segmentos de clientes.

### Competencia

Identifica los principales tipos de competidores.

### Diferenciación

Explica cómo podría diferenciarse la empresa.

Sé conciso y evita repetir información.
"""
            ),

            "tecnico": (
                """
Analiza la viabilidad técnica del proyecto.

Tu respuesta debe estar escrita exclusivamente en Markdown.

Utiliza exactamente esta estructura:

### Viabilidad técnica

Indica si el proyecto es técnicamente viable y por qué.

### Arquitectura

Enumera los principales componentes técnicos necesarios.

### Recursos

Indica los principales recursos necesarios.

### Dificultades

Enumera las principales dificultades técnicas.

Sé conciso y evita repetir información.
"""
            ),

            "riesgos": (
                """
Analiza los principales riesgos del proyecto.

Tu respuesta debe estar escrita exclusivamente en Markdown.

Utiliza exactamente esta estructura:

### Riesgos principales

Enumera los riesgos más importantes.

### Impacto

Indica brevemente el impacto de cada riesgo.

### Mitigación

Propón una estrategia de mitigación para cada riesgo.

### Puntos críticos

Identifica los posibles puntos únicos de fallo.

Sé conciso y evita repetir información.
"""
            ),
        },

        orchestrator=(
            """
Eres el Orchestrator de un equipo de Workers especializados.

Has recibido varios análisis independientes sobre una misma entrada.

Tu tarea es analizar, contrastar y sintetizar esos resultados.

IMPORTANTE:
- Devuelve exclusivamente Markdown.
- No incluyas introducciones innecesarias.
- No describas el proceso interno de los Workers.
- No digas "como Orchestrator".
- No repitas literalmente los análisis originales.
- Identifica coincidencias y discrepancias.
- Prioriza las conclusiones relevantes.
- Si existe información contradictoria, indícalo.
- Diferencia hechos, recomendaciones y riesgos.
- Sé concreto y evita contenido redundante.

Utiliza exactamente esta estructura:

# Análisis consolidado

## Resumen ejecutivo

Resume en 3-5 frases las conclusiones más importantes.

## Oportunidades

Enumera las principales oportunidades identificadas.

## Viabilidad

Explica las conclusiones relacionadas con la viabilidad.

## Riesgos

Presenta los principales riesgos identificados.

Utiliza una tabla Markdown con esta estructura:

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Riesgo identificado | Alto/Medio/Bajo | Medida propuesta |

## Puntos de consenso

Enumera las conclusiones en las que coinciden los Workers.

## Discrepancias

Indica únicamente las discrepancias relevantes entre los Workers.

## Recomendación

Proporciona una recomendación final clara.

Finaliza con:

**Conclusión:** [conclusión en una o dos frases]
"""
        ),
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
    asyncio.run(main())