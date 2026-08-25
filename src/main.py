# ═══════════════════════════════════════════════════════════════
#  PATRÓN 5 — ORCHESTRATOR-WORKERS (jefe y especialistas)
# ═══════════════════════════════════════════════════════════════
#
#                ┌──▶ 📊 analista de mercado ──┐
#   idea ──▶ 👔 ─┼──▶ 🔧 experto técnico     ──┼──▶ 👔 síntesis
#                └──▶ ⚠️ analista de riesgos ──┘
#              (fan-out: en paralelo)      (fan-in: juntar todo)
#
#  Idea clave: los tres especialistas trabajan A LA VEZ
#  (asyncio.gather), cada uno con su propio prompt. Al final,
#  el orquestador junta las tres opiniones en una sola respuesta.
#
#  Ejemplo: evaluar una idea de negocio con un comité de expertos.
# ═══════════════════════════════════════════════════════════════

import os
import asyncio
from typing import Dict, Tuple, TypedDict

from ollama import AsyncClient
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemma4:31b"

# Configuración

OLLAMA_HOST = os.environ.get('OLLAMA_HOST')
OLLAMA_API_KEY = os.environ.get('OLLAMA_API_KEY')

class ResultadoComite(TypedDict):
    opiniones: Dict[str, str]
    veredicto: str

def make_client() -> AsyncClient:
    """
    Crea un cliente Ollama con soporte para API Key.
    Si no se proporciona OLLAMA_API_KEY funcionará igualmente
    para instancias locales sin autenticación.
    """

    return AsyncClient(
        host=OLLAMA_HOST,
        headers={
            "Authorization": f"Bearer {OLLAMA_API_KEY}"
        },
    )

def paso(icono: str, mensaje: str) -> None:
    print(f"{icono} {mensaje}")

async def consultar_especialista(
    client: AsyncClient,
    rol: str,
    instrucciones: str,
    idea: str,
) -> Tuple[str, str]:

    respuesta = await client.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": instrucciones,
            },
            {
                "role": "user",
                "content": f"Idea de negocio: {idea}",
            },
        ],
    )

    print(f"   ✔ {rol} ha terminado")

    return rol, respuesta["message"]["content"]


async def evaluar_idea(
    idea: str,
    team:dict,
    client: AsyncClient | None = None,
) -> ResultadoComite:

    client = client or make_client()

    paso("🚀", "Fan-out: los especialistas trabajan en paralelo...")

    pares = await asyncio.gather(
        *[
            consultar_especialista(
                client,
                rol,
                instrucciones,
                idea,
            )
            for rol, instrucciones in team.items()
        ]
    )

    opiniones = dict(pares)

    paso("👔", "Fan-in: sintetizando opiniones...")

    contexto = "\n\n".join(
        f"[{rol.upper()}]\n{texto}"
        for rol, texto in opiniones.items()
    )

    sintesis = await client.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres el orquestador del comité. "
                    "Integra las tres opiniones en un único veredicto. "
                    "Indica si merece la pena intentarlo, por qué, "
                    "y menciona desacuerdos si existen."
                ),
            },
            {
                "role": "user",
                "content": contexto,
            },
        ],
    )

    return {
        "opiniones": opiniones,
        "veredicto": sintesis["message"]["content"],
    }

async def main(idea:str, team:dict):

    resultado = await evaluar_idea(
        idea,
        team
    )

    paso("✅", "Veredicto del comité")
    print(resultado["veredicto"])

if __name__ == "__main__":

    asyncio.run(main(
        idea="una app que publique tweets de una estación meteorológica",
        team= {
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
        }
    ))