"""Prueba del agente: razonamiento en dos pasos y memoria por thread_id.

Cada corrida recrea la base SQLite para que la traza sea reproducible.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from graph import RECURSION_LIMIT, build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("agente")

THREAD_ID = "sesion-cliente-102"
DB_PATH = Path("data/checkpoints.sqlite")
TRACE_PATH = Path("traces/ejecucion.json")

TURNOS = [
    (
        "multi_paso",
        "¿Cuántos pedidos tuvo el cliente 102, cuál fue el total "
        "y qué productos incluye el último pedido?",
    ),
    (
        "memoria",
        "¿Y el penúltimo, qué productos tenía?",
    ),
    (
        "error",
        "¿Cuántos pedidos tuvo el cliente 999 y cuál fue el total?",
    ),
]


def message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


def dump_message(message: BaseMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": message.type,
        "content": message_text(message),
    }
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        payload["tool_calls"] = [
            {
                "name": call.get("name"),
                "args": call.get("args"),
                "id": call.get("id"),
            }
            for call in tool_calls
        ]
    name = getattr(message, "name", None)
    if name:
        payload["name"] = name
    return payload


def print_cycle(usuario: str, steps: list[dict[str, Any]], respuesta: str) -> None:
    print(f'\nUsuario: "{usuario}"')
    for step in steps:
        calls = step.get("tool_calls") or []
        if calls:
            for call in calls:
                args = ", ".join(f"{key}={value}" for key, value in (call["args"] or {}).items())
                print(f"→ El agente decide usar la herramienta: {call['name']}({args})")
        elif step["type"] == "tool":
            print(f"→ {step.get('name')} devuelve: {step['content']}")
    print(f'Respuesta: "{respuesta}"')


async def run_turn(graph: Any, question: str) -> dict[str, Any]:
    config = {
        "configurable": {"thread_id": THREAD_ID},
        "recursion_limit": RECURSION_LIMIT,
    }
    # SqliteSaver expone get_state/invoke síncronos. Se corren en un hilo
    # para no bloquear el event loop del script.
    previous = await asyncio.to_thread(graph.get_state, config)
    previous_count = len(previous.values.get("messages", [])) if previous.values else 0

    result = await asyncio.to_thread(
        graph.invoke,
        {"messages": [HumanMessage(content=question)]},
        config,
    )
    new_messages = result["messages"][previous_count:]
    steps = [dump_message(message) for message in new_messages]
    respuesta = ""
    for step in reversed(steps):
        if step["type"] == "ai" and not step.get("tool_calls") and step["content"]:
            respuesta = step["content"]
            break

    tool_names = [
        call["name"]
        for step in steps
        for call in (step.get("tool_calls") or [])
    ]
    return {
        "usuario": question,
        "pasos": steps,
        "respuesta": respuesta,
        "herramientas": tool_names,
    }


def assert_trace(turnos: list[dict[str, Any]]) -> None:
    multi, memoria, error = turnos
    if multi["herramientas"].count("buscar_pedidos") < 1:
        raise SystemExit("El turno multi-paso no llamó a buscar_pedidos.")
    if multi["herramientas"].count("detalle_pedido") < 1:
        raise SystemExit(
            "El turno multi-paso no llamó a detalle_pedido. "
            "La conclusión tiene que salir de al menos dos invocaciones."
        )
    plano = multi["respuesta"].lower().replace(".", "").replace(",", "").replace("$", "")
    tiene_cantidad = "3" in plano or "tres" in multi["respuesta"].lower()
    if not tiene_cantidad or "14500" not in plano:
        raise SystemExit(f"La respuesta multi-paso no resume 3 pedidos / 14500:\n{multi['respuesta']}")
    if "soporte" not in multi["respuesta"].lower() or "mouse" not in multi["respuesta"].lower():
        raise SystemExit(f"Faltan los productos del último pedido:\n{multi['respuesta']}")

    if "102" in memoria["usuario"] or "5008" in memoria["usuario"]:
        raise SystemExit("El turno de memoria no debería repetir el cliente ni el id.")
    if "monitor" not in memoria["respuesta"].lower():
        raise SystemExit(
            "Con el mismo thread_id el agente no recordó el penúltimo pedido:\n"
            f"{memoria['respuesta']}"
        )

    if "buscar_pedidos" not in error["herramientas"]:
        raise SystemExit("Ante el cliente 999 el agente no consultó la herramienta.")
    inventado = "14500" in error["respuesta"] or "14.500" in error["respuesta"]
    if inventado:
        raise SystemExit(f"El agente inventó un total para el cliente 999:\n{error['respuesta']}")


async def main() -> None:
    load_dotenv()
    if DB_PATH.exists():
        DB_PATH.unlink()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # SqliteSaver no implementa aget/aput. El script es async y corre
    # cada invoke en un hilo para no bloquear el event loop.
    with SqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        checkpointer.setup()
        graph = build_graph(checkpointer)
        turnos: list[dict[str, Any]] = []
        for nombre, pregunta in TURNOS:
            logger.info("Turno %s | thread_id=%s", nombre, THREAD_ID)
            turno = await run_turn(graph, pregunta)
            turno["nombre"] = nombre
            turnos.append(turno)
            print_cycle(pregunta, turno["pasos"], turno["respuesta"])

    traza = {
        "thread_id": THREAD_ID,
        "recursion_limit": RECURSION_LIMIT,
        "checkpointer": "SqliteSaver",
        "turnos": [
            {
                "nombre": turno["nombre"],
                "usuario": turno["usuario"],
                "herramientas_invocadas": turno["herramientas"],
                "pasos": turno["pasos"],
                "respuesta": turno["respuesta"],
            }
            for turno in turnos
        ],
    }
    TRACE_PATH.write_text(
        json.dumps(traza, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Traza guardada en %s", TRACE_PATH)
    assert_trace(turnos)


if __name__ == "__main__":
    asyncio.run(main())
