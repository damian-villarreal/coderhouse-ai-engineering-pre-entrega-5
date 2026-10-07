"""StateGraph ReAct: nodo del modelo, herramientas y SqliteSaver."""

import logging
import os
from collections.abc import Sequence
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from state import AgentState
from tools import TOOLS

logger = logging.getLogger("agente.graph")

# Techo del ciclo modelo → herramientas → modelo. Evita bucles de costo.
RECURSION_LIMIT = 10

# Si el hilo crece, se conserva solo la cola. El checkpointer sigue
# guardando el historial completo; esto recorta lo que ve el modelo.
MAX_CONTEXT_MESSAGES = 20

SYSTEM_PROMPT = SystemMessage(
    content=(
        "Sos un agente de atención comercial. Respondé en español, "
        "concreto, y solo con datos que hayan devuelto las herramientas.\n"
        "Elegí vos qué herramienta llamar. No hay un flujo fijo.\n"
        "buscar_pedidos devuelve cantidad, total e ids, pero marca "
        "status incompleto porque no trae productos. Si preguntan qué "
        "se compró, el último pedido o el penúltimo, llamá después a "
        "detalle_pedido con el pedido_id de esa respuesta. No cierres "
        "la respuesta sobre productos con el resumen incompleto.\n"
        "Si una herramienta devuelve status error:\n"
        "- pedido_no_encontrado: reintentá con un pedido_id obtenido "
        "de buscar_pedidos. No inventes ids.\n"
        "- cliente_no_encontrado: no reintentes el mismo id y no inventes "
        "cifras. Pedile al usuario un cliente_id válido.\n"
        "En turnos siguientes, el mismo thread ya tiene la conversación. "
        "Usá ese contexto (cliente, ids, cuál era el último) y llamá "
        "herramientas solo para los datos que todavía no tengas."
    )
)


def build_model() -> BaseChatModel:
    """Chat model según LLM_PROVIDER, sin herramientas todavía."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Falta OPENAI_API_KEY en el entorno (.env).")
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("Falta ANTHROPIC_API_KEY en el entorno (.env).")
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Falta GEMINI_API_KEY en el entorno (.env).")
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            temperature=temperature,
            max_output_tokens=max_tokens,
            api_key=api_key,
        )

    raise ValueError(
        f"Proveedor no soportado: '{provider}'. Usa 'openai', 'anthropic' o 'gemini'."
    )


def trim_history(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Recorta el contexto que ve el modelo sin partir un par tool_call/resultado."""
    if len(messages) <= MAX_CONTEXT_MESSAGES:
        return list(messages)

    window = list(messages[-MAX_CONTEXT_MESSAGES:])
    while window and isinstance(window[0], ToolMessage):
        window.pop(0)
    logger.info(
        "Contexto recortado de %s a %s mensajes para el modelo.",
        len(messages),
        len(window),
    )
    return window


def build_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """Modelo con tools enlazadas, arista condicional y checkpointer."""
    llm = build_model().bind_tools(TOOLS)

    def call_model(state: AgentState) -> dict[str, Any]:
        window = trim_history(state["messages"])
        response = llm.invoke([SYSTEM_PROMPT, *window])
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)
