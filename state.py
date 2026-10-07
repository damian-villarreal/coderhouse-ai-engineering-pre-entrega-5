"""Esquema de estado del agente."""

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Estado del grafo.

    Hereda ``messages`` de MessagesState. Ese campo ya trae el reducer
    ``add_messages``: cada nodo devuelve solo los mensajes nuevos y el
    grafo los acumula. No se reemplaza la lista entera.
    """
