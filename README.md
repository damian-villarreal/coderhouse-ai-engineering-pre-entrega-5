# Pre-entrega 5: Agente de razonamiento cíclico con memoria persistente

Agente ReAct en LangGraph. El modelo decide cuándo llamar a una herramienta, puede hacer un segundo paso si la primera respuesta viene incompleta o con error, y retoma la conversación con el mismo `thread_id` gracias a `SqliteSaver`.

## Requisitos

- Python **3.12**
- Una API key de OpenAI, Anthropic o Gemini

## Estructura

```
.
├── state.py          # AgentState(MessagesState)
├── tools.py          # buscar_pedidos y detalle_pedido (@tool)
├── graph.py          # StateGraph, tools_condition, bind_tools, SqliteSaver
├── main.py           # prueba multi-paso + memoria + error
├── traces/
│   └── ejecucion.json
├── requirements.txt
├── .env.example
└── README.md
```

## Instalación

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Editar .env y cargar la API key del proveedor elegido
```

## Variables de entorno

| Variable | Descripción | Ejemplo |
|---|---|---|
| `LLM_PROVIDER` | `openai`, `anthropic` o `gemini` | `openai` |
| `OPENAI_API_KEY` | Clave de OpenAI | `sk-...` |
| `ANTHROPIC_API_KEY` | Clave de Anthropic | `sk-ant-...` |
| `GEMINI_API_KEY` | Clave de Gemini | `AIza...` |
| `OPENAI_MODEL` | Modelo de OpenAI, opcional | `gpt-4o-mini` |
| `ANTHROPIC_MODEL` | Modelo de Anthropic, opcional | `claude-3-5-haiku-latest` |
| `GEMINI_MODEL` | Modelo de Gemini, opcional | `gemini-3.8-flash` |
| `LLM_TEMPERATURE` | Temperatura, opcional | `0` |
| `LLM_MAX_TOKENS` | Máximo de tokens, opcional | `1024` |

## Cómo ejecutar

```bash
source .venv/bin/activate
python main.py
```

El script borra `data/checkpoints.sqlite` al empezar, corre tres turnos con el mismo `thread_id` (`sesion-cliente-102`) y escribe la traza en `traces/ejecucion.json`. El tope del ciclo es `recursion_limit=10`.

1. **Multi-paso.** Pregunta cuántos pedidos tuvo el cliente 102, el total y los productos del último. `buscar_pedidos` devuelve cantidad y total, pero marca el resultado como incompleto (no trae productos). El agente llama después a `detalle_pedido`.
2. **Memoria.** Con el mismo hilo pregunta por el penúltimo, sin repetir el cliente. El checkpointer conserva el contexto.
3. **Error.** Consulta el cliente 999. La herramienta responde que no existe y el agente pide un id válido en lugar de inventar cifras.

## Grafo

`AgentState` hereda de `MessagesState`, así que `messages` se acumula con el reducer `add_messages`. El grafo tiene dos nodos:

- `agent`: llama al modelo ya vinculado con `llm.bind_tools()`.
- `tools`: ejecuta la herramienta que el modelo eligió.

`tools_condition` decide si vuelve a `tools` o termina. Si el contexto que ve el modelo pasa de 20 mensajes, se recorta la cola sin cortar un par tool-call / resultado. El historial completo sigue en SQLite.
