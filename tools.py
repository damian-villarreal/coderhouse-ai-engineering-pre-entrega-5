"""Herramientas del agente. Simulan una base de pedidos.

El modelo elige cuál llamar leyendo el docstring. ``buscar_pedidos``
devuelve el agregado y los ids, pero nunca los productos: si la
pregunta pide qué se compró, el resultado queda incompleto y el agente
tiene que llamar a ``detalle_pedido``.
"""

import json
import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger("agente.tools")

# cliente 102: 3 pedidos, total 14500. El último es el 5012.
_PEDIDOS: dict[int, list[dict[str, Any]]] = {
    102: [
        {
            "pedido_id": 5001,
            "fecha": "2026-01-12",
            "total": 3200,
            "productos": ["Teclado mecánico"],
        },
        {
            "pedido_id": 5008,
            "fecha": "2026-03-02",
            "total": 4800,
            "productos": ["Monitor 27 pulgadas"],
        },
        {
            "pedido_id": 5012,
            "fecha": "2026-06-18",
            "total": 6500,
            "productos": ["Soporte para notebook", "Mouse inalámbrico"],
        },
    ],
    215: [
        {
            "pedido_id": 5100,
            "fecha": "2026-02-01",
            "total": 8900,
            "productos": ["Auriculares"],
        },
    ],
    340: [
        {
            "pedido_id": 5204,
            "fecha": "2026-04-19",
            "total": 2100,
            "productos": ["Cable HDMI"],
        },
        {
            "pedido_id": 5210,
            "fecha": "2026-05-03",
            "total": 1500,
            "productos": ["Hub USB-C"],
        },
    ],
}

_PEDIDO_POR_ID: dict[int, dict[str, Any]] = {
    pedido["pedido_id"]: {**pedido, "cliente_id": cliente_id}
    for cliente_id, pedidos in _PEDIDOS.items()
    for pedido in pedidos
}


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


@tool
def buscar_pedidos(cliente_id: int) -> str:
    """Busca el resumen de pedidos de un cliente en la base comercial.

    Usala cuando pregunten cuántos pedidos tuvo un cliente, cuál fue el
    total facturado, las fechas o los ids de esos pedidos.

    Parámetros:
        cliente_id: identificador numérico del cliente, por ejemplo 102.
            No acepta nombres. Si no tenés el id, no inventes uno.

    Devuelve un JSON:
        - status "ok" no ocurre en esta herramienta: el resumen nunca
          incluye los productos de cada pedido.
        - status "incompleto": trae pedidos (cantidad), total, la lista
          ordenada de más viejo a más nuevo y ultimo_pedido_id. NO trae
          productos. Si la pregunta pide qué se compró, qué incluye un
          pedido o el detalle del último o del penúltimo, tenés que
          llamar después a detalle_pedido con el pedido_id que figure
          acá. No respondas sobre productos antes de esa segunda llamada.
        - status "error" con codigo "cliente_no_encontrado": ese id no
          existe. No inventes totales. Pedile al usuario un cliente_id
          válido. No reintentes el mismo id.
    """
    logger.info("buscar_pedidos(cliente_id=%s)", cliente_id)
    pedidos = _PEDIDOS.get(cliente_id)
    if pedidos is None:
        return _dump(
            {
                "status": "error",
                "codigo": "cliente_no_encontrado",
                "cliente_id": cliente_id,
                "mensaje": (
                    f"No existe el cliente {cliente_id}. "
                    "No reintentes este id. Pedile al usuario un cliente_id válido. "
                    f"Clientes disponibles: {sorted(_PEDIDOS)}."
                ),
            }
        )

    resumen = [
        {
            "pedido_id": pedido["pedido_id"],
            "fecha": pedido["fecha"],
            "total": pedido["total"],
        }
        for pedido in pedidos
    ]
    total = sum(pedido["total"] for pedido in pedidos)
    return _dump(
        {
            "status": "incompleto",
            "cliente_id": cliente_id,
            "pedidos": len(pedidos),
            "total": total,
            "ordenados_del_mas_viejo_al_mas_nuevo": resumen,
            "ultimo_pedido_id": pedidos[-1]["pedido_id"],
            "mensaje": (
                "Resumen agregado listo. Esta consulta no incluye productos. "
                "Si necesitás saber qué se compró en un pedido, llamá a "
                "detalle_pedido(pedido_id) con el id de esta lista antes de "
                "responder. No inventes los productos."
            ),
        }
    )


@tool
def detalle_pedido(pedido_id: int) -> str:
    """Trae los productos de un pedido concreto.

    Usala después de buscar_pedidos, cuando haga falta saber qué se
    compró: el último pedido, el penúltimo, o un pedido_id puntual.
    El pedido_id sale de la lista que devuelve buscar_pedidos. No lo
    inventes.

    Parámetros:
        pedido_id: id numérico del pedido, por ejemplo 5012.

    Devuelve un JSON:
        - status "ok": cliente_id, fecha, total y productos.
        - status "error" con codigo "pedido_no_encontrado": ese id no
          existe. No inventes productos. Si todavía no consultaste los
          pedidos del cliente, llamá a buscar_pedidos y reintentá
          detalle_pedido con un pedido_id de esa respuesta.
    """
    logger.info("detalle_pedido(pedido_id=%s)", pedido_id)
    pedido = _PEDIDO_POR_ID.get(pedido_id)
    if pedido is None:
        return _dump(
            {
                "status": "error",
                "codigo": "pedido_no_encontrado",
                "pedido_id": pedido_id,
                "mensaje": (
                    f"No existe el pedido {pedido_id}. No inventes productos. "
                    "Llamá a buscar_pedidos(cliente_id) y reintentá "
                    "detalle_pedido con un pedido_id de esa respuesta."
                ),
            }
        )

    return _dump(
        {
            "status": "ok",
            "pedido_id": pedido["pedido_id"],
            "cliente_id": pedido["cliente_id"],
            "fecha": pedido["fecha"],
            "total": pedido["total"],
            "productos": pedido["productos"],
        }
    )


TOOLS = [buscar_pedidos, detalle_pedido]
