"""HTTP API endpoint for the orchestrator.

Accepts inference requests via POST /inference and returns generated text.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from aiohttp import web

if TYPE_CHECKING:
    from petals_llm_chain_scheduling.orchestrator.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


def create_app(orchestrator: "Orchestrator") -> web.Application:
    """Create an aiohttp application with the inference endpoint."""
    app = web.Application()
    app["orchestrator"] = orchestrator

    app.router.add_post("/inference", handle_inference)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/metrics", handle_metrics)

    return app


async def handle_inference(request: web.Request) -> web.Response:
    """POST /inference — submit an inference request."""
    orchestrator: Orchestrator = request.app["orchestrator"]
    try:
        body = await request.json()
        prompt = body.get("prompt", "")
        max_new_tokens = body.get("max_new_tokens", 20)
        input_tokens = body.get("input_tokens", 0)

        result = await orchestrator.handle_request(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            input_tokens=input_tokens,
        )
        return web.json_response({
            "request_id": result.request_id,
            "generated_text": result.generated_text,
            "response_time": result.metrics.response_time,
            "waiting_time": result.metrics.waiting_time,
            "service_time": result.metrics.service_time,
        })
    except Exception as e:
        logger.exception("Inference request failed")
        return web.json_response({"error": str(e)}, status=500)


async def handle_health(request: web.Request) -> web.Response:
    """GET /health — health check."""
    orchestrator: Orchestrator = request.app["orchestrator"]
    return web.json_response({
        "status": "ok",
        "queue_length": orchestrator.dispatcher.queue_length,
        "total_ongoing": orchestrator.dispatcher.total_ongoing,
    })


async def handle_metrics(request: web.Request) -> web.Response:
    """GET /metrics — return collected metrics summary."""
    orchestrator: Orchestrator = request.app["orchestrator"]
    from petals_llm_chain_scheduling.experiment.metrics import compute_summary
    summary = compute_summary(orchestrator.collected_metrics)
    return web.json_response({
        "num_requests": len(orchestrator.collected_metrics),
        "response_time_mean": summary.response_time.mean,
        "waiting_time_mean": summary.waiting_time.mean,
        "service_time_mean": summary.service_time.mean,
    })
