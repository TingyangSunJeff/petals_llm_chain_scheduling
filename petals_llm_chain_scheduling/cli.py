"""Command-line entry points for petals_llm_chain_scheduling."""

import argparse
import asyncio
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def launch_server():
    """Launch a PETALS server with assigned blocks from the orchestrator."""
    parser = argparse.ArgumentParser(description="Launch server with assigned blocks")
    parser.add_argument("--model", required=True, help="Model name (e.g. meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--start-block", type=int, required=True)
    parser.add_argument("--end-block", type=int, required=True)
    parser.add_argument("--cache-reservation", type=int, default=1)
    parser.add_argument("--initial-peers", nargs="+", default=[])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from petals_llm_chain_scheduling.server.assigned_server import AssignedServer
    server = AssignedServer(
        start_block=args.start_block,
        end_block=args.end_block,
        cache_reservation=args.cache_reservation,
        converted_model_name_or_path=args.model,
        initial_peers=args.initial_peers,
        device=args.device,
    )
    server.run()


def launch_orchestrator():
    """Launch the centralized orchestrator process."""
    parser = argparse.ArgumentParser(description="Launch orchestrator")
    parser.add_argument("--config", required=True, help="Path to system config JSON")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    from petals_llm_chain_scheduling.data_models import ServerParams
    from petals_llm_chain_scheduling.orchestrator.orchestrator import Orchestrator
    from petals_llm_chain_scheduling.orchestrator.api import create_app
    from aiohttp import web

    servers = [ServerParams(**s) for s in cfg["servers"]]
    orchestrator = Orchestrator(
        servers=servers,
        L=cfg["num_blocks"],
        s_m=cfg["block_size_gb"],
        s_c=cfg["cache_size_gb"],
        lam=cfg["arrival_rate"],
        rho_bar=cfg.get("rho_bar", 0.7),
    )
    orchestrator.setup()

    app = create_app(orchestrator)
    logger.info(f"Orchestrator starting on port {args.port}")
    web.run_app(app, port=args.port)


def run_experiment():
    """Run the end-to-end experiment reproducing Table 1."""
    parser = argparse.ArgumentParser(description="Run full experiment")
    parser.add_argument("--config", required=True, help="Path to system config JSON")
    parser.add_argument("--trace", required=True, help="Path to Azure trace CSV")
    parser.add_argument("--num-requests", type=int, default=1000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    from petals_llm_chain_scheduling.data_models import ModelParams, ServerParams
    from petals_llm_chain_scheduling.experiment.runner import ExperimentRunner

    servers = [ServerParams(**s) for s in cfg["servers"]]
    model = ModelParams(
        num_blocks=cfg["num_blocks"],
        block_size_gb=cfg["block_size_gb"],
        cache_size_gb=cfg["cache_size_gb"],
        model_name=cfg.get("model_name", "meta-llama/Llama-2-7b-hf"),
    )

    runner = ExperimentRunner(
        servers=servers,
        model_params=model,
        trace_path=args.trace,
        num_requests=args.num_requests,
        arrival_rate=cfg.get("arrival_rate", 2.57),
    )

    table = asyncio.run(runner.run_all())
    print(table)


def mig_setup():
    """Create MIG partitions on A100 GPUs."""
    parser = argparse.ArgumentParser(description="Setup MIG partitions")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    from petals_llm_chain_scheduling.infra.mig_setup import create_mig_partitions, enable_mig_mode
    enable_mig_mode(args.gpus)
    instances = create_mig_partitions(args.gpus)
    for inst in instances:
        print(f"GPU {inst.gpu_index}: {inst.profile} (UUID: {inst.uuid})")


def mig_teardown():
    """Destroy all MIG partitions and restore GPUs."""
    parser = argparse.ArgumentParser(description="Teardown MIG partitions")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    from petals_llm_chain_scheduling.infra.mig_setup import destroy_mig_partitions
    destroy_mig_partitions(args.gpus)
    print("MIG partitions destroyed.")
