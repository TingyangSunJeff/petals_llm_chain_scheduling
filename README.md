# petals_llm_chain_scheduling

Server chain composition and load balancing for distributed LLM serving via [PETALS](https://github.com/bigscience-workshop/petals).

Implements the algorithms from **"Serving Chain-structured Jobs with Large Memory Footprints with Application to Large Foundation Model Serving"** (MobiHoc '26, Sun et al.).

## Algorithms

- **GBP-CR** (Algorithm 1): Greedy Block Placement with Cache Reservation
- **GCA** (Algorithm 2): Greedy Cache Allocation
- **JFFC** (Algorithm 3): Join-the-Fastest-Free-Chain load balancing
- **Theorem 4 Bounds**: Closed-form response time bounds via birth-death process analysis
- **Parameter c Optimizer**: Automatic cache reservation tuning

## Setup

```bash
pip install -e ".[dev]"
```

## Experimental Setup

Targets 3× A100 (80 GB) GPUs partitioned via MIG into 9 instances:
- 3× 3g.40gb (high-performance)
- 6× 2g.20gb (low-performance)

Model: LLaMA-2-7B (32 transformer blocks)

## Usage

```bash
# Setup MIG partitions
chain-mig-setup --gpus 0 1 2

# Launch orchestrator
chain-launch-orchestrator --config config.json --port 8080

# Launch servers with assigned blocks
chain-launch-server --model meta-llama/Llama-2-7b-hf --start-block 0 --end-block 16

# Run full experiment
chain-run-experiment --config config.json --trace azure_trace.csv
```

## Project Structure

```
petals_llm_chain_scheduling/
├── algorithms/          # Core algorithms (GBP-CR, GCA, JFFC, Theorem 4)
├── orchestrator/        # Centralized orchestrator with JFFC dispatch
├── server/              # Extended PETALS server with fixed block assignment
├── infra/               # MIG setup, network latency emulation
├── benchmarks/          # PETALS baseline, BPRR, JFFC-only
├── experiment/          # Azure trace replay, metrics, experiment runner
├── data_models.py       # Shared data classes
└── cli.py               # Command-line entry points
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Sun26arXiv,
  author  = {Tingyang Sun and Ting He and I-Hong Hou},
  title   = {Serving Chain-structured Jobs with Large Memory Footprints with Application to Large Foundation Model Serving},
  journal = {CoRR},
  year    = {2026},
  doi     = {10.48550/arXiv.2604.14993},
  url     = {https://doi.org/10.48550/arXiv.2604.14993}
}
```

This project also builds on [PETALS](https://github.com/bigscience-workshop/petals):

```bibtex
@inproceedings{borzunov2023petals,
  title={Petals: Collaborative Inference and Fine-tuning of Large Models},
  author={Borzunov, Alexander and Baranchuk, Dmitry and Dettmers, Tim and Riabinin, Maksim and Belkada, Younes and Chumachenko, Artem and Samygin, Pavel and Raffel, Colin},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)},
  year={2023}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
