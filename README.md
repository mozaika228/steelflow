# SteelFlow

SteelFlow is a lightweight, high-performance library for running large language models **locally** in 2026.
No external APIs. No cloud lock-in. Maximum efficiency on your hardware.

## Key Capabilities

- Automatic model and quantization selection based on hardware
- Hybrid RAG (dense + sparse + rerank + query rewriting)
- Multi-agent pipelines without LangChain bloat
- Built-in observability (Prometheus + Grafana)
- Energy-aware inference (adapts to battery level)

## Quickstart

```bash
pip install steelflow
```

Run modern local engines with optimal performance:

- Ollama
- vLLM
- exllama-v2
- llama.cpp
- transformers + bitsandbytes
- torch.compile

SteelFlow automatically selects the best combination of quantization, backend, and precision for your machine.

## RAG Without the Pain

- Hybrid retrieval (dense + sparse + rerank)
- Query rewriting + self-correction
- Long-term project memory (vector index + dialogue history)

## Multi-Agent Pipelines

- Planner → Researcher → Critic → Executor → Writer
- Tools: search, file I/O, code execution, git operations

## Observability Out of the Box

- p50/p95/p99 latency by pipeline step
- tokens/sec, VRAM, energy consumption
- Grafana dashboard included

## Self-Improvement

- Store response logs
- Self-critique quality scoring
- Train on your data with LoRA/DPO/ORPO (online or offline)

## Works Anywhere

- Auto-detects CPU/GPU and selects q8/q6/q5/q4/GGUF/INT8
- Energy-aware mode reduces quality on low battery

## Why This Is Hard (and Worth It)

- Not just a wrapper — a unified system (inference + RAG + agents + observability + self-improvement)
- 100% local execution — no external APIs
- Hardware-aware adaptation (still rare even in 2026)
- One API for all major engines (vLLM, exllama, torch.compile, llama.cpp)
- Fine-tuning from your logs (LoRA/DPO/ORPO)

## Project Status

SteelFlow is under active development. The current repository provides the initial scaffold and public roadmap.

## License

MIT. See `LICENSE`.

## Contributing

See `CONTRIBUTING.md`.
