---
name: openclaw-moe-orchestrator
description: Orchestrate OpenClaw agents with DeepSpeed MoE (Mixture of Experts) for high-performance AI tasks. Use for: setting up autonomous agents, configuring DeepSpeed ZeRO-2/3, and running MoE-based market or data analysis.
---

# OpenClaw MoE Orchestrator

This skill enables the orchestration of autonomous agents using the **OpenClaw** framework, optimized with **DeepSpeed Mixture of Experts (MoE)**. It is designed for high-performance computing tasks in sandboxed environments, specifically for data analysis, anomaly detection, and autonomous decision-making.

## Core Workflow

### 1. Environment Setup
To use this skill, the environment must have high-performance AI dependencies. Use `uv` for fast installation:
```bash
sudo pip3 install uv && sudo uv pip install --system torch deepspeed transformers accelerate yfinance pandas numpy matplotlib mpi4py
sudo apt-get update && sudo apt-get install -y build-essential g++ python3.11-dev libopenmpi-dev ninja-build
```

### 2. OpenClaw Initialization
Clone and build the OpenClaw framework:
```bash
git clone https://github.com/openclaw/openclaw.git
cd openclaw && sudo corepack enable && pnpm install && pnpm build
```

### 3. DeepSpeed MoE Configuration
Use the provided templates to configure DeepSpeed ZeRO-2/3. For CPU-based environments, ZeRO-2 with offload is recommended.
- Template: `templates/ds_config_zero2.json`

### 4. Running Autonomous Missions
Orchestrate the agent to perform specific tasks using the `scripts/` provided:
- `prepare_market_data.py`: Fetches and normalizes market data.
- `moe_anomaly_detector.py`: Trains an MoE model for anomaly detection.
- `visualize_anomalies.py`: Generates visual reports.

## Best Practices
- **Always use `uv`**: Standard `pip` is too slow for massive AI libraries.
- **CPU Offloading**: In sandboxed environments without dedicated GPUs, ensure `cpu_offload: true` is set in DeepSpeed config.
- **JIT Compilation**: Ensure `build-essential` and `python-dev` are installed to allow DeepSpeed to compile its C++ extensions.

## Bundled Resources
- `scripts/`: Implementation of market data gathering and MoE anomaly detection.
- `templates/`: DeepSpeed and OpenClaw configuration boilerplates.
- `references/`: Documentation for advanced MoE tuning.
