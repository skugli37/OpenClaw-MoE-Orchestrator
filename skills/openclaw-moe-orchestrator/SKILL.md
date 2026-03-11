---
name: openclaw-moe-orchestrator
description: Orchestrate OpenClaw agents with DeepSpeed MoE (Mixture of Experts) for high-performance AI tasks. Use for: setting up autonomous agents, configuring DeepSpeed ZeRO-2/3, and running MoE-based market or data analysis.
---

# OpenClaw MoE Orchestrator

This skill enables the orchestration of autonomous agents using the **OpenClaw** framework, optimized with **DeepSpeed Mixture of Experts (MoE)**. It is designed for high-performance computing tasks focused on real market-data analysis, anomaly detection, and autonomous decision-making.

## Core Workflow

### 1. Environment Setup
Use the repository's packaged runtime instead of ad hoc global installs:
```bash
python -m pip install -e .
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
Use the packaged CLI or the thin compatibility wrappers in `scripts/`:
- `python -m openclaw_moe_orchestrator run-mission`
- `python -m openclaw_moe_orchestrator run-integrated`
- `python -m openclaw_moe_orchestrator detect-single`
- `python -m openclaw_moe_orchestrator detect-multi`

## Best Practices
- **Use the packaged entry points**: Production runs should use the Python package, not hardcoded home-directory paths.
- **Prefer GPU execution when available**: Runtime adapts DeepSpeed config to the local machine and avoids CPU-offload settings that require fragile JIT builds.
- **Keep templates only as templates**: Production code should consume checked-in configs under `configs/`, while `templates/` remain starting points for variants.

## Bundled Resources
- `scripts/`: Thin compatibility wrappers around the packaged production runtime.
- `templates/`: DeepSpeed and OpenClaw configuration boilerplates.
- `references/`: Documentation for advanced MoE tuning.
