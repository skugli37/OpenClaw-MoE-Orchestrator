---
name: openclaw-moe-orchestrator
description: Orchestrate OpenClaw agents with DeepSpeed MoE and Ollama Cloud model rotation. Use for: configuring OpenClaw auth profiles, wiring Ollama providers, and running MoE-based market or data analysis without OpenAI or Anthropic providers.
---

# OpenClaw MoE Orchestrator

This skill covers the production path for this repository: OpenClaw orchestration, Ollama-backed cloud model rotation, and DeepSpeed MoE workflows. Keep OpenAI and Anthropic out of the active runtime path.

## Core Workflow

### 1. Install the packaged runtime
Use the repository environment instead of ad hoc global installs:
```bash
python3 -m pip install -e .
```

### 2. Bring up OpenClaw + Ollama Cloud access
Use the Ollama daemon as the provider surface, then pull or route Ollama Cloud model tags:
```bash
ollama signin
ollama serve
ollama pull gpt-oss:120b-cloud
ollama pull qwen3-coder-next:cloud
ollama pull qwen3-next:80b-cloud
ollama pull qwen3-vl:235b-cloud
curl http://127.0.0.1:11434/api/tags
```

### 3. Seed OpenClaw auth for Ollama operation
Use `templates/auth-profiles.json` as the starting point for `~/.openclaw/agents/<agentId>/agent/auth-profiles.json`.
- The checked-in profile already carries the non-secret marker value `ollama-cloud`.
- The real Cloud credential belongs on the Ollama daemon side, not in the OpenClaw provider profile.

### 4. Configure the Ollama provider in native mode
For production, prefer OpenClaw's native Ollama adapter:
- `baseUrl`: `http://127.0.0.1:11434`
- `api`: `ollama`
- Model refs: `ollama/<model>`
- Rotation/failover should use Ollama Cloud model tags from `configs/ollama_model_manifest.json`
- Do not add `/v1` unless you intentionally need OpenAI-compatible proxy mode.
- Use repeated `--reasoning-model`, `--coding-model`, `--general-model`, `--vision-model`, `--embedding-model`, and `--safety-model` flags on `openclaw-moe install-openclaw-cloud` when you want a different active role order without editing the manifest.

### 5. Run repository workflows
Use the packaged CLI or the thin compatibility wrappers in `scripts/`:
- `openclaw-moe serve-gui`
- `openclaw-moe run-mission`
- `openclaw-moe run-integrated`
- `openclaw-moe detect-single`
- `openclaw-moe detect-multi`

## Best Practices
- Use the packaged entry points for production runs.
- Keep OpenClaw on a loopback bind unless you have a deliberate gateway design.
- Prefer explicit Ollama provider config in production so model inventory, cloud tags, and context windows stay deterministic.
- Treat `templates/` as bootstrap material and move deployed configs into your OpenClaw state directory.
- Use `docs/openclaw_local_ollama_integration.md` for the full Ollama Cloud setup sequence.

## Bundled Resources
- `templates/auth-profiles.json`: env-backed Ollama auth profile template.
- `templates/ds_config_zero3.json`: DeepSpeed bootstrap config.
- `scripts/`: thin compatibility wrappers around the packaged production runtime.
