# OpenClaw With Ollama Cloud Models

## Goal

Run OpenClaw and this repository with Ollama as the only LLM provider surface, using Ollama Cloud model tags and rotation, with no OpenAI or Anthropic provider path.

## Prerequisites

- Linux or macOS host with local shell access
- `python` 3.12+
- OpenClaw installed locally
- Ollama installed locally
- Ollama account access for cloud model tags

## 1. Authenticate Ollama and prepare cloud models

```bash
ollama signin
ollama serve
ollama pull gpt-oss:120b-cloud
ollama pull qwen3-coder-next:cloud
ollama pull qwen3-next:80b-cloud
ollama pull qwen3-vl:235b-cloud
curl http://127.0.0.1:11434/api/tags
```

The daemon stays on `http://127.0.0.1:11434`, while the selected models are Ollama Cloud tags.

## 2. Install this repository runtime

```bash
cd /home/kugli/OpenClaw-MoE-Orchestrator
python3 -m pip install -e .
```

## 3. Seed OpenClaw auth profiles

Use the checked-in template as the bootstrap file:

```bash
mkdir -p ~/.openclaw/agents/main/agent
cp skills/openclaw-moe-orchestrator/templates/auth-profiles.json \
  ~/.openclaw/agents/main/agent/auth-profiles.json
```

The checked-in auth profile already uses the non-secret marker `ollama-cloud`. The real Cloud credential belongs on the Ollama daemon side.

## 4. Configure OpenClaw for Ollama Cloud rotation

Write `~/.openclaw/openclaw.json` with an explicit Ollama provider and cloud model refs. The repository can generate this overlay with:

```bash
openclaw-moe install-openclaw-cloud
```

The generated config uses:

- `gateway.mode = "local"`
- `models.providers.ollama.baseUrl = "http://127.0.0.1:11434"`
- `api = "ollama"`
- cloud-capable primary/fallback model refs from `configs/ollama_model_manifest.json`

You can override the active role order without editing the manifest:

```bash
openclaw-moe install-openclaw-cloud \
  --reasoning-model qwen3-next:80b-cloud \
  --reasoning-model gpt-oss:120b-cloud \
  --coding-model qwen3-coder-next:cloud \
  --general-model gpt-oss:120b-cloud
```

## 5. Verify the provider path

```bash
openclaw-moe doctor-openclaw-cloud
openclaw-moe sync-ollama-models --role reasoning --dry-run
openclaw-moe sync-ollama-models --role coding --dry-run
```

Expected result:
- OpenClaw resolves only `ollama` provider auth
- model refs resolve as `ollama/<cloud-model-tag>`
- no `openai`, `anthropic`, or `claude` provider is required

## 6. Run OpenClaw and repository workflows

```bash
openclaw gateway --port 18789 --verbose
openclaw-moe run-mission
openclaw-moe run-integrated
```

## Production Notes

- Keep OpenAI and Anthropic providers out of `auth-profiles.json` and `openclaw.json`.
- The active rotation set is defined in `configs/ollama_model_manifest.json`.
- If a cloud tag is unavailable or rate-limited, OpenClaw should fail over to the next Ollama model in the configured order.
- `/v1` OpenAI-compatible mode is not the primary production path here.
