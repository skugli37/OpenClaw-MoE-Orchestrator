from __future__ import annotations

from dataclasses import dataclass

import requests


class OllamaClientError(RuntimeError):
    """Raised when the local Ollama runtime cannot satisfy a request."""


@dataclass(frozen=True)
class OllamaResponse:
    model: str
    content: str
    prompt_eval_count: int | None = None
    eval_count: int | None = None


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout_seconds: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._session = requests.Session()

    def list_models(self) -> list[str]:
        response = self._session.get(f"{self.base_url}/api/tags", timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        return [str(model["name"]) for model in payload.get("models", []) if model.get("name")]

    def is_healthy(self) -> bool:
        try:
            self.list_models()
        except requests.RequestException:
            return False
        return True

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        options: dict | None = None,
        stream: bool = False,
    ) -> OllamaResponse:
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if options:
            payload["options"] = options

        try:
            response = self._session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise OllamaClientError(f"Ollama chat request failed for model {model}: {error}") from error

        body = response.json()
        message = body.get("message") or {}
        content = str(message.get("content") or "").strip()
        if not content:
            raise OllamaClientError(f"Ollama model {model} returned an empty response")
        return OllamaResponse(
            model=str(body.get("model") or model),
            content=content,
            prompt_eval_count=body.get("prompt_eval_count"),
            eval_count=body.get("eval_count"),
        )
