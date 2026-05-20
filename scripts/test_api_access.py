"""Smoke test API access for SafeAlert model providers."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI


TEST_PROMPT = "Say hello."
REQUEST_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model: str
    api_key_env: str
    base_url: str
    endpoint: str


MODELS = [
    ModelConfig(
        name="GPT-4o mini",
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        endpoint="https://api.openai.com/v1/chat/completions",
    ),
    ModelConfig(
        name="Llama 3.1 8B",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        api_key_env="TOGETHER_API_KEY",
        base_url="https://api.together.xyz/v1",
        endpoint="https://api.together.xyz/v1/chat/completions",
    ),
]


def run_test(config: ModelConfig) -> None:
    """Run one test prompt against a configured chat completions endpoint."""
    print(f"\nModel: {config.name}")
    print(f"Endpoint: {config.endpoint}")

    api_key = os.getenv(config.api_key_env)
    if not api_key:
        print(f"Status: error")
        print(f"Error: {config.api_key_env} is not set in .env or the environment.")
        return

    try:
        client = OpenAI(api_key=api_key, base_url=config.base_url)
        response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            temperature=0,
            max_tokens=20,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response_text = response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001 - smoke test should report all provider failures.
        print("Status: error")
        print(f"Error: {type(exc).__name__}: {exc}")
        return

    print("Status: success")
    print(f"Response: {response_text.strip()}")


def main() -> None:
    """Load API keys from .env and test each configured provider."""
    load_dotenv()
    for config in MODELS:
        run_test(config)


if __name__ == "__main__":
    main()
