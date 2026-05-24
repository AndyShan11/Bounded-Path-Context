"""OpenAI-compatible HTTP LLM client.

Drop-in replacement for LLMClient: exposes `chat_batch(prompts) -> list[str]`
and `__call__(prompt) -> str`, identical to the vLLM-backed client.

Supported paths for this project:
  - DeepSeek API      base_url=https://api.deepseek.com/v1     model=deepseek-chat
  - Local Qwen vLLM   base_url=http://localhost:8000/v1        model=Qwen/Qwen3.5-9B

The API key is read from an env var whose name is passed in (e.g. DEEPSEEK_API_KEY)
so we never store the secret in code or config files.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


class APIClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        max_concurrency: int = 16,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"API key env var '{api_key_env}' not set. "
                f"Run `export {api_key_env}=<your-key>` in the shell first."
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,  # we do our own retry below for finer control
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries

    def _one_call(self, prompt: str) -> str:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = resp.choices[0].message.content
                return (content or "").strip()
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 1s, 2s, 4s
        print(
            f"  [api_client] FINAL FAILURE after {self.max_retries} tries: "
            f"{type(last_err).__name__}: {last_err}",
            flush=True,
        )
        return ""

    def chat_batch(self, prompts):
        if not prompts:
            return []
        results = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            fut_to_i = {ex.submit(self._one_call, p): i for i, p in enumerate(prompts)}
            for fut in as_completed(fut_to_i):
                results[fut_to_i[fut]] = fut.result()
        return results

    def __call__(self, prompt: str) -> str:
        return self._one_call(prompt)
