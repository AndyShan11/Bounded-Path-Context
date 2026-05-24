"""vLLM-based local LLM client.

Defaults tuned for 4x RTX 2080 Ti (11GB, sm_75 / Turing):
  - max_model_len=4096 (default vLLM is 32768, too big for KV cache here)
  - gpu_memory_utilization=0.85 (leave ~15% headroom)
  - enforce_eager=True (skip CUDA graphs to save 1-2GB; 2080 Ti has no FA2 anyway)

Qwen3.5 support requires a current vLLM build; see scripts/setup_env.sh.
"""

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class LLMClient:
    def __init__(
        self,
        model: str = "QuantTrio/Qwen3.5-9B-AWQ",
        tp: int = 4,
        dtype: str = "float16",
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: int = 42,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        enforce_eager: bool = True,
        language_model_only: bool = True,
        enable_thinking: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.enable_thinking = enable_thinking
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tp,
            dtype=dtype,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            language_model_only=language_model_only,
            trust_remote_code=True,
        )
        self.params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )

    def _format(self, user_msg: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    def chat_batch(self, prompts):
        if not prompts:
            return []
        chat_prompts = [self._format(p) for p in prompts]
        outs = self.llm.generate(chat_prompts, self.params, use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outs]

    def __call__(self, prompt: str) -> str:
        return self.chat_batch([prompt])[0]
