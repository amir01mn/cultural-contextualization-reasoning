"""
Unified LLM backend — supports Ollama (local Mac) and HuggingFace (Narval GPU).

Usage:
    backend = LLMBackend(backend="ollama")   # Mac
    backend = LLMBackend(backend="hf")       # Narval

Both expose the same generate(system_prompt, user_prompt) -> str interface.
"""
from __future__ import annotations
import os
import time

OLLAMA_MODEL = "qwen2.5:7b"
HF_MODEL     = "Qwen/Qwen2.5-7B-Instruct"
MAX_TOKENS   = 1024
MAX_RETRIES  = 3
RETRY_DELAY  = 3.0


class LLMBackend:
    def __init__(self, backend: str = "ollama"):
        self.backend = backend.lower()
        self._client = None
        self._pipeline = None

        if self.backend == "ollama":
            self._init_ollama()
        elif self.backend == "hf":
            self._init_hf()
        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'ollama' or 'hf'.")

    # ------------------------------------------------------------------ ollama
    def _init_ollama(self) -> None:
        from openai import OpenAI
        self._client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        print(f"[llm_backend] Ollama backend ready ({OLLAMA_MODEL})")

    def _generate_ollama(self, system_prompt: str, user_prompt: str) -> str:
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=OLLAMA_MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=0.3,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"[llm_backend] Ollama error: {e}")
                    return ""
        return ""

    # ------------------------------------------------------------------ hf
    def _init_hf(self) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        print(f"[llm_backend] Loading {HF_MODEL} via HuggingFace ...")
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
        )
        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_TOKENS,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        print(f"[llm_backend] HuggingFace backend ready")

    def _generate_hf(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        try:
            output = self._pipeline(messages)
            # HF returns full conversation; extract last assistant turn
            generated = output[0]["generated_text"]
            if isinstance(generated, list):
                return generated[-1].get("content", "")
            return str(generated)
        except Exception as e:
            print(f"[llm_backend] HF error: {e}")
            return ""

    # ------------------------------------------------------------------ unified
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.backend == "ollama":
            return self._generate_ollama(system_prompt, user_prompt)
        return self._generate_hf(system_prompt, user_prompt)
