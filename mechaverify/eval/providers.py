"""Model providers behind one interface.

    gemini      -> Google Gemini via the google-genai SDK (direct API key)
    openrouter  -> any OpenRouter model (closed or open-weight) via the
                   OpenAI-compatible endpoint; use model strings like
                   "openai/gpt-...", "google/gemma-...", "qwen/...:free"
    dummy       -> offline stub for plumbing / dry runs (no network)

API keys are read from the environment (GEMINI_API_KEY / OPENROUTER_API_KEY);
``load_dotenv`` will pull them from a local, gitignored .env if present.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import time

from mechaverify.eval.prompts import SYSTEM


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no dependency); does not overwrite existing env vars."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _read_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _mime(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "image/png"


class Provider:
    name = "base"

    def generate(self, prompt: str, image_path: str | None = None) -> str:
        raise NotImplementedError

    def generate_with_retry(self, prompt: str, image_path: str | None = None,
                            retries: int = 3, backoff: float = 4.0) -> str:
        last = None
        for attempt in range(retries):
            try:
                return self.generate(prompt, image_path)
            except Exception as e:  # noqa: BLE001 - retry, then give up
                last = e
                if attempt < retries - 1:
                    time.sleep(backoff * (attempt + 1))
        print(f"  provider error after {retries} tries: {type(last).__name__}: {last}")
        return ""  # an empty response never verifies and never counts as well-formed


class DummyProvider(Provider):
    name = "dummy"

    def generate(self, prompt: str, image_path: str | None = None) -> str:
        return "<think>offline dry run; no model called</think><answer>unknown</answer>"


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-pro", api_key: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 1536):
        from google import genai  # lazy import
        self._genai = genai
        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, image_path: str | None = None) -> str:
        from google.genai import types
        contents = []
        if image_path:
            with open(image_path, "rb") as f:
                contents.append(types.Part.from_bytes(data=f.read(), mime_type=_mime(image_path)))
        contents.append(prompt)
        resp = self.client.models.generate_content(
            model=self.model, contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM, temperature=self.temperature,
                max_output_tokens=self.max_tokens),
        )
        return resp.text or ""


class OpenRouterProvider(Provider):
    name = "openrouter"

    def __init__(self, model: str, api_key: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 1536):
        from openai import OpenAI  # lazy import
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1",
                             api_key=api_key or os.environ["OPENROUTER_API_KEY"])
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, image_path: str | None = None) -> str:
        content: list = [{"type": "text", "text": prompt}]
        if image_path:
            url = f"data:{_mime(image_path)};base64,{_read_b64(image_path)}"
            content.append({"type": "image_url", "image_url": {"url": url}})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": content}],
            temperature=self.temperature, max_tokens=self.max_tokens,
            extra_headers={"HTTP-Referer": "https://github.com/itswaseemsajjad/MechaVerify",
                           "X-Title": "MechaVerify"},
            # reasoning models otherwise spend the whole budget on hidden reasoning
            # and return empty content; cap reasoning and keep room for the answer.
            extra_body={"reasoning": {"effort": "low"}},
        )
        msg = resp.choices[0].message
        out = msg.content or ""
        if not out:  # fall back to reasoning trace if visible content is empty
            out = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", "") or ""
        return out


def get_provider(name: str, model: str | None = None, **kw) -> Provider:
    load_dotenv()
    name = name.lower()
    if name == "gemini":
        return GeminiProvider(model or "gemini-2.5-pro", **kw)
    if name == "openrouter":
        if not model:
            raise ValueError("openrouter requires --model (e.g. 'qwen/qwen2.5-vl-72b-instruct:free')")
        return OpenRouterProvider(model, **kw)
    if name == "dummy":
        return DummyProvider()
    raise ValueError(f"unknown provider {name!r} (use gemini | openrouter | dummy)")
