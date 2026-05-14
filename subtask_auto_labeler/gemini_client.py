import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import GeminiSettings
from .io_utils import JsonObject

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - handled at runtime.
    genai = None
    types = None

IMAGE_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class ModelResponseFormatError(ValueError):
    def __init__(self, message: str, content: str):
        super().__init__(f"{message}: {content}")
        self.content = content


def strip_json_fence(content: str) -> str:
    stripped = content.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def parse_json_response(content: str, required_keys: Optional[Iterable[str]] = None) -> JsonObject:
    if not content or not content.strip():
        raise ModelResponseFormatError("Model returned an empty response", content)
    try:
        parsed = json.loads(strip_json_fence(content))
    except json.JSONDecodeError as exc:
        raise ModelResponseFormatError("Model returned invalid JSON", content) from exc
    if not isinstance(parsed, dict):
        raise ModelResponseFormatError("Model returned JSON that is not an object", content)
    if required_keys:
        missing = sorted(set(required_keys) - parsed.keys())
        if missing:
            raise ModelResponseFormatError(f"Model returned JSON missing required keys {missing}", content)
    return parsed


def get_retry_after_seconds(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    retry_after = headers.get("retry-after")
    if not retry_after:
        return None
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        return None


def get_error_status_code(exc: Exception) -> Optional[int]:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    return None


def is_retryable_api_error(exc: Exception) -> bool:
    status_code = get_error_status_code(exc)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    message = f"{exc.__class__.__name__} {exc}".lower()
    retry_markers = (
        "429",
        "rate limit",
        "resource_exhausted",
        "quota",
        "timeout",
        "deadline",
        "temporarily unavailable",
        "connection",
        "proxy",
        "ssl",
        "network",
    )
    return any(marker in message for marker in retry_markers)


class GeminiClient:
    def __init__(self, settings: GeminiSettings):
        if genai is None or types is None:
            raise ImportError("Missing google-genai package. Install it with: pip install google-genai")
        self.settings = settings
        self.client = genai.Client(api_key=settings.api_key)

    def image_to_part(self, image_path: Path) -> Any:
        mime_type = IMAGE_MIME_BY_EXT.get(image_path.suffix.lower(), "image/jpeg")
        with image_path.open("rb") as f:
            image_bytes = f.read()
        return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    def build_config(self, system_instruction: str) -> Any:
        config_kwargs: JsonObject = {
            "system_instruction": system_instruction,
            "temperature": self.settings.temperature,
        }
        if self.settings.max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = self.settings.max_output_tokens
        if self.settings.json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        thinking_kwargs: JsonObject = {}
        if self.settings.include_thoughts:
            thinking_kwargs["include_thoughts"] = True
        if self.settings.thinking_level:
            thinking_kwargs["thinking_level"] = self.settings.thinking_level
        if self.settings.thinking_budget is not None:
            thinking_kwargs["thinking_budget"] = self.settings.thinking_budget
        if thinking_kwargs:
            config_kwargs["thinking_config"] = types.ThinkingConfig(**thinking_kwargs)
        return types.GenerateContentConfig(**config_kwargs)

    def generate_json(
        self,
        *,
        system_instruction: str,
        prompt: str,
        image_paths: Sequence[Path] = (),
        required_keys: Optional[Iterable[str]] = None,
    ) -> Tuple[JsonObject, JsonObject]:
        image_parts = [self.image_to_part(path) for path in image_paths]
        config = self.build_config(system_instruction)
        for format_attempt in range(self.settings.max_response_retries + 1):
            request_prompt = prompt
            if format_attempt > 0:
                request_prompt = (
                    f"{prompt}\n\nThe previous response was invalid or incomplete. "
                    "Return exactly one valid JSON object with the required keys."
                )
            contents: List[Any] = [request_prompt, *image_parts]
            response = self._generate_content_with_retries(contents=contents, config=config)
            response_text, thought_summaries = self._extract_text_and_thoughts(response)
            try:
                parsed = parse_json_response(response_text, required_keys=required_keys)
            except ModelResponseFormatError as exc:
                if format_attempt >= self.settings.max_response_retries:
                    raise
                preview = exc.content.replace("\n", " ")[:120]
                print(
                    "[retry] invalid_or_empty_model_response "
                    f"attempt={format_attempt + 1}/{self.settings.max_response_retries} "
                    f"content={preview!r}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(min(10.0, 1.0 + format_attempt))
                continue

            metadata: JsonObject = {}
            if self.settings.save_thoughts and thought_summaries:
                metadata["thought_summaries"] = thought_summaries
            return parsed, metadata
        raise RuntimeError("unreachable")

    def _generate_content_with_retries(self, *, contents: List[Any], config: Any) -> Any:
        attempt = 0
        while True:
            try:
                return self.client.models.generate_content(
                    model=self.settings.model,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:
                if not is_retryable_api_error(exc) or attempt >= self.settings.max_retries:
                    raise
                retry_after = get_retry_after_seconds(exc)
                fallback_delay = self.settings.retry_initial_delay * (2 ** attempt)
                delay_seconds = min(self.settings.retry_max_delay, retry_after or fallback_delay)
                status_code = get_error_status_code(exc) or ""
                print(
                    "[retry] "
                    f"api_error={exc.__class__.__name__} status={status_code} "
                    f"attempt={attempt + 1}/{self.settings.max_retries} wait={delay_seconds:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay_seconds)
                attempt += 1

    @staticmethod
    def _extract_text_and_thoughts(response: Any) -> Tuple[str, List[str]]:
        answer_parts: List[str] = []
        thought_parts: List[str] = []
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                text = getattr(part, "text", None)
                if not text:
                    continue
                if getattr(part, "thought", False):
                    thought_parts.append(text)
                else:
                    answer_parts.append(text)
        if answer_parts:
            return "".join(answer_parts), thought_parts
        try:
            fallback_text = getattr(response, "text", None)
        except Exception:
            fallback_text = None
        return fallback_text or "", thought_parts
