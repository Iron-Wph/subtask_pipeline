import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency error is raised later if needed.
    load_dotenv = None


DEFAULT_MODEL = "gemini-3.1-pro-preview"


@dataclass(frozen=True)
class GeminiSettings:
    api_key: str
    model: str
    temperature: float = 0.2
    max_output_tokens: int = 4096
    max_retries: int = 8
    retry_initial_delay: float = 10.0
    retry_max_delay: float = 120.0
    max_response_retries: int = 3
    json_mode: bool = True
    thinking_level: Optional[str] = None
    thinking_budget: Optional[int] = None
    include_thoughts: bool = False
    save_thoughts: bool = False


def load_environment(env_file: Optional[Path] = None) -> None:
    if load_dotenv is None:
        return
    if env_file is not None:
        load_dotenv(env_file)
    else:
        load_dotenv()


def build_gemini_settings(
    *,
    env_file: Optional[Path] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    max_retries: Optional[int] = None,
    retry_initial_delay: Optional[float] = None,
    retry_max_delay: Optional[float] = None,
    max_response_retries: Optional[int] = None,
    json_mode: bool = True,
    thinking_level: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    include_thoughts: bool = False,
    save_thoughts: bool = False,
) -> GeminiSettings:
    load_environment(env_file)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY. Put it in .env or the shell.")

    env_model = os.getenv("GEMINI_MODEL")
    return GeminiSettings(
        api_key=api_key,
        model=model or env_model or DEFAULT_MODEL,
        temperature=temperature if temperature is not None else float(os.getenv("GEMINI_TEMPERATURE", "0.2")),
        max_output_tokens=(
            max_output_tokens
            if max_output_tokens is not None
            else int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096"))
        ),
        max_retries=max_retries if max_retries is not None else int(os.getenv("GEMINI_MAX_RETRIES", "8")),
        retry_initial_delay=(
            retry_initial_delay
            if retry_initial_delay is not None
            else float(os.getenv("GEMINI_RETRY_INITIAL_DELAY", "10.0"))
        ),
        retry_max_delay=(
            retry_max_delay
            if retry_max_delay is not None
            else float(os.getenv("GEMINI_RETRY_MAX_DELAY", "120.0"))
        ),
        max_response_retries=(
            max_response_retries
            if max_response_retries is not None
            else int(os.getenv("GEMINI_MAX_RESPONSE_RETRIES", "3"))
        ),
        json_mode=json_mode,
        thinking_level=thinking_level,
        thinking_budget=thinking_budget,
        include_thoughts=include_thoughts,
        save_thoughts=save_thoughts,
    )
