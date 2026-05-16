import argparse
from pathlib import Path
from typing import Optional

from .cli import DEFAULT_PROMPT_CONFIG
from .config import build_gemini_settings
from .generation import run_generation_pipeline
from .gemini_client import GeminiClient
from .prompts import PromptCatalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate generic structured dataset labels from autolabel prompt information."
    )
    parser.add_argument("--annotation-json", type=Path, required=True, help="Annotation JSON file or directory.")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--prompt-info-json",
        "--autolabel-json",
        "--task-prior-json",
        dest="prompt_info_json",
        type=Path,
        help="Autolabel prompt-info JSON for a single episode.",
    )
    parser.add_argument(
        "--prompt-info-root",
        "--autolabel-root",
        "--prior-root",
        dest="prompt_info_root",
        type=Path,
        help="Directory containing autolabel prompt-info JSON outputs.",
    )
    parser.add_argument("--frame-stride", type=int, default=80)
    parser.add_argument("--request-delay", type=float, default=0.0)
    parser.add_argument("--include-previous-image", action="store_true")
    parser.add_argument("--episode-limit", type=int)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--env-file", type=Path, help="Optional .env path.")
    parser.add_argument("--prompt-config", type=Path, default=DEFAULT_PROMPT_CONFIG)
    parser.add_argument("--model", help="Gemini model. Overrides GEMINI_MODEL from .env.")
    parser.add_argument("--temperature", type=float, help="Gemini temperature.")
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--retry-initial-delay", type=float)
    parser.add_argument("--retry-max-delay", type=float)
    parser.add_argument("--max-response-retries", type=int)
    parser.add_argument("--disable-json-mode", action="store_true")
    parser.add_argument("--thinking-level", choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--thinking-budget", type=int)
    parser.add_argument("--include-thoughts", action="store_true")
    parser.add_argument("--save-thoughts", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = build_gemini_settings(
        env_file=args.env_file,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        max_retries=args.max_retries,
        retry_initial_delay=args.retry_initial_delay,
        retry_max_delay=args.retry_max_delay,
        max_response_retries=args.max_response_retries,
        json_mode=not args.disable_json_mode,
        thinking_level=args.thinking_level,
        thinking_budget=args.thinking_budget,
        include_thoughts=args.include_thoughts,
        save_thoughts=args.save_thoughts,
    )
    run_generation_pipeline(
        annotation_path=args.annotation_json,
        image_root=args.image_root,
        output_path=args.output,
        prompt_catalog=PromptCatalog.from_file(args.prompt_config),
        gemini_client=GeminiClient(settings),
        prompt_info_json=args.prompt_info_json,
        prompt_info_root=args.prompt_info_root,
        frame_stride=args.frame_stride,
        request_delay=args.request_delay,
        include_previous_image=args.include_previous_image,
        episode_limit=args.episode_limit,
        episode_offset=args.episode_offset,
        resume=args.resume,
    )
    return 0
