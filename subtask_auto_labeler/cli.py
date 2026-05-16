import argparse
from pathlib import Path
from typing import Optional

from .config import build_gemini_settings
from .generation import run_generation_pipeline
from .gemini_client import GeminiClient
from .prior import run_prior_pipeline
from .prompts import PromptCatalog

DEFAULT_PROMPT_CONFIG = Path("prompts/default_prompts.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Subtask prior and structured label generation pipeline.")
    parser.add_argument("--env-file", type=Path, help="Optional .env path. Defaults to .env in the current directory.")
    parser.add_argument("--prompt-config", type=Path, default=DEFAULT_PROMPT_CONFIG)
    parser.add_argument("--model", help="Gemini model. Overrides GEMINI_MODEL from .env.")
    parser.add_argument("--temperature", type=float, help="Gemini temperature.")
    parser.add_argument("--max-output-tokens", type=int, help="Maximum output tokens.")
    parser.add_argument("--max-retries", type=int, help="Retry attempts for retryable API errors.")
    parser.add_argument("--retry-initial-delay", type=float, help="Initial retry delay in seconds.")
    parser.add_argument("--retry-max-delay", type=float, help="Maximum retry delay in seconds.")
    parser.add_argument("--max-response-retries", type=int, help="Retries for invalid JSON responses.")
    parser.add_argument("--disable-json-mode", action="store_true")
    parser.add_argument("--thinking-level", choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--thinking-budget", type=int)
    parser.add_argument("--include-thoughts", action="store_true")
    parser.add_argument("--save-thoughts", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)

    prior = subparsers.add_parser("prior", help="Run child subtask agents and parent task prior agent.")
    prior.add_argument("--annotation-json", type=Path, required=True)
    prior.add_argument("--image-root", type=Path, required=True)
    prior.add_argument("-o", "--output-dir", type=Path, required=True)
    prior.add_argument("-k", "--sample-k", type=int, default=10)
    prior.add_argument("--request-delay", type=float, default=0.0)

    generate = subparsers.add_parser("generate", help="Generate generic structured model_response labels.")
    generate.add_argument("--annotation-json", type=Path, required=True, help="Annotation JSON file or directory.")
    generate.add_argument("--image-root", type=Path, required=True)
    generate.add_argument("-o", "--output", type=Path, required=True)
    generate.add_argument(
        "--prompt-info-json",
        "--autolabel-json",
        "--task-prior-json",
        dest="prompt_info_json",
        type=Path,
        help="Autolabel prompt-info JSON for a single episode.",
    )
    generate.add_argument(
        "--prompt-info-root",
        "--autolabel-root",
        "--prior-root",
        dest="prompt_info_root",
        type=Path,
        help="Directory containing autolabel prompt-info JSON outputs.",
    )
    generate.add_argument("--frame-stride", type=int, default=80)
    generate.add_argument("--request-delay", type=float, default=0.0)
    generate.add_argument("--include-previous-image", action="store_true")
    generate.add_argument("--episode-limit", type=int, help="Process at most this many selected episodes.")
    generate.add_argument("--episode-offset", type=int, default=0, help="Skip this many sorted episodes first.")
    generate.add_argument("--resume", action="store_true", help="Skip complete existing per-episode outputs.")

    run_all = subparsers.add_parser("run-all", help="Run prior first, then structured generation for one episode.")
    run_all.add_argument("--annotation-json", type=Path, required=True)
    run_all.add_argument("--image-root", type=Path, required=True)
    run_all.add_argument("-o", "--output-dir", type=Path, required=True)
    run_all.add_argument("-k", "--sample-k", type=int, default=10)
    run_all.add_argument("--frame-stride", type=int, default=80)
    run_all.add_argument("--request-delay", type=float, default=0.0)
    run_all.add_argument("--include-previous-image", action="store_true")
    return parser


def build_client(args: argparse.Namespace) -> GeminiClient:
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
    return GeminiClient(settings)


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    prompt_catalog = PromptCatalog.from_file(args.prompt_config)
    client = build_client(args)

    if args.command == "prior":
        run_prior_pipeline(
            annotation_json=args.annotation_json,
            image_root=args.image_root,
            output_dir=args.output_dir,
            prompt_catalog=prompt_catalog,
            gemini_client=client,
            k=args.sample_k,
            request_delay=args.request_delay,
        )
        return 0

    if args.command == "generate":
        run_generation_pipeline(
            annotation_path=args.annotation_json,
            image_root=args.image_root,
            output_path=args.output,
            prompt_catalog=prompt_catalog,
            gemini_client=client,
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

    if args.command == "run-all":
        prior_dir = args.output_dir / "prior"
        generation_dir = args.output_dir / "generation"
        run_prior_pipeline(
            annotation_json=args.annotation_json,
            image_root=args.image_root,
            output_dir=prior_dir,
            prompt_catalog=prompt_catalog,
            gemini_client=client,
            k=args.sample_k,
            request_delay=args.request_delay,
        )
        run_generation_pipeline(
            annotation_path=args.annotation_json,
            image_root=args.image_root,
            output_path=generation_dir,
            prompt_catalog=prompt_catalog,
            gemini_client=client,
            task_prior_json=prior_dir / "task_prior.json",
            frame_stride=args.frame_stride,
            request_delay=args.request_delay,
            include_previous_image=args.include_previous_image,
        )
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
