import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .checkpoint import checkpoint_status, error_to_json, save_checkpoint
from .dataset import (
    SampledFrame,
    SkillSpec,
    find_annotation_jsons,
    iter_stride_frames,
    load_episode,
    resolve_episode_image_root,
)
from .gemini_client import GeminiClient
from .io_utils import JsonObject, read_json, write_json
from .prompts import PromptCatalog
from .visual_guidance import sanitize_visual_guidance_text

MODEL_RESPONSE_KEYS = {
    "reasoning",
    "new_memory",
    "subtask",
    "current_skill_status",
    "visible_transition",
    "is_subtask_completed",
}
GLOBAL_PRIOR_KEYS = (
    "task_summary",
)
SKILL_PRIOR_KEYS = (
    "stage_idx",
    "skill_idx",
    "skill_description",
    "skill_type_hypothesis",
    "subtask_name",
    "target_binding",
    "target_visual_description",
    "action_change_process",
    "pre_completion_state",
    "in_progress_state",
    "completion_gates",
    "completion_conditions",
    "required_visual_evidence",
    "state_transition_evidence",
    "negative_conditions",
    "not_sufficient_for_completion",
    "common_false_positives",
    "ambiguous_cases",
    "generation_prompt_guidance",
)
COMPLETED_STATUSES = {"completed", "completed_and_transitioning"}
OUTPUT_STYLE_MAX_RETRIES = 2
META_RULE_LANGUAGE_PATTERNS = (
    r"\baccording to\b",
    r"\bcriteria\b",
    r"\bcriterion\b",
    r"\bcompletion gate\b",
    r"\bcompletion gates\b",
    r"\btask rule\b",
    r"\btask rules\b",
    r"\bjudging rule\b",
    r"\bjudging rules\b",
    r"\brule requires\b",
    r"\brules require\b",
    r"\brequired by\b",
    r"\brequirement\b",
    r"\brequirements\b",
    r"\bguidance\b",
    r"\bguardrail\b",
    r"\bguardrails\b",
    r"\bprompt\b",
    r"\binstruction\b",
    r"\binstructions\b",
    r"\bto satisfy\b",
    r"\bsatisfy the\b",
    r"\bcurrent_skill_status\b",
    r"\bis_subtask_completed\b",
    r"\bstatus should\b",
    r"\blabel should\b",
    r"\bshould be marked\b",
    r"\bmust be marked\b",
    r"\ballowed to mark\b",
    r"\bskill remains in progress\b",
    r"\bskill is in progress\b",
)


def run_generation_pipeline(
    *,
    annotation_path: Path,
    image_root: Path,
    output_path: Path,
    prompt_catalog: PromptCatalog,
    gemini_client: GeminiClient,
    prompt_info_json: Optional[Path] = None,
    prompt_info_root: Optional[Path] = None,
    task_prior_json: Optional[Path] = None,
    prior_root: Optional[Path] = None,
    frame_stride: int = 80,
    request_delay: float = 0.0,
    include_previous_image: bool = False,
    episode_limit: Optional[int] = None,
    episode_offset: int = 0,
    resume: bool = False,
    save_rendered_prompts: bool = False,
) -> JsonObject:
    annotation_jsons = find_annotation_jsons(annotation_path)
    total_episode_count = len(annotation_jsons)
    if episode_offset < 0:
        raise ValueError("episode_offset must be >= 0")
    if episode_limit is not None and episode_limit < 1:
        raise ValueError("episode_limit must be >= 1")
    if episode_offset or episode_limit is not None:
        start = episode_offset
        end = None if episode_limit is None else episode_offset + episode_limit
        annotation_jsons = annotation_jsons[start:end]
    if not annotation_jsons:
        raise ValueError("No annotation JSON files selected. Check episode_offset and episode_limit.")

    multiple = len(annotation_jsons) > 1 or annotation_path.is_dir()
    if output_path.suffix.lower() == ".json":
        aggregate_path = output_path
        output_dir = output_path.parent
    else:
        aggregate_path = output_path / "generation_results.json"
        output_dir = output_path

    episodes: List[JsonObject] = []
    current_episode: Optional[Path] = None
    try:
        for annotation_json in annotation_jsons:
            current_episode = annotation_json
            episode_image_root = resolve_episode_image_root(
                annotation_json,
                image_root,
                multiple_episodes=multiple,
            )
            episode_output_path = output_dir / f"{annotation_json.stem}_generation.json"
            if resume and episode_output_path.exists():
                existing_output = read_json(episode_output_path)
                if is_complete_generation_output(existing_output):
                    print(f"[skip] existing={episode_output_path}", flush=True)
                    episodes.append(existing_output)
                    continue

            episode_output = run_episode_generation(
                annotation_json=annotation_json,
                image_root=episode_image_root,
                output_path=episode_output_path,
                prompt_catalog=prompt_catalog,
                gemini_client=gemini_client,
                prompt_info_json=resolve_prompt_info_path(
                    annotation_json=annotation_json,
                    explicit_prompt_info_json=prompt_info_json or task_prior_json,
                    prompt_info_root=prompt_info_root or prior_root,
                    multiple_episodes=multiple,
                ),
                frame_stride=frame_stride,
                request_delay=request_delay,
                include_previous_image=include_previous_image,
                save_rendered_prompts=save_rendered_prompts,
            )
            episodes.append(episode_output)
    except BaseException as exc:
        aggregate = build_generation_aggregate(
            annotation_path=annotation_path,
            image_root=image_root,
            episode_offset=episode_offset,
            episode_limit=episode_limit,
            selected_episode_count=len(annotation_jsons),
            total_episode_count=total_episode_count,
            episodes=episodes,
            status=checkpoint_status(exc),
            error=error_to_json(exc),
            current_episode=str(current_episode) if current_episode else "",
        )
        save_checkpoint(aggregate_path, aggregate)
        raise

    aggregate = build_generation_aggregate(
        annotation_path=annotation_path,
        image_root=image_root,
        episode_offset=episode_offset,
        episode_limit=episode_limit,
        selected_episode_count=len(annotation_jsons),
        total_episode_count=total_episode_count,
        episodes=episodes,
        status="complete",
    )
    write_json(aggregate_path, aggregate)
    print(f"[saved] {aggregate_path}", flush=True)
    return aggregate


def run_episode_generation(
    *,
    annotation_json: Path,
    image_root: Path,
    output_path: Path,
    prompt_catalog: PromptCatalog,
    gemini_client: GeminiClient,
    prompt_info_json: Optional[Path],
    frame_stride: int,
    request_delay: float,
    include_previous_image: bool,
    save_rendered_prompts: bool = False,
) -> JsonObject:
    episode = load_episode(annotation_json, image_root)
    prompt_info = read_json(prompt_info_json) if prompt_info_json else {}
    global_prompt_info = build_global_prompt_info(prompt_info)
    subtask_prior_by_stage = build_subtask_prior_index(prompt_info)
    sampled = iter_stride_frames(episode.annotation, image_root, frame_stride)
    if not sampled:
        raise ValueError(
            "No generation frames were selected. Check that --image-root points to the frame directory "
            "containing stage_00/frame_000123.jpg files and that frame numbers overlap annotation durations."
        )
    last_request_index_by_stage: Dict[int, int] = {}
    for sample_index, (sampled_skill, _) in enumerate(sampled, start=1):
        last_request_index_by_stage[sampled_skill.stage_idx] = sample_index

    old_memory = ""
    previous_image_path: Optional[Path] = None
    records: List[JsonObject] = []
    system_instruction = prompt_catalog.get("generation_system")
    rendered_prompt_dir = get_rendered_prompt_output_dir(output_path) if save_rendered_prompts else None
    print(
        "[sampled] "
        f"episode={annotation_json.name} samples={len(sampled)} frame_stride={frame_stride} "
        "source=valid_duration_stride",
        flush=True,
    )

    current_request: Optional[JsonObject] = None
    try:
        for request_index, (skill, sample) in enumerate(sampled, start=1):
            current_request = {
                "request_index": request_index,
                "total_requests": len(sampled),
                "stage_idx": skill.stage_idx,
                "skill_idx": skill.skill_idx,
                "frame_number": sample.frame_number,
                "image_path": str(sample.image_path),
            }
            subtask_prior = subtask_prior_by_stage.get(skill.stage_idx, {})
            has_previous_image = include_previous_image and previous_image_path is not None
            completion_guidance = build_completion_guidance(global_prompt_info, subtask_prior)
            completion_gate_context = build_completion_gate_context(
                skill=skill,
                subtask_prior=subtask_prior,
                request_index=request_index,
                last_request_index=last_request_index_by_stage.get(skill.stage_idx),
            )
            prompt_values = {
                "task_name": episode.task_name,
                "old_memory": old_memory,
                "skill_description": skill.skill_description,
                "object_id": skill.object_id,
                "manuipation_object_id": skill.manuipation_object_id,
                "frame_duration": list(skill.frame_duration),
                "frame_number": sample.frame_number,
                "image_block": build_image_block(has_previous_image),
                "completion_gate_context": completion_gate_context,
                "completion_guidance": completion_guidance,
            }
            prompt = prompt_catalog.render("generation_user", prompt_values)
            image_paths = [sample.image_path]
            if has_previous_image and previous_image_path is not None:
                image_paths = [previous_image_path, sample.image_path]

            rendered_prompt_path: Optional[Path] = None
            if rendered_prompt_dir is not None:
                rendered_prompt_path = (
                    rendered_prompt_dir
                    / f"request_{request_index:06d}_stage_{skill.stage_idx:02d}_frame_{sample.frame_number:06d}.json"
                )
                write_rendered_prompt(
                    path=rendered_prompt_path,
                    request_index=request_index,
                    skill=skill,
                    sample=sample,
                    previous_image_path=previous_image_path if has_previous_image else None,
                    system_instruction=system_instruction,
                    user_prompt=prompt,
                    completion_guidance=completion_guidance,
                    prompt_values=prompt_values,
                )
                print(
                    f"[prompt-saved] json={rendered_prompt_path} markdown={rendered_prompt_path.with_suffix('.md')}",
                    flush=True,
                )

            print(
                "[generate] "
                f"sample={request_index}/{len(sampled)} stage_idx={skill.stage_idx} "
                f"frame={sample.frame_number} image={sample.image_path}",
                flush=True,
            )
            model_response: JsonObject = {}
            metadata: JsonObject = {}
            output_style_retries = 0
            generation_prompt = prompt
            for output_style_attempt in range(OUTPUT_STYLE_MAX_RETRIES + 1):
                response, metadata = gemini_client.generate_json(
                    system_instruction=system_instruction,
                    prompt=generation_prompt,
                    image_paths=image_paths,
                    required_keys=MODEL_RESPONSE_KEYS,
                )
                model_response = normalize_model_response(response)
                meta_language_hits = find_meta_rule_language(model_response)
                if not meta_language_hits:
                    break
                if output_style_attempt >= OUTPUT_STYLE_MAX_RETRIES:
                    raise ValueError(
                        "Model response contains meta-rule language in visible output fields: "
                        + "; ".join(meta_language_hits)
                    )
                output_style_retries += 1
                print(
                    "[retry] meta_rule_language_in_model_response "
                    f"attempt={output_style_attempt + 1}/{OUTPUT_STYLE_MAX_RETRIES} "
                    f"matches={meta_language_hits[:3]}",
                    flush=True,
                )
                generation_prompt = (
                    f"{prompt}\n\nRewrite the previous JSON in plain visual-observation language. "
                    "Explain only what is visible in the current image and what changed from the previous "
                    "image when one is provided. Do not cite or name any external standard, checklist, "
                    "hidden instruction, output field, or evaluation text."
                )
            request_input: JsonObject = {
                "image_path": str(sample.image_path),
                "main_task": episode.task_name,
                "old_memory": old_memory,
                "skill_description": skill.skill_description,
                "object_id": skill.object_id,
                "manuipation_object_id": skill.manuipation_object_id,
                "frame_number": sample.frame_number,
                "frame_duration": list(skill.frame_duration),
                "completion_guidance": completion_guidance,
                "completion_gate_context": completion_gate_context,
            }
            if has_previous_image and previous_image_path is not None:
                request_input["previous_image_path"] = str(previous_image_path)
            hard_completion_gate = apply_hard_completion_gates(
                response=model_response,
                skill=skill,
                subtask_prior=subtask_prior,
                request_index=request_index,
                last_request_index=last_request_index_by_stage.get(skill.stage_idx),
            )
            record: JsonObject = {
                "skill_idx": skill.skill_idx,
                "stage_idx": skill.stage_idx,
                "image_dir": sample.image_path.parent.name,
                "image_path": str(sample.image_path),
                "image_index_in_stage": sample.image_index_in_stage,
                "frame_number": sample.frame_number,
                "frame_duration": list(skill.frame_duration),
                "frame_selection": f"valid_duration_stride_{frame_stride}",
                "frame_stride": frame_stride,
                "skill_description": skill.skill.get("skill_description", ""),
                "object_id": skill.skill.get("object_id", ""),
                "manuipation_object_id": skill.manuipation_object_id,
                "request_input": request_input,
                "model_response": model_response,
                "result_used": True,
            }
            if hard_completion_gate:
                record["hard_completion_gate"] = hard_completion_gate
            if output_style_retries:
                metadata = dict(metadata)
                metadata["output_style_retries"] = output_style_retries
            if metadata:
                record["google_response_metadata"] = metadata
            if rendered_prompt_path is not None:
                record["rendered_prompt_path"] = str(rendered_prompt_path)
                record["rendered_prompt_markdown_path"] = str(rendered_prompt_path.with_suffix(".md"))
            records.append(record)

            new_memory = model_response.get("new_memory")
            if new_memory:
                old_memory = new_memory if isinstance(new_memory, str) else json.dumps(new_memory, ensure_ascii=False)
            previous_image_path = sample.image_path
            if request_delay > 0 and request_index < len(sampled):
                time.sleep(request_delay)
    except BaseException as exc:
        output = build_episode_generation_output(
            annotation_json=annotation_json,
            image_root=image_root,
            task_name=episode.task_name,
            prompt_info_json=prompt_info_json,
            rendered_prompt_dir=rendered_prompt_dir,
            frame_stride=frame_stride,
            records=records,
            expected_count=len(sampled),
            status=checkpoint_status(exc),
            error=error_to_json(exc),
            current_request=current_request,
        )
        save_checkpoint(output_path, output)
        raise

    output = build_episode_generation_output(
        annotation_json=annotation_json,
        image_root=image_root,
        task_name=episode.task_name,
        prompt_info_json=prompt_info_json,
        rendered_prompt_dir=rendered_prompt_dir,
        frame_stride=frame_stride,
        records=records,
        expected_count=len(sampled),
        status="complete",
    )
    write_json(output_path, output)
    print(f"[saved] {output_path}", flush=True)
    return output


def build_generation_aggregate(
    *,
    annotation_path: Path,
    image_root: Path,
    episode_offset: int,
    episode_limit: Optional[int],
    selected_episode_count: int,
    total_episode_count: int,
    episodes: List[JsonObject],
    status: str,
    error: Optional[JsonObject] = None,
    current_episode: str = "",
) -> JsonObject:
    aggregate: JsonObject = {
        "status": status,
        "annotation_path": str(annotation_path),
        "image_root": str(image_root),
        "episode_offset": episode_offset,
        "episode_limit": episode_limit,
        "selected_episode_count": selected_episode_count,
        "total_episode_count": total_episode_count,
        "episode_count": len(episodes),
        "processed_count": sum(int(episode.get("processed_count", 0)) for episode in episodes),
        "episodes": episodes,
    }
    if current_episode:
        aggregate["current_episode"] = current_episode
    if error is not None:
        aggregate["error"] = error
    return aggregate


def build_episode_generation_output(
    *,
    annotation_json: Path,
    image_root: Path,
    task_name: str,
    prompt_info_json: Optional[Path],
    rendered_prompt_dir: Optional[Path],
    frame_stride: int,
    records: List[JsonObject],
    expected_count: int,
    status: str,
    error: Optional[JsonObject] = None,
    current_request: Optional[JsonObject] = None,
) -> JsonObject:
    output: JsonObject = {
        "status": status,
        "annotation_json": str(annotation_json),
        "image_root": str(image_root),
        "task_name": task_name,
        "prompt_info_json": str(prompt_info_json) if prompt_info_json else "",
        "task_prior_json": str(prompt_info_json) if prompt_info_json else "",
        "rendered_prompt_dir": str(rendered_prompt_dir) if rendered_prompt_dir is not None else "",
        "frame_selection": f"valid_duration_stride_{frame_stride}",
        "frame_stride": frame_stride,
        "sample_source": "valid_duration_stride",
        "expected_count": expected_count,
        "processed_count": len(records),
        "used_count": len(records),
        "results": records,
    }
    if current_request is not None:
        output["current_request"] = current_request
    if error is not None:
        output["error"] = error
    return output


def find_meta_rule_language(response: JsonObject) -> List[str]:
    checked_fields = {
        "reasoning": response.get("reasoning"),
        "new_memory": response.get("new_memory"),
        "subtask": response.get("subtask"),
        "visible_transition": response.get("visible_transition"),
    }
    hits: List[str] = []
    for field_name, field_value in checked_fields.items():
        for text in iter_text_values(field_value):
            lowered = text.lower()
            for pattern in META_RULE_LANGUAGE_PATTERNS:
                if re.search(pattern, lowered):
                    hits.append(f"{field_name}:{pattern}")
                    break
    return hits


def iter_text_values(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        texts: List[str] = []
        for nested in value.values():
            texts.extend(iter_text_values(nested))
        return texts
    if isinstance(value, list):
        texts = []
        for item in value:
            texts.extend(iter_text_values(item))
        return texts
    return []


def normalize_model_response(response: JsonObject) -> JsonObject:
    status = response.get("current_skill_status")
    if isinstance(status, str):
        normalized = status.strip().lower().replace(" ", "_").replace("-", "_")
        response["current_skill_status"] = normalized
        if normalized == "no_for_sure":
            response["is_subtask_completed"] = False
    return response


def build_completion_gate_context(
    *,
    skill: SkillSpec,
    subtask_prior: JsonObject,
    request_index: int,
    last_request_index: Optional[int],
) -> str:
    if not is_move_to_skill(skill, subtask_prior):
        return "No extra completion gate."
    if last_request_index is None:
        return "No extra completion gate."
    if request_index < last_request_index:
        return (
            "Hard output constraint for move-to or navigation skills: this is not the final sampled "
            "request for the current move-to skill. Do not set current_skill_status to completed or "
            "completed_and_transitioning, and set is_subtask_completed to false. Use in_progress when "
            "the robot is visibly approaching or settling near the target. Use no_for_sure when the "
            "settled target interaction pose or immediate reachability is unclear."
        )
    return (
        "Hard output constraint for move-to or navigation skills: this is the final sampled request "
        "for the current move-to skill. Set current_skill_status to completed and set "
        "is_subtask_completed to true. Keep the subtask field focused on the current move-to skill, "
        "and describe the visible scene conservatively in reasoning and new_memory."
    )


def apply_hard_completion_gates(
    *,
    response: JsonObject,
    skill: SkillSpec,
    subtask_prior: JsonObject,
    request_index: int,
    last_request_index: Optional[int],
) -> Optional[JsonObject]:
    if not is_move_to_skill(skill, subtask_prior):
        return None
    if last_request_index is None:
        return None
    target_name = get_candidate_subtask_name(skill, subtask_prior)
    original_status = response.get("current_skill_status")
    original_is_completed = response.get("is_subtask_completed")

    if request_index >= last_request_index:
        if original_status in COMPLETED_STATUSES and original_is_completed is True:
            return None
        response["current_skill_status"] = "completed"
        response["is_subtask_completed"] = True
        response["new_memory"] = build_move_to_completed_memory(response, target_name)
        return {
            "rule": "move_to_completion_fixed_by_last_request",
            "reason": (
                "Move-to or navigation skills are deterministically marked completed on the last "
                "sampled request for that skill."
            ),
            "original_current_skill_status": original_status,
            "original_is_subtask_completed": original_is_completed,
            "forced_current_skill_status": "completed",
            "forced_is_subtask_completed": True,
            "request_index": request_index,
            "last_request_index_for_stage": last_request_index,
        }

    if original_status not in COMPLETED_STATUSES and original_is_completed is not True:
        return None

    response["current_skill_status"] = "in_progress"
    response["is_subtask_completed"] = False
    response["visible_transition"] = ""
    response["new_memory"] = build_move_to_hard_gate_memory(response, target_name)
    return {
        "rule": "move_to_completion_only_allowed_on_last_request",
        "reason": (
            "Move-to or navigation skills may only output completed on the last sampled request "
            "for that skill."
        ),
        "original_current_skill_status": original_status,
        "original_is_subtask_completed": original_is_completed,
        "forced_current_skill_status": "in_progress",
        "forced_is_subtask_completed": False,
        "request_index": request_index,
        "last_request_index_for_stage": last_request_index,
    }


def build_move_to_hard_gate_memory(response: JsonObject, target_name: str) -> JsonObject:
    world_state = ""
    existing_memory = response.get("new_memory")
    if isinstance(existing_memory, dict):
        world_state = str(existing_memory.get("World state", "")).strip()
    if not world_state:
        world_state = (
            "The robot is near the target area, but the settled interaction pose and immediate "
            "reachability are being judged conservatively."
        )
    return {
        "Progress": (
            f"The robot is still moving toward or settling near {target_name}. "
            "The move-to subtask is not recorded as completed yet."
        ),
        "World state": world_state,
    }


def build_move_to_completed_memory(response: JsonObject, target_name: str) -> JsonObject:
    world_state = ""
    existing_memory = response.get("new_memory")
    if isinstance(existing_memory, dict):
        world_state = str(existing_memory.get("World state", "")).strip()
    if not world_state:
        world_state = "The robot is at the target interaction location for the move-to subtask."
    return {
        "Progress": f"Moved to {target_name}.",
        "World state": world_state,
    }


def is_move_to_skill(skill: SkillSpec, subtask_prior: JsonObject) -> bool:
    child_prior = subtask_prior.get("child_prior")
    child = child_prior if isinstance(child_prior, dict) else {}
    candidates = [
        skill.skill_description,
        skill.skill.get("skill_description"),
        child.get("skill_description"),
        child.get("skill_type_hypothesis"),
        child.get("subtask_name"),
    ]
    text = " ".join(normalize_action_text(value) for value in candidates)
    return any(pattern in text for pattern in ("move to", "navigate to", "go to"))


def normalize_action_text(value: object) -> str:
    normalized = normalize_skill_label(value).lower()
    return normalized.replace("_", " ").replace("-", " ")


def get_candidate_subtask_name(skill: SkillSpec, subtask_prior: JsonObject) -> str:
    child_prior = subtask_prior.get("child_prior")
    child = child_prior if isinstance(child_prior, dict) else {}
    return first_text_value(
        [
            child.get("subtask_name"),
            skill.skill.get("subtask_name"),
            normalize_skill_label(skill.skill_description),
            "the target interaction pose",
        ]
    )


def get_rendered_prompt_output_dir(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_prompts"


def write_rendered_prompt(
    *,
    path: Path,
    request_index: int,
    skill: SkillSpec,
    sample: SampledFrame,
    previous_image_path: Optional[Path],
    system_instruction: str,
    user_prompt: str,
    completion_guidance: str,
    prompt_values: JsonObject,
) -> None:
    markdown_path = path.with_suffix(".md")
    payload: JsonObject = {
        "request_index": request_index,
        "skill_idx": skill.skill_idx,
        "stage_idx": skill.stage_idx,
        "frame_number": sample.frame_number,
        "image_path": str(sample.image_path),
        "previous_image_path": str(previous_image_path) if previous_image_path is not None else "",
        "skill_description": skill.skill_description,
        "object_id": skill.object_id,
        "manuipation_object_id": skill.manuipation_object_id,
        "system_instruction": system_instruction,
        "user_prompt": user_prompt,
        "completion_guidance": completion_guidance,
        "prompt_values": prompt_values,
        "markdown_path": str(markdown_path),
    }
    write_json(path, payload)
    write_text(markdown_path, build_rendered_prompt_markdown(payload))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_rendered_prompt_markdown(payload: JsonObject) -> str:
    prompt_values = payload.get("prompt_values")
    if not isinstance(prompt_values, dict):
        prompt_values = {}
    metadata = {
        "request_index": payload.get("request_index"),
        "stage_idx": payload.get("stage_idx"),
        "skill_idx": payload.get("skill_idx"),
        "frame_number": payload.get("frame_number"),
        "image_path": payload.get("image_path"),
        "previous_image_path": payload.get("previous_image_path"),
    }
    return "\n".join(
        [
            "# Rendered Generation Prompt",
            "",
            "## Request Metadata",
            "",
            "```json",
            json.dumps(metadata, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Prompt Values Injected Into Template",
            "",
            "```json",
            json.dumps(prompt_values, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Completion Guidance Injected As Rule 16",
            "",
            "```text",
            str(payload.get("completion_guidance", "")),
            "```",
            "",
            "## System Instruction",
            "",
            "```text",
            str(payload.get("system_instruction", "")),
            "```",
            "",
            "## Full User Prompt",
            "",
            "```text",
            str(payload.get("user_prompt", "")),
            "```",
            "",
        ]
    )


def build_subtask_prior_index(task_prior: JsonObject) -> Dict[int, JsonObject]:
    child_index: Dict[int, JsonObject] = {}
    subtasks = task_prior.get("subtask_priors")
    if not isinstance(subtasks, list):
        subtasks = task_prior.get("skills")
    if isinstance(subtasks, list):
        for item in subtasks:
            if not isinstance(item, dict):
                continue
            stage_idx = item.get("stage_idx")
            if isinstance(stage_idx, int):
                child_index[stage_idx] = compact_skill_prior(item)

    merged: Dict[int, JsonObject] = {}
    for stage_idx, child_prior in child_index.items():
        merged[stage_idx] = {"child_prior": child_prior}
    return merged


def build_global_prompt_info(task_prior: JsonObject) -> JsonObject:
    model_response = task_prior.get("model_response")
    source = model_response if isinstance(model_response, dict) else task_prior
    compact: JsonObject = {}
    for key in ("task_name", "prior_min_items"):
        if key in task_prior:
            compact[key] = task_prior[key]
    for key in GLOBAL_PRIOR_KEYS:
        value = source.get(key)
        if has_prompt_value(value):
            compact[key] = value
    return compact


def compact_skill_prior(skill_prior: JsonObject) -> JsonObject:
    compact: JsonObject = {}
    for key in SKILL_PRIOR_KEYS:
        value = skill_prior.get(key)
        if has_prompt_value(value):
            compact[key] = value
    return compact


def has_prompt_value(value: object) -> bool:
    return value not in (None, "", [], {})


def build_image_block(has_previous_image: bool) -> str:
    if has_previous_image:
        return (
            "Images:\n"
            "- Previous observation image: <previous_image>\n"
            "- Current observation image: <current_image>\n\n"
            "Use the previous observation image only as temporal context. "
            "The current observation image is the one to label."
        )
    return "Image:\n<current_image>"


def build_completion_guidance(global_prompt_info: JsonObject, subtask_prior: JsonObject) -> str:
    lines: List[str] = []
    task_context = render_task_context(global_prompt_info)
    if task_context:
        lines.append(task_context)

    child_prior = subtask_prior.get("child_prior")

    if isinstance(child_prior, dict) and child_prior:
        if lines:
            lines.append("")
        natural_guidance = first_text_value(
            [
                child_prior.get("generation_prompt_guidance"),
                child_prior.get("completion_guidance"),
                child_prior.get("prompt_guidance"),
            ]
        )
        if natural_guidance:
            lines.append("Primary child-agent visual postcondition notes for the current candidate skill:")
            lines.append(sanitize_visual_guidance_text(natural_guidance))
        else:
            lines.append(render_skill_prior_as_natural_guidance(child_prior))
        child_state_guidance = render_child_structured_guardrails(child_prior)
        if child_state_guidance:
            lines.append("")
            lines.append("Child-agent structured visual state notes:")
            lines.append(child_state_guidance)
        action_guardrails = render_action_primitive_generation_guardrails(child_prior)
        if action_guardrails:
            lines.append("")
            lines.append("Generic action primitive visual notes:")
            lines.append(action_guardrails)
    else:
        if lines:
            lines.append("")
        lines.append(
            "No task-specific prior is available for this candidate skill. Use the generic visible "
            "postcondition rules above and stay conservative when the decisive visual evidence is missing."
        )

    lines.append("")
    lines.append(
        "This section describes what visual facts to look for; it is not visual evidence. "
        "The current image and the previous memory still decide the label. "
        "Mark the candidate skill completed only when every decisive completion gate is directly visible "
        "in the current image. If the current image still matches a before-completion state, an in-progress "
        "state, or an insufficient-progress pattern, do not mark completed. If the decisive result state is "
        "occluded, color-ambiguous, hidden by the robot, or not directly visible, use no_for_sure instead of "
        "completed. Contact, hovering, pressing motion, support, or gripper movement cannot substitute for "
        "the required visible result state."
    )
    return "\n".join(lines)


def render_action_primitive_generation_guardrails(child_prior: JsonObject) -> str:
    if is_object_acquisition_prior(child_prior):
        return (
            "For pick-up, grasp, lift, or object-acquisition labeling, interpret all completion gates "
            "strictly as direct object-body evidence. The candidate skill can be completed only when the "
            "current image clearly shows the robot gripper controlling the target object body itself and "
            "the target object body no longer being supported by the original surface, container, floor, "
            "or fixture. The decisive evidence must come from the real object-support boundary or from "
            "the whole target body being visibly carried in free space, not from indirect cues. For every "
            "target object, top contact, rim contact, handle contact, side contact, edge contact, squeezing, "
            "deforming, or partial enclosure is still not enough; the lower body, bottom, or support/contact "
            "boundary must be visibly separated from the original support, or the whole target body must be "
            "visibly carried. "
            "On glossy, glass, transparent, reflective, metallic, dark, or high-glare surfaces, do not use "
            "shadows, reflections, glare, highlights, mirrored object copies, dark seams, perspective "
            "distortion, or a tiny apparent gap as evidence that the object has left the surface. Do not "
            "claim that a clear gap is visible unless the real object bottom/lower body and the real support "
            "surface boundary are both visible and unobscured. If the support/contact boundary, underside, "
            "lower body, or lift-off region is hidden by the robot gripper, the target object, a countertop "
            "edge, a container wall, glare, reflection, blur, or viewpoint, use no_for_sure instead of "
            "completed. Treat reaching, hovering, touching, closing the gripper, handle-only contact, rim-only "
            "contact, top-edge contact, side contact, partial occlusion, leaning, rotating, sliding, or an "
            "unchanged support region as in_progress or no_for_sure, not completed."
        )
    if is_state_change_prior(child_prior):
        return (
            "For press, toggle, switch, turn-on, turn-off, or state-change labeling, completion requires "
            "the exact target part to be directly visible in the current image with the required final "
            "state. The same target button, switch, indicator, display, or movable part named by the "
            "current skill must show the final color, light, depression, pose, aperture, or on/off state. "
            "Robot contact, a pressing motion, the gripper still touching the target, a task sequence, or "
            "previous memory is not enough. For color or light changes, do not mark completed when the "
            "target part is covered by the gripper or only partly visible, when the color is changing, "
            "mixed, dim, saturated, shadowed, or visible only as a tiny exposed edge, or when the apparent "
            "final color could come from a nearby light, reflection, colored mark, wall signal, stove "
            "control, or other non-target part. If the target state region is occluded, cropped, blurred, "
            "reflective, glare-covered, or ambiguous, use no_for_sure instead of completed. Do not claim "
            "a red-to-green, off-to-on, open-to-closed, or similar transition unless the current image "
            "clearly shows the same target part in the final state."
        )
    if is_open_close_prior(child_prior):
        return (
            "For open or close labeling, bind completion to the movable part itself: the door, lid, "
            "drawer, panel, flap, cover, hinge side, handle side, or aperture named by the current skill. "
            "The candidate skill can be completed only when the current image clearly shows the same "
            "movable part in the required final geometry. Treat this final geometry as the primary gate: "
            "evaluate release/no-contact only after the geometry gate is satisfied. For open, this means "
            "the door/lid/panel is visibly separated from its frame and the aperture angle clearly shows "
            "the required open state. If the skill or child prior says fully open, open completely, or open "
            "wide enough for access, require a large, stable opening that gives unobstructed access to the "
            "interior or target region. If the movable part still overlaps, covers, blocks, or narrows a "
            "large part of the front opening or access region, keep the skill in_progress or no_for_sure "
            "even when the gripper has released it. A small crack, mechanism activation, medium-angle "
            "opening, or partially open pose is still not completed. Seeing an object or interior through "
            "a transparent door, window, reflection, front panel, or partial gap is not evidence that the "
            "door/lid/panel has reached the required open geometry. For close, this means the movable part "
            "is visibly flush or aligned with the frame and the opening is no longer visible. Robot release, "
            "gripper retraction, or lack of contact is necessary only when the geometry gate already holds; "
            "release cannot compensate for insufficient open/closed geometry. Robot contact with a handle, "
            "edge, door face, side panel, control panel, nearby button, switch, indicator, light, knob, or "
            "colored mark is not completion evidence unless the same movable part's open/closed geometry "
            "is visible. Do not reinterpret an open/close skill as pressing a button or turning on a device "
            "just because a button, switch, indicator, or light appears near the robot gripper. If the door "
            "edge, hinge, aperture, handle-side gap, actual opening, or closed seam is occluded, cropped, "
            "reflective, blurred, or ambiguous, use no_for_sure instead of completed."
        )
    return ""


def is_object_acquisition_prior(child_prior: JsonObject) -> bool:
    candidates = [
        child_prior.get("skill_description"),
        child_prior.get("skill_type_hypothesis"),
        child_prior.get("subtask_name"),
    ]
    text = " ".join(normalize_action_text(value) for value in candidates)
    patterns = (
        "object acquisition",
        "pick up",
        "pickup",
        "pick ",
        "grasp",
        "lift",
        "take ",
    )
    return any(pattern in text for pattern in patterns)


def is_state_change_prior(child_prior: JsonObject) -> bool:
    candidates = [
        child_prior.get("skill_description"),
        child_prior.get("skill_type_hypothesis"),
        child_prior.get("subtask_name"),
    ]
    text = " ".join(normalize_action_text(value) for value in candidates)
    patterns = (
        "state change",
        "state change press toggle",
        "press",
        "toggle",
        "switch",
        "turn on",
        "turn off",
    )
    return any(pattern in text for pattern in patterns)


def is_open_close_prior(child_prior: JsonObject) -> bool:
    skill_type = normalize_action_text(child_prior.get("skill_type_hypothesis"))
    if "open close" in skill_type:
        return True
    candidates = [
        child_prior.get("skill_description"),
        child_prior.get("subtask_name"),
    ]
    text = f" {' '.join(normalize_action_text(value) for value in candidates)} "
    if " close to " in text or " move close " in text:
        return False
    patterns = (
        " open ",
        " opening ",
        " close ",
        " closing ",
        " shut ",
    )
    return any(pattern in text for pattern in patterns)


def render_child_structured_guardrails(child_prior: JsonObject) -> str:
    lines: List[str] = []
    action_change_items = normalize_guidance_items(child_prior.get("action_change_process"))
    if action_change_items:
        lines.append(
            "Observed atomic action-change process from the prior stage: "
            f"{join_guidance_items(action_change_items)}."
        )
    for value, label in (
        (
            child_prior.get("pre_completion_state"),
            "Visible states before completion; if the current image matches any of these, do not mark completed",
        ),
        (
            child_prior.get("in_progress_state"),
            "Visible in-progress states that are not completed",
        ),
        (
            child_prior.get("completion_gates"),
            "Decisive completion gates; all must be directly visible as AND conditions before completed is allowed",
        ),
        (
            child_prior.get("not_sufficient_for_completion"),
            "Insufficient progress patterns; these must stay in_progress or no_for_sure, not completed",
        ),
    ):
        items = normalize_guidance_items(value)
        if items:
            lines.append(f"{label}: {join_guidance_items(items)}.")
    return "\n".join(lines)


def render_task_context(global_prompt_info: JsonObject) -> str:
    sentences: List[str] = []
    task_summary = global_prompt_info.get("task_summary")
    if has_prompt_value(task_summary):
        sentences.append(f"At the task level, the episode goal is {strip_terminal_period(str(task_summary))}.")
    return "\n".join(sentences)


def render_skill_prior_as_natural_guidance(skill_prior: JsonObject) -> str:
    sentences: List[str] = []
    skill_label = normalize_skill_label(skill_prior.get("skill_description"))
    subtask_name = first_text_value([skill_prior.get("subtask_name")])
    target_description = describe_target_naturally(skill_prior.get("target_visual_description"))
    if skill_label and subtask_name and target_description:
        sentences.append(
            f"For the current candidate \"{skill_label}\" skill, judge whether the robot has completed "
            f"the subtask: {subtask_name}. "
            f"The visual target is {target_description}."
        )
    elif skill_label and subtask_name:
        sentences.append(
            f"For the current candidate \"{skill_label}\" skill, judge whether the robot has completed "
            f"the subtask: {subtask_name}."
        )
    elif skill_label and target_description:
        sentences.append(
            f"For the current candidate \"{skill_label}\" skill, the visual target is {target_description}."
        )
    elif skill_label:
        sentences.append(f"For the current candidate \"{skill_label}\" skill, use the visible postcondition rules below.")
    elif subtask_name and target_description:
        sentences.append(
            f"For this candidate skill, judge whether the robot has completed the subtask: {subtask_name}. "
            f"The visual target is {target_description}."
        )
    elif subtask_name:
        sentences.append(f"For this candidate skill, judge whether the robot has completed the subtask: {subtask_name}.")
    elif target_description:
        sentences.append(f"For this candidate skill, the visual target is {target_description}.")

    completion_items = normalize_guidance_items(skill_prior.get("completion_conditions"))
    completion_gates = normalize_guidance_items(skill_prior.get("completion_gates"))
    if completion_gates:
        sentences.append(
            "Require all decisive completion gates to be visible: "
            f"{join_guidance_items(completion_gates)}."
        )
    if completion_items:
        sentences.append(
            "Treat the skill as completed only when "
            f"{join_guidance_items(completion_items)}."
        )

    in_progress_items = normalize_guidance_items(skill_prior.get("in_progress_state"))
    if in_progress_items:
        sentences.append(f"Treat the skill as still in progress when {join_guidance_items(in_progress_items)}.")

    evidence_items = normalize_guidance_items(skill_prior.get("required_visual_evidence"))
    if evidence_items:
        sentences.append(f"The current image should visibly support this with {join_guidance_items(evidence_items)}.")

    transition_items = normalize_guidance_items(skill_prior.get("state_transition_evidence"))
    action_change_items = normalize_guidance_items(skill_prior.get("action_change_process"))
    if action_change_items:
        sentences.append(
            "The prior-stage action-change process for this atomic skill is "
            f"{join_guidance_items(action_change_items)}."
        )
    if transition_items:
        sentences.append(
            "When temporal context is available, useful transition evidence includes "
            f"{join_guidance_items(transition_items)}. Do not claim a transition unless the current image "
            "shows the final state clearly."
        )

    negative_items = normalize_guidance_items(skill_prior.get("negative_conditions"))
    if negative_items:
        sentences.append(f"Keep the skill not completed when {join_guidance_items(negative_items)}.")

    insufficient_items = normalize_guidance_items(skill_prior.get("not_sufficient_for_completion"))
    if insufficient_items:
        sentences.append(
            "Do not treat insufficient progress as completion, including "
            f"{join_guidance_items(insufficient_items)}."
        )

    false_positive_items = normalize_guidance_items(skill_prior.get("common_false_positives"))
    if false_positive_items:
        sentences.append(
            "Do not mark completion for look-alikes or insufficient evidence such as "
            f"{join_guidance_items(false_positive_items)}."
        )

    ambiguous_items = normalize_guidance_items(skill_prior.get("ambiguous_cases"))
    if ambiguous_items:
        sentences.append(f"Use no_for_sure when {join_guidance_items(ambiguous_items)}.")

    if not sentences:
        return (
            "Use the generic visible postcondition rules for this skill and stay conservative when the "
            "decisive target state, contact, support, release, or color evidence is not visible."
        )
    return "\n".join(sentences)


def describe_target_naturally(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    target_object = first_text_value([value.get("target_object")])
    target_part = first_text_value([value.get("target_part")])
    normalized_part = target_part.lower()
    if target_object and normalized_part.startswith("whole "):
        target = f"the whole {target_object}"
    elif target_object and target_part and normalized_part == target_object.lower():
        target = f"the {target_object}"
    elif target_object and target_part:
        target = f"the {target_part} of the {target_object}"
    else:
        target = target_part or target_object or "the target object or part"

    attributes: List[str] = []
    for key, phrase in (
        ("color", "its visible color or state is {}"),
        ("shape", "it has a {} shape or outline"),
        ("position", "it is located {}"),
        ("size", "its visible size is {}"),
        ("count", "the relevant count is {}"),
    ):
        field_value = value.get(key)
        if has_prompt_value(field_value):
            attributes.append(phrase.format(strip_terminal_period(str(field_value))))
    if attributes:
        return f"{target}; {join_guidance_items(attributes)}"
    return target


def normalize_skill_label(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text[0] in "[\"'":
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, list):
            return ", ".join(str(item).strip() for item in parsed if str(item).strip())
        if isinstance(parsed, str):
            return parsed.strip()
    return text.strip("\"'")


def first_text_value(values) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def join_guidance_items(items: List[str]) -> str:
    cleaned = [strip_terminal_period(item) for item in items if strip_terminal_period(item)]
    return "; ".join(cleaned)


def strip_terminal_period(text: str) -> str:
    return text.strip().rstrip(".")


def normalize_guidance_items(value: object) -> List[str]:
    if isinstance(value, list):
        items: List[str] = []
        for item in value:
            cleaned = sanitize_visual_guidance_text(str(item).strip())
            if cleaned:
                items.append(cleaned)
        return items
    if isinstance(value, str) and value.strip():
        return [sanitize_visual_guidance_text(value.strip())]
    return []


def is_complete_generation_output(output: object) -> bool:
    if not isinstance(output, dict) or not isinstance(output.get("results"), list):
        return False
    if output.get("status") in {"failed", "interrupted"}:
        return False
    results = output.get("results", [])
    if int(output.get("processed_count", 0)) != len(results):
        return False
    expected_count = output.get("expected_count")
    if isinstance(expected_count, int) and expected_count != len(results):
        return False
    return True


def resolve_prompt_info_path(
    *,
    annotation_json: Path,
    explicit_prompt_info_json: Optional[Path],
    prompt_info_root: Optional[Path],
    multiple_episodes: bool,
) -> Optional[Path]:
    if explicit_prompt_info_json is not None:
        return explicit_prompt_info_json
    if prompt_info_root is None:
        return None

    candidate_dirs = []
    if multiple_episodes:
        candidate_dirs.append(prompt_info_root / annotation_json.stem)
    else:
        candidate_dirs.append(prompt_info_root)

    candidate_names = (
        "autolabel_prompt_info.json",
        "prompt_info.json",
        "task_prior.json",
        f"{annotation_json.stem}.json",
    )
    for candidate_dir in candidate_dirs:
        for name in candidate_names:
            candidate = candidate_dir / name
            if candidate.exists():
                return candidate
    return None


def resolve_task_prior_path(
    *,
    annotation_json: Path,
    explicit_task_prior_json: Optional[Path],
    prior_root: Optional[Path],
    multiple_episodes: bool,
) -> Optional[Path]:
    return resolve_prompt_info_path(
        annotation_json=annotation_json,
        explicit_prompt_info_json=explicit_task_prior_json,
        prompt_info_root=prior_root,
        multiple_episodes=multiple_episodes,
    )
