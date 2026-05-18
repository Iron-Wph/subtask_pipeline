import json
import time
from pathlib import Path
from typing import Dict, List, Optional

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
    "global_completion_order",
    "global_visual_adjustments",
    "cross_subtask_false_positive_risks",
)
SKILL_PRIOR_KEYS = (
    "stage_idx",
    "skill_idx",
    "skill_description",
    "subtask_name",
    "target_visual_description",
    "completion_conditions",
    "required_visual_evidence",
    "state_transition_evidence",
    "negative_conditions",
    "common_false_positives",
    "ambiguous_cases",
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
    for annotation_json in annotation_jsons:
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

    aggregate = {
        "annotation_path": str(annotation_path),
        "image_root": str(image_root),
        "episode_offset": episode_offset,
        "episode_limit": episode_limit,
        "selected_episode_count": len(annotation_jsons),
        "total_episode_count": total_episode_count,
        "episode_count": len(episodes),
        "processed_count": sum(int(episode.get("processed_count", 0)) for episode in episodes),
        "episodes": episodes,
    }
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
    old_memory = ""
    previous_image_path: Optional[Path] = None
    records: List[JsonObject] = []
    system_instruction = prompt_catalog.get("generation_system")
    rendered_prompt_dir = get_rendered_prompt_output_dir(output_path) if save_rendered_prompts else None

    for request_index, (skill, sample) in enumerate(sampled, start=1):
        subtask_prior = subtask_prior_by_stage.get(skill.stage_idx, {})
        has_previous_image = include_previous_image and previous_image_path is not None
        completion_guidance = build_completion_guidance(global_prompt_info, subtask_prior)
        prompt_values = {
            "task_name": episode.task_name,
            "old_memory": old_memory,
            "skill_description": skill.skill_description,
            "object_id": skill.object_id,
            "manuipation_object_id": skill.manuipation_object_id,
            "frame_duration": list(skill.frame_duration),
            "frame_number": sample.frame_number,
            "image_block": build_image_block(has_previous_image),
            "completion_guidance": completion_guidance,
        }
        prompt = append_optional_prompt(
            prompt_catalog.render("generation_user", prompt_values),
            prompt_catalog.render_optional("target_consistency_rules", prompt_values),
        )
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
            )
            print(f"[prompt-saved] {rendered_prompt_path}", flush=True)

        print(
            "[generate] "
            f"sample={request_index}/{len(sampled)} stage_idx={skill.stage_idx} "
            f"frame={sample.frame_number} image={sample.image_path}",
            flush=True,
        )
        response, metadata = gemini_client.generate_json(
            system_instruction=system_instruction,
            prompt=prompt,
            image_paths=image_paths,
            required_keys=MODEL_RESPONSE_KEYS,
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
        }
        if has_previous_image and previous_image_path is not None:
            request_input["previous_image_path"] = str(previous_image_path)
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
            "model_response": normalize_model_response(response),
            "result_used": True,
        }
        if metadata:
            record["google_response_metadata"] = metadata
        if rendered_prompt_path is not None:
            record["rendered_prompt_path"] = str(rendered_prompt_path)
        records.append(record)

        new_memory = response.get("new_memory")
        if new_memory:
            old_memory = new_memory if isinstance(new_memory, str) else json.dumps(new_memory, ensure_ascii=False)
        previous_image_path = sample.image_path
        if request_delay > 0 and request_index < len(sampled):
            time.sleep(request_delay)

    output = {
        "annotation_json": str(annotation_json),
        "image_root": str(image_root),
        "task_name": episode.task_name,
        "prompt_info_json": str(prompt_info_json) if prompt_info_json else "",
        "task_prior_json": str(prompt_info_json) if prompt_info_json else "",
        "rendered_prompt_dir": str(rendered_prompt_dir) if rendered_prompt_dir is not None else "",
        "frame_selection": f"valid_duration_stride_{frame_stride}",
        "processed_count": len(records),
        "used_count": len(records),
        "results": records,
    }
    write_json(output_path, output)
    print(f"[saved] {output_path}", flush=True)
    return output


def normalize_model_response(response: JsonObject) -> JsonObject:
    status = response.get("current_skill_status")
    if isinstance(status, str):
        normalized = status.strip().lower().replace(" ", "_").replace("-", "_")
        response["current_skill_status"] = normalized
        if normalized == "no_for_sure":
            response["is_subtask_completed"] = False
    return response


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
) -> None:
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
    }
    write_json(path, payload)


def build_subtask_prior_index(task_prior: JsonObject) -> Dict[int, JsonObject]:
    child_index: Dict[int, JsonObject] = {}
    child_stage_by_skill_idx: Dict[int, int] = {}
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
                skill_idx = item.get("skill_idx")
                if isinstance(skill_idx, int):
                    child_stage_by_skill_idx[skill_idx] = stage_idx

    adjusted_index: Dict[int, JsonObject] = {}
    model_response = task_prior.get("model_response")
    if isinstance(model_response, dict):
        adjusted = model_response.get("skills")
        if not isinstance(adjusted, list):
            adjusted = model_response.get("subtask_priors")
        if isinstance(adjusted, list):
            for item in adjusted:
                if not isinstance(item, dict):
                    continue
                stage_idx = item.get("stage_idx")
                if not isinstance(stage_idx, int):
                    skill_idx = item.get("skill_idx")
                    if isinstance(skill_idx, int):
                        stage_idx = child_stage_by_skill_idx.get(skill_idx, skill_idx)
                if isinstance(stage_idx, int):
                    adjusted_index[stage_idx] = compact_skill_prior(item)

    merged: Dict[int, JsonObject] = {}
    for stage_idx, child_prior in child_index.items():
        merged[stage_idx] = {"child_prior": child_prior}
        if stage_idx in adjusted_index:
            merged[stage_idx]["parent_adjusted_prior"] = adjusted_index[stage_idx]
    for stage_idx, adjusted_prior in adjusted_index.items():
        merged.setdefault(stage_idx, {})["parent_adjusted_prior"] = adjusted_prior
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
    if global_prompt_info:
        lines.append("Task-level guidance:")
        append_named_value(lines, "Task summary", global_prompt_info.get("task_summary"))
        append_named_items(lines, "Expected completion order", global_prompt_info.get("global_completion_order"))
        append_named_items(lines, "Global visual adjustments", global_prompt_info.get("global_visual_adjustments"))
        append_named_items(
            lines,
            "Cross-subtask false-positive risks",
            global_prompt_info.get("cross_subtask_false_positive_risks"),
        )

    parent_prior = subtask_prior.get("parent_adjusted_prior")
    child_prior = subtask_prior.get("child_prior")
    primary_prior = parent_prior if isinstance(parent_prior, dict) else child_prior
    secondary_prior = child_prior if isinstance(parent_prior, dict) and isinstance(child_prior, dict) else None

    if isinstance(primary_prior, dict) and primary_prior:
        if lines:
            lines.append("")
        lines.append("Current-skill visible postconditions:")
        append_skill_guidance(lines, primary_prior)
    else:
        if lines:
            lines.append("")
        lines.append("Current-skill visible postconditions:")
        lines.append("- No autolabel prior is available for this skill; use the generic visible postcondition rules.")

    if isinstance(secondary_prior, dict) and secondary_prior:
        child_detail_lines: List[str] = []
        append_named_items(child_detail_lines, "Child-agent completion details", secondary_prior.get("completion_conditions"))
        append_named_items(child_detail_lines, "Child-agent required visual evidence", secondary_prior.get("required_visual_evidence"))
        append_named_items(child_detail_lines, "Child-agent state transitions", secondary_prior.get("state_transition_evidence"))
        if child_detail_lines:
            lines.append("")
            lines.append("Additional child-agent visual details to preserve when they do not conflict with parent guidance:")
            lines.extend(child_detail_lines)

    lines.append("")
    lines.append(
        "Use these task-specific postconditions in place of hard-coded task rules. "
        "They are guidance, not visual evidence; the current image still decides the label."
    )
    return "\n".join(lines)


def append_skill_guidance(lines: List[str], skill_prior: JsonObject) -> None:
    append_named_value(lines, "Subtask", skill_prior.get("subtask_name"))
    target_description = describe_target(skill_prior.get("target_visual_description"))
    append_named_value(lines, "Target", target_description)
    append_named_items(lines, "Mark completed only when", skill_prior.get("completion_conditions"))
    append_named_items(lines, "Required visible evidence", skill_prior.get("required_visual_evidence"))
    append_named_items(lines, "Visible state transitions to look for", skill_prior.get("state_transition_evidence"))
    append_named_items(lines, "Keep not completed when", skill_prior.get("negative_conditions"))
    append_named_items(lines, "Do not count these false positives as completed", skill_prior.get("common_false_positives"))
    append_named_items(lines, "Use no_for_sure for ambiguous cases", skill_prior.get("ambiguous_cases"))


def describe_target(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    pieces = []
    for key, label in (
        ("target_object", "object"),
        ("target_part", "part"),
        ("color", "color/state"),
        ("shape", "shape"),
        ("position", "position"),
        ("size", "size"),
        ("count", "count"),
    ):
        field_value = value.get(key)
        if has_prompt_value(field_value):
            pieces.append(f"{label}: {field_value}")
    return "; ".join(str(piece) for piece in pieces)


def append_named_value(lines: List[str], label: str, value: object) -> None:
    if has_prompt_value(value):
        lines.append(f"- {label}: {value}")


def append_named_items(lines: List[str], label: str, value: object) -> None:
    items = normalize_guidance_items(value)
    if not items:
        return
    lines.append(f"- {label}:")
    for item in items:
        lines.append(f"  - {item}")


def normalize_guidance_items(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def append_optional_prompt(prompt: str, optional_prompt: str) -> str:
    if not optional_prompt.strip():
        return prompt
    return f"{prompt}\n\n{optional_prompt.strip()}"


def is_complete_generation_output(output: object) -> bool:
    return (
        isinstance(output, dict)
        and isinstance(output.get("results"), list)
        and int(output.get("processed_count", 0)) == len(output.get("results", []))
    )


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
