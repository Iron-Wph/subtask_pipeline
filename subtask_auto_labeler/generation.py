import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from .dataset import (
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
) -> JsonObject:
    episode = load_episode(annotation_json, image_root)
    prompt_info = read_json(prompt_info_json) if prompt_info_json else {}
    subtask_prior_by_stage = build_subtask_prior_index(prompt_info)
    sampled = iter_stride_frames(episode.annotation, image_root, frame_stride)
    old_memory = ""
    previous_image_path: Optional[Path] = None
    records: List[JsonObject] = []
    system_instruction = prompt_catalog.get("generation_system")

    for request_index, (skill, sample) in enumerate(sampled, start=1):
        subtask_prior = subtask_prior_by_stage.get(skill.stage_idx, {})
        prompt = prompt_catalog.render(
            "generation_user",
            {
                "task_name": episode.task_name,
                "old_memory": old_memory,
                "skill_description": skill.skill_description,
                "object_id": skill.object_id,
                "manuipation_object_id": skill.manuipation_object_id,
                "frame_duration": list(skill.frame_duration),
                "frame_number": sample.frame_number,
                "task_prior_json": json.dumps(prompt_info, ensure_ascii=False, indent=2),
                "prompt_info_json": json.dumps(prompt_info, ensure_ascii=False, indent=2),
                "subtask_prior_json": json.dumps(subtask_prior, ensure_ascii=False, indent=2),
            },
        )
        image_paths = [sample.image_path]
        if include_previous_image and previous_image_path is not None:
            image_paths = [previous_image_path, sample.image_path]

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
        }
        if include_previous_image and previous_image_path is not None:
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
                child_index[stage_idx] = item
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
                    adjusted_index[stage_idx] = item

    merged: Dict[int, JsonObject] = {}
    for stage_idx, child_prior in child_index.items():
        merged[stage_idx] = {"child_prior": child_prior}
        if stage_idx in adjusted_index:
            merged[stage_idx]["parent_adjusted_prior"] = adjusted_index[stage_idx]
    for stage_idx, adjusted_prior in adjusted_index.items():
        merged.setdefault(stage_idx, {})["parent_adjusted_prior"] = adjusted_prior
    return merged


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
