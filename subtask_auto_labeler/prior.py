import json
import time
from pathlib import Path
from typing import List

from .dataset import EpisodeData, SkillSpec, load_episode, sample_subtask_images
from .gemini_client import GeminiClient
from .io_utils import JsonObject, write_json
from .prompts import PromptCatalog

SUBTASK_FRAME_KEYS = {
    "frame_reasoning",
    "subtask_name",
    "target_visual_description",
    "completion_conditions",
    "required_visual_evidence",
    "state_transition_evidence",
    "negative_conditions",
    "common_false_positives",
    "ambiguous_cases",
    "status_hint",
}
SUBTASK_SUMMARY_KEYS = {
    "subtask_name",
    "target_visual_description",
    "completion_conditions",
    "required_visual_evidence",
    "state_transition_evidence",
    "negative_conditions",
    "common_false_positives",
    "ambiguous_cases",
}
PARENT_PRIOR_KEYS = {
    "task_summary",
    "global_completion_order",
    "global_visual_adjustments",
    "cross_subtask_false_positive_risks",
    "skills",
}
DEFAULT_PRIOR_MIN_ITEMS = 4


def run_prior_pipeline(
    *,
    annotation_json: Path,
    image_root: Path,
    output_dir: Path,
    prompt_catalog: PromptCatalog,
    gemini_client: GeminiClient,
    k: int = 10,
    prior_min_items: int = DEFAULT_PRIOR_MIN_ITEMS,
    request_delay: float = 0.0,
) -> JsonObject:
    if prior_min_items < 1:
        raise ValueError("prior_min_items must be at least 1.")
    episode = load_episode(annotation_json, image_root)
    if not episode.task_name:
        raise ValueError("Annotation JSON must provide task_name, main_task, or task_description.")

    subtask_dir = output_dir / "subtasks"
    subtask_results: List[JsonObject] = []
    for skill in episode.skills:
        result = run_subtask_prior(
            episode=episode,
            skill=skill,
            output_path=subtask_dir / f"subtask_{skill.stage_idx:02d}_prior.json",
            prompt_catalog=prompt_catalog,
            gemini_client=gemini_client,
            k=k,
            prior_min_items=prior_min_items,
            request_delay=request_delay,
        )
        subtask_results.append(result)

    parent = run_parent_prior(
        episode=episode,
        subtask_results=subtask_results,
        output_path=output_dir / "task_prior.json",
        prompt_catalog=prompt_catalog,
        gemini_client=gemini_client,
        prior_min_items=prior_min_items,
    )
    prompt_info_path = output_dir / "autolabel_prompt_info.json"
    write_json(prompt_info_path, parent)
    print(f"[saved] {prompt_info_path}", flush=True)
    return {
        "annotation_json": str(annotation_json),
        "image_root": str(image_root),
        "output_dir": str(output_dir),
        "subtask_count": len(subtask_results),
        "prior_min_items": prior_min_items,
        "subtask_prior_paths": [str(subtask_dir / f"subtask_{skill.stage_idx:02d}_prior.json") for skill in episode.skills],
        "task_prior_path": str(output_dir / "task_prior.json"),
        "prompt_info_path": str(prompt_info_path),
        "task_prior": parent,
    }


def run_subtask_prior(
    *,
    episode: EpisodeData,
    skill: SkillSpec,
    output_path: Path,
    prompt_catalog: PromptCatalog,
    gemini_client: GeminiClient,
    k: int,
    prior_min_items: int,
    request_delay: float,
) -> JsonObject:
    samples = sample_subtask_images(episode.image_root, skill, k)
    frame_results: List[JsonObject] = []
    system_instruction = prompt_catalog.get("subtask_prior_system")
    for request_index, sample in enumerate(samples, start=1):
        previous_record = frame_results[-1] if frame_results else None
        previous_context = (
            json.dumps(
                {
                    "previous_frame_number": previous_record["frame_number"],
                    "previous_model_response": previous_record["model_response"],
                },
                ensure_ascii=False,
                indent=2,
            )
            if previous_record is not None
            else "No previous sampled frame for this skill."
        )
        prompt = prompt_catalog.render(
            "subtask_prior_user",
            {
                "task_name": episode.task_name,
                "stage_idx": skill.stage_idx,
                "skill_idx": skill.skill_idx,
                "skill_description": skill.skill_description,
                "object_id": skill.object_id,
                "manuipation_object_id": skill.manuipation_object_id,
                "frame_duration": list(skill.frame_duration),
                "frame_number": sample.frame_number,
                "sample_index": request_index,
                "sample_count": len(samples),
                "prior_min_items": prior_min_items,
                "previous_frame_context": previous_context,
            },
        )
        image_paths = [sample.image_path]
        if previous_record is not None:
            image_paths = [Path(previous_record["image_path"]), sample.image_path]
        print(
            "[prior-subtask-frame] "
            f"stage_idx={skill.stage_idx} sample={request_index}/{len(samples)} "
            f"frame={sample.frame_number} image={sample.image_path}",
            flush=True,
        )
        response, metadata = gemini_client.generate_json(
            system_instruction=system_instruction,
            prompt=prompt,
            image_paths=image_paths,
            required_keys=SUBTASK_FRAME_KEYS,
        )
        frame_record: JsonObject = {
            "request_index": request_index,
            "frame_number": sample.frame_number,
            "image_path": str(sample.image_path),
            "image_index_in_stage": sample.image_index_in_stage,
            "model_response": response,
        }
        if previous_record is not None:
            frame_record["previous_image_path"] = previous_record["image_path"]
            frame_record["previous_frame_number"] = previous_record["frame_number"]
        if metadata:
            frame_record["google_response_metadata"] = metadata
        frame_results.append(frame_record)
        if request_delay > 0 and request_index < len(samples):
            time.sleep(request_delay)

    subtask_prior = summarize_subtask_prior(episode, skill, frame_results)
    subtask_prior["prior_min_items"] = prior_min_items
    subtask_prior = consolidate_subtask_prior(
        episode=episode,
        skill=skill,
        preliminary_prior=subtask_prior,
        prompt_catalog=prompt_catalog,
        gemini_client=gemini_client,
        prior_min_items=prior_min_items,
    )
    write_json(output_path, subtask_prior)
    print(f"[saved] {output_path}", flush=True)
    return subtask_prior


def summarize_subtask_prior(
    episode: EpisodeData,
    skill: SkillSpec,
    frame_results: List[JsonObject],
) -> JsonObject:
    completion_conditions = merge_string_lists(
        record["model_response"].get("completion_conditions", []) for record in frame_results
    )
    required_visual_evidence = merge_string_lists(
        record["model_response"].get("required_visual_evidence", []) for record in frame_results
    )
    state_transition_evidence = merge_string_lists(
        record["model_response"].get("state_transition_evidence", []) for record in frame_results
    )
    negative_conditions = merge_string_lists(
        record["model_response"].get("negative_conditions", []) for record in frame_results
    )
    common_false_positives = merge_string_lists(
        record["model_response"].get("common_false_positives", []) for record in frame_results
    )
    ambiguous_cases = merge_string_lists(
        record["model_response"].get("ambiguous_cases", []) for record in frame_results
    )
    subtask_name = first_text_value(
        record["model_response"].get("subtask_name") for record in frame_results
    )
    target_visual_description = first_json_object(
        record["model_response"].get("target_visual_description") for record in frame_results
    )
    timeline = [
        {
            "frame_number": record["frame_number"],
            "image_path": record["image_path"],
            "previous_frame_number": record.get("previous_frame_number"),
            "previous_image_path": record.get("previous_image_path", ""),
            "status_hint": record["model_response"].get("status_hint", ""),
            "frame_reasoning": record["model_response"].get("frame_reasoning", ""),
        }
        for record in frame_results
    ]
    return {
        "agent_type": "subtask_prior_agent",
        "task_name": episode.task_name,
        "annotation_json": str(episode.annotation_json),
        "image_root": str(episode.image_root),
        "stage_idx": skill.stage_idx,
        "skill_idx": skill.skill_idx,
        "skill_description": skill.skill_description,
        "subtask_name": subtask_name or skill.skill_description,
        "target_visual_description": target_visual_description,
        "object_id": skill.object_id,
        "manuipation_object_id": skill.manuipation_object_id,
        "frame_duration": list(skill.frame_duration),
        "sample_count": len(frame_results),
        "completion_conditions": completion_conditions,
        "required_visual_evidence": required_visual_evidence,
        "state_transition_evidence": state_transition_evidence,
        "negative_conditions": negative_conditions,
        "common_false_positives": common_false_positives,
        "ambiguous_cases": ambiguous_cases,
        "sampled_frame_analysis": timeline,
        "raw_frame_requests": frame_results,
    }


def consolidate_subtask_prior(
    *,
    episode: EpisodeData,
    skill: SkillSpec,
    preliminary_prior: JsonObject,
    prompt_catalog: PromptCatalog,
    gemini_client: GeminiClient,
    prior_min_items: int,
) -> JsonObject:
    system_instruction = prompt_catalog.get("subtask_prior_summary_system")
    prompt = prompt_catalog.render(
        "subtask_prior_summary_user",
        {
            "task_name": episode.task_name,
            "stage_idx": skill.stage_idx,
            "skill_idx": skill.skill_idx,
            "skill_description": skill.skill_description,
            "object_id": skill.object_id,
            "manuipation_object_id": skill.manuipation_object_id,
            "frame_duration": list(skill.frame_duration),
            "prior_min_items": prior_min_items,
            "preliminary_prior_json": json.dumps(preliminary_prior, ensure_ascii=False, indent=2),
        },
    )
    print(
        "[prior-subtask-summary] "
        f"stage_idx={skill.stage_idx} skill_idx={skill.skill_idx}",
        flush=True,
    )
    response, metadata = gemini_client.generate_json(
        system_instruction=system_instruction,
        prompt=prompt,
        required_keys=SUBTASK_SUMMARY_KEYS,
    )

    consolidated = dict(preliminary_prior)
    for key in SUBTASK_SUMMARY_KEYS:
        consolidated[key] = response.get(key)
    consolidated["summary_model_response"] = response
    if metadata:
        consolidated["summary_google_response_metadata"] = metadata
    return consolidated


def run_parent_prior(
    *,
    episode: EpisodeData,
    subtask_results: List[JsonObject],
    output_path: Path,
    prompt_catalog: PromptCatalog,
    gemini_client: GeminiClient,
    prior_min_items: int,
) -> JsonObject:
    system_instruction = prompt_catalog.get("parent_prior_system")
    prompt = prompt_catalog.render(
        "parent_prior_user",
        {
            "task_name": episode.task_name,
            "annotation_json": str(episode.annotation_json),
            "image_root": str(episode.image_root),
            "prior_min_items": prior_min_items,
            "subtask_priors_json": json.dumps(subtask_results, ensure_ascii=False, indent=2),
        },
    )
    print("[prior-parent] adjusting task-level prior", flush=True)
    response, metadata = gemini_client.generate_json(
        system_instruction=system_instruction,
        prompt=prompt,
        required_keys=PARENT_PRIOR_KEYS,
    )
    parent_prior: JsonObject = {
        "agent_type": "parent_prior_agent",
        "task_name": episode.task_name,
        "annotation_json": str(episode.annotation_json),
        "image_root": str(episode.image_root),
        "prior_min_items": prior_min_items,
        "model_response": response,
        "subtask_prior_count": len(subtask_results),
        "subtask_priors": subtask_results,
    }
    if metadata:
        parent_prior["google_response_metadata"] = metadata
    write_json(output_path, parent_prior)
    print(f"[saved] {output_path}", flush=True)
    return parent_prior


def merge_string_lists(groups) -> List[str]:
    merged: List[str] = []
    seen = set()
    for group in groups:
        if isinstance(group, str):
            items = [group]
        elif isinstance(group, list):
            items = group
        else:
            continue
        for item in items:
            text = str(item).strip()
            key = text.lower()
            if text and key not in seen:
                merged.append(text)
                seen.add(key)
    return merged


def first_text_value(values) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def first_json_object(values) -> JsonObject:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}
