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
    "completion_conditions",
    "visual_changes",
    "false_positive_risks",
    "status_hint",
}
PARENT_PRIOR_KEYS = {
    "task_summary",
    "global_completion_order",
    "global_visual_adjustments",
    "cross_subtask_false_positive_risks",
    "subtask_priors",
}


def run_prior_pipeline(
    *,
    annotation_json: Path,
    image_root: Path,
    output_dir: Path,
    prompt_catalog: PromptCatalog,
    gemini_client: GeminiClient,
    k: int = 10,
    request_delay: float = 0.0,
) -> JsonObject:
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
            request_delay=request_delay,
        )
        subtask_results.append(result)

    parent = run_parent_prior(
        episode=episode,
        subtask_results=subtask_results,
        output_path=output_dir / "task_prior.json",
        prompt_catalog=prompt_catalog,
        gemini_client=gemini_client,
    )
    return {
        "annotation_json": str(annotation_json),
        "image_root": str(image_root),
        "output_dir": str(output_dir),
        "subtask_count": len(subtask_results),
        "subtask_prior_paths": [str(subtask_dir / f"subtask_{skill.stage_idx:02d}_prior.json") for skill in episode.skills],
        "task_prior_path": str(output_dir / "task_prior.json"),
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
    request_delay: float,
) -> JsonObject:
    samples = sample_subtask_images(episode.image_root, skill, k)
    frame_results: List[JsonObject] = []
    system_instruction = prompt_catalog.get("subtask_prior_system")
    for request_index, sample in enumerate(samples, start=1):
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
            },
        )
        print(
            "[prior-subtask-frame] "
            f"stage_idx={skill.stage_idx} sample={request_index}/{len(samples)} "
            f"frame={sample.frame_number} image={sample.image_path}",
            flush=True,
        )
        response, metadata = gemini_client.generate_json(
            system_instruction=system_instruction,
            prompt=prompt,
            image_paths=[sample.image_path],
            required_keys=SUBTASK_FRAME_KEYS,
        )
        frame_record: JsonObject = {
            "request_index": request_index,
            "frame_number": sample.frame_number,
            "image_path": str(sample.image_path),
            "image_index_in_stage": sample.image_index_in_stage,
            "model_response": response,
        }
        if metadata:
            frame_record["google_response_metadata"] = metadata
        frame_results.append(frame_record)
        if request_delay > 0 and request_index < len(samples):
            time.sleep(request_delay)

    subtask_prior = summarize_subtask_prior(episode, skill, frame_results)
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
    visual_changes = merge_string_lists(
        record["model_response"].get("visual_changes", []) for record in frame_results
    )
    false_positive_risks = merge_string_lists(
        record["model_response"].get("false_positive_risks", []) for record in frame_results
    )
    timeline = [
        {
            "frame_number": record["frame_number"],
            "image_path": record["image_path"],
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
        "object_id": skill.object_id,
        "manuipation_object_id": skill.manuipation_object_id,
        "frame_duration": list(skill.frame_duration),
        "sample_count": len(frame_results),
        "completion_conditions": completion_conditions,
        "visual_changes": visual_changes,
        "false_positive_risks": false_positive_risks,
        "sampled_frame_analysis": timeline,
        "raw_frame_requests": frame_results,
    }


def run_parent_prior(
    *,
    episode: EpisodeData,
    subtask_results: List[JsonObject],
    output_path: Path,
    prompt_catalog: PromptCatalog,
    gemini_client: GeminiClient,
) -> JsonObject:
    system_instruction = prompt_catalog.get("parent_prior_system")
    prompt = prompt_catalog.render(
        "parent_prior_user",
        {
            "task_name": episode.task_name,
            "annotation_json": str(episode.annotation_json),
            "image_root": str(episode.image_root),
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
