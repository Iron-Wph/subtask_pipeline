import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

from .checkpoint import checkpoint_status, error_to_json, save_checkpoint
from .dataset import EpisodeData, SkillSpec, load_episode, sample_subtask_images
from .gemini_client import GeminiClient
from .io_utils import JsonObject, write_json
from .prompts import PromptCatalog
from .visual_guidance import sanitize_visual_guidance

SUBTASK_FRAME_KEYS = {
    "frame_reasoning",
    "objects",
    "robot_state",
    "scene_context",
    "skill_relevant_observations",
    "temporal_change_from_previous",
    "uncertainty",
}
SUBTASK_SUMMARY_KEYS = {
    "skill_type_hypothesis",
    "target_binding",
    "subtask_name",
    "target_visual_description",
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
}
DEFAULT_PRIOR_MIN_ITEMS = 4
DEFAULT_PARENT_PRIOR_ATTEMPTS = 3
PARENT_REVIEW_KEYS = (
    "stage_idx",
    "skill_idx",
    "skill_description",
    "subtask_name",
    "review_status",
    "review_notes",
    "target_consistency_issues",
    "missing_or_weak_child_criteria",
    "completion_gate_corrections",
    "additional_negative_conditions",
    "additional_not_sufficient_for_completion",
    "additional_ambiguous_cases",
    "cross_skill_risks",
    "parent_review_guidance",
)
PARENT_REVIEW_LIST_FIELDS = {
    "review_notes",
    "target_consistency_issues",
    "missing_or_weak_child_criteria",
    "completion_gate_corrections",
    "additional_negative_conditions",
    "additional_not_sufficient_for_completion",
    "additional_ambiguous_cases",
    "cross_skill_risks",
}
PARENT_REVIEW_CHILD_IDENTITY_FIELDS = {
    "stage_idx",
    "skill_idx",
    "skill_description",
    "subtask_name",
}
CHILD_PRIOR_SNAPSHOT_KEYS = (
    "skill_type_hypothesis",
    "target_binding",
    "target_visual_description",
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
RUNNING_PRIOR_CONTEXT_MAX_ITEMS = 12
RUNNING_PRIOR_CONTEXT_RECENT_FRAMES = 3


def run_prior_pipeline(
    *,
    annotation_json: Path,
    image_root: Path,
    output_dir: Path,
    prompt_catalog: PromptCatalog,
    gemini_client: GeminiClient,
    k: int = 10,
    prior_frame_stride: Optional[int] = None,
    prior_min_items: int = DEFAULT_PRIOR_MIN_ITEMS,
    parent_prior_attempts: int = DEFAULT_PARENT_PRIOR_ATTEMPTS,
    request_delay: float = 0.0,
) -> JsonObject:
    if prior_min_items < 1:
        raise ValueError("prior_min_items must be at least 1.")
    if parent_prior_attempts < 1:
        raise ValueError("parent_prior_attempts must be at least 1.")
    if prior_frame_stride is not None and prior_frame_stride < 1:
        raise ValueError("prior_frame_stride must be >= 1.")
    episode = load_episode(annotation_json, image_root)
    if not episode.task_name:
        raise ValueError("Annotation JSON must provide task_name, main_task, or task_description.")

    subtask_dir = output_dir / "subtasks"
    subtask_results: List[JsonObject] = []
    current_skill: Optional[SkillSpec] = None
    try:
        for skill in episode.skills:
            current_skill = skill
            result = run_subtask_prior(
                episode=episode,
                skill=skill,
                output_path=subtask_dir / f"subtask_{skill.stage_idx:02d}_prior.json",
                prompt_catalog=prompt_catalog,
                gemini_client=gemini_client,
                k=k,
                prior_frame_stride=prior_frame_stride,
                prior_min_items=prior_min_items,
                request_delay=request_delay,
            )
            subtask_results.append(result)

        current_skill = None
        parent = run_parent_prior(
            episode=episode,
            subtask_results=subtask_results,
            output_path=output_dir / "task_prior.json",
            prompt_catalog=prompt_catalog,
            gemini_client=gemini_client,
            prior_min_items=prior_min_items,
            parent_prior_attempts=parent_prior_attempts,
        )
    except BaseException as exc:
        checkpoint = build_prior_pipeline_checkpoint(
            annotation_json=annotation_json,
            image_root=image_root,
            output_dir=output_dir,
            episode=episode,
            subtask_results=subtask_results,
            k=k,
            prior_frame_stride=prior_frame_stride,
            prior_min_items=prior_min_items,
            parent_prior_attempts=parent_prior_attempts,
            status=checkpoint_status(exc),
            error=error_to_json(exc),
            current_skill=current_skill,
        )
        save_checkpoint(output_dir / "prior_checkpoint.json", checkpoint)
        raise

    prompt_info_path = output_dir / "autolabel_prompt_info.json"
    write_json(prompt_info_path, parent)
    print(f"[saved] {prompt_info_path}", flush=True)
    return {
        "annotation_json": str(annotation_json),
        "image_root": str(image_root),
        "output_dir": str(output_dir),
        "subtask_count": len(subtask_results),
        "sample_k": k,
        "prior_frame_stride": prior_frame_stride,
        "prior_min_items": prior_min_items,
        "parent_prior_attempts": parent_prior_attempts,
        "subtask_prior_paths": [str(subtask_dir / f"subtask_{skill.stage_idx:02d}_prior.json") for skill in episode.skills],
        "task_prior_path": str(output_dir / "task_prior.json"),
        "prompt_info_path": str(prompt_info_path),
        "status": "complete",
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
    prior_frame_stride: Optional[int],
    prior_min_items: int,
    request_delay: float,
) -> JsonObject:
    samples = sample_subtask_images(episode.image_root, skill, k, frame_stride=prior_frame_stride)
    frame_results: List[JsonObject] = []
    system_instruction = prompt_catalog.get("subtask_prior_system")
    current_request: Optional[JsonObject] = None
    subtask_prior: Optional[JsonObject] = None
    try:
        for request_index, sample in enumerate(samples, start=1):
            current_request = {
                "phase": "subtask_frame_prior",
                "stage_idx": skill.stage_idx,
                "skill_idx": skill.skill_idx,
                "request_index": request_index,
                "total_requests": len(samples),
                "frame_number": sample.frame_number,
                "image_path": str(sample.image_path),
            }
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
            running_observation_context = render_running_observation_context(episode, skill, frame_results)
            prompt_values = {
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
                "running_observation_context": running_observation_context,
                "universal_visual_rubric": prompt_catalog.render_optional("universal_visual_rubric", {}),
                "action_primitive_rubric": prompt_catalog.render_optional("action_primitive_rubric", {}),
            }
            prompt = prompt_catalog.render("subtask_prior_user", prompt_values)
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
            response = sanitize_visual_guidance(response)
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

        current_request = {"phase": "subtask_summary_prior", "stage_idx": skill.stage_idx, "skill_idx": skill.skill_idx}
        subtask_prior = build_subtask_checkpoint_payload(
            episode=episode,
            skill=skill,
            frame_results=frame_results,
            expected_sample_count=len(samples),
            k=k,
            prior_frame_stride=prior_frame_stride,
            prior_min_items=prior_min_items,
            status="frame_prior_complete",
        )
        subtask_prior = consolidate_subtask_prior(
            episode=episode,
            skill=skill,
            preliminary_prior=subtask_prior,
            prompt_catalog=prompt_catalog,
            gemini_client=gemini_client,
            prior_min_items=prior_min_items,
        )
    except BaseException as exc:
        checkpoint_payload = subtask_prior or build_subtask_checkpoint_payload(
            episode=episode,
            skill=skill,
            frame_results=frame_results,
            expected_sample_count=len(samples),
            k=k,
            prior_frame_stride=prior_frame_stride,
            prior_min_items=prior_min_items,
            status=checkpoint_status(exc),
            current_request=current_request,
            error=error_to_json(exc),
        )
        if subtask_prior is not None:
            checkpoint_payload["status"] = checkpoint_status(exc)
            checkpoint_payload["current_request"] = current_request
            checkpoint_payload["error"] = error_to_json(exc)
        save_checkpoint(output_path, checkpoint_payload)
        raise

    subtask_prior["status"] = "complete"
    write_json(output_path, subtask_prior)
    print(f"[saved] {output_path}", flush=True)
    return subtask_prior


def build_prior_pipeline_checkpoint(
    *,
    annotation_json: Path,
    image_root: Path,
    output_dir: Path,
    episode: EpisodeData,
    subtask_results: List[JsonObject],
    k: int,
    prior_frame_stride: Optional[int],
    prior_min_items: int,
    parent_prior_attempts: int,
    status: str,
    error: JsonObject,
    current_skill: Optional[SkillSpec],
) -> JsonObject:
    checkpoint: JsonObject = {
        "status": status,
        "annotation_json": str(annotation_json),
        "image_root": str(image_root),
        "output_dir": str(output_dir),
        "task_name": episode.task_name,
        "expected_subtask_count": len(episode.skills),
        "subtask_count": len(subtask_results),
        "sample_k": k,
        "prior_frame_stride": prior_frame_stride,
        "prior_min_items": prior_min_items,
        "parent_prior_attempts": parent_prior_attempts,
        "subtask_prior_paths": [
            str(output_dir / "subtasks" / f"subtask_{skill.stage_idx:02d}_prior.json")
            for skill in episode.skills
        ],
        "task_prior_path": str(output_dir / "task_prior.json"),
        "prompt_info_path": str(output_dir / "autolabel_prompt_info.json"),
        "subtask_priors": subtask_results,
        "error": error,
    }
    if current_skill is not None:
        checkpoint["current_skill"] = {
            "stage_idx": current_skill.stage_idx,
            "skill_idx": current_skill.skill_idx,
            "skill_description": current_skill.skill_description,
            "frame_duration": list(current_skill.frame_duration),
        }
    else:
        checkpoint["current_phase"] = "parent_prior"
    return checkpoint


def build_subtask_checkpoint_payload(
    *,
    episode: EpisodeData,
    skill: SkillSpec,
    frame_results: List[JsonObject],
    expected_sample_count: int,
    k: int,
    prior_frame_stride: Optional[int],
    prior_min_items: int,
    status: str,
    current_request: Optional[JsonObject] = None,
    error: Optional[JsonObject] = None,
) -> JsonObject:
    subtask_prior = summarize_subtask_prior(episode, skill, frame_results)
    subtask_prior["status"] = status
    subtask_prior["sampling_strategy"] = "frame_stride" if prior_frame_stride is not None else "uniform_k"
    subtask_prior["sample_k"] = k
    subtask_prior["prior_frame_stride"] = prior_frame_stride
    subtask_prior["prior_min_items"] = prior_min_items
    subtask_prior["expected_sample_count"] = expected_sample_count
    subtask_prior["processed_sample_count"] = len(frame_results)
    if current_request is not None:
        subtask_prior["current_request"] = current_request
    if error is not None:
        subtask_prior["error"] = error
    return subtask_prior


def summarize_subtask_prior(
    episode: EpisodeData,
    skill: SkillSpec,
    frame_results: List[JsonObject],
) -> JsonObject:
    frame_observation_schemas = [build_frame_observation_record(record) for record in frame_results]
    timeline = [
        {
            "frame_number": record["frame_number"],
            "image_path": record["image_path"],
            "previous_frame_number": record.get("previous_frame_number"),
            "previous_image_path": record.get("previous_image_path", ""),
            "frame_reasoning": record["model_response"].get("frame_reasoning", ""),
            "objects": record["model_response"].get("objects", []),
            "robot_state": record["model_response"].get("robot_state", {}),
            "scene_context": record["model_response"].get("scene_context", ""),
            "skill_relevant_observations": record["model_response"].get("skill_relevant_observations", []),
            "temporal_change_from_previous": record["model_response"].get("temporal_change_from_previous", ""),
            "uncertainty": record["model_response"].get("uncertainty", []),
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
        "skill_type_hypothesis": "",
        "subtask_name": skill.skill_description,
        "target_binding": {},
        "target_visual_description": {},
        "object_id": skill.object_id,
        "manuipation_object_id": skill.manuipation_object_id,
        "frame_duration": list(skill.frame_duration),
        "sample_count": len(frame_results),
        "pre_completion_state": [],
        "in_progress_state": [],
        "completion_gates": [],
        "completion_conditions": [],
        "required_visual_evidence": [],
        "state_transition_evidence": [],
        "negative_conditions": [],
        "not_sufficient_for_completion": [],
        "common_false_positives": [],
        "ambiguous_cases": [],
        "frame_observation_schemas": frame_observation_schemas,
        "sampled_frame_analysis": timeline,
        "raw_frame_requests": frame_results,
    }


def build_frame_observation_record(record: JsonObject) -> JsonObject:
    model_response = record.get("model_response", {})
    return {
        "frame_number": record.get("frame_number"),
        "image_path": record.get("image_path", ""),
        "previous_frame_number": record.get("previous_frame_number"),
        "frame_reasoning": model_response.get("frame_reasoning", ""),
        "objects": model_response.get("objects", []),
        "robot_state": model_response.get("robot_state", {}),
        "scene_context": model_response.get("scene_context", ""),
        "skill_relevant_observations": model_response.get("skill_relevant_observations", []),
        "temporal_change_from_previous": model_response.get("temporal_change_from_previous", ""),
        "uncertainty": model_response.get("uncertainty", []),
    }


def render_running_observation_context(
    episode: EpisodeData,
    skill: SkillSpec,
    frame_results: List[JsonObject],
) -> str:
    payload = build_running_observation_context(episode, skill, frame_results)
    if not payload:
        return "No accumulated sampled-frame observations yet."
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_running_observation_context(
    episode: EpisodeData,
    skill: SkillSpec,
    frame_results: List[JsonObject],
) -> JsonObject:
    if not frame_results:
        return {}
    latest_record = frame_results[-1]
    payload: JsonObject = {
        "description": (
            "Compact accumulated observation draft from earlier sampled frames for this same skill. "
            "Use it only to maintain object identity, visible state progression, robot state changes, "
            "and uncertainty across sampled frames. Do not convert observations into completion rules here."
        ),
        "processed_sample_count": len(frame_results),
        "latest_frame_number": latest_record.get("frame_number"),
    }
    payload["observed_object_names"] = merge_observed_object_names(frame_results)[:RUNNING_PRIOR_CONTEXT_MAX_ITEMS]
    payload["recent_skill_relevant_observations"] = merge_string_lists(
        record.get("model_response", {}).get("skill_relevant_observations", []) for record in frame_results
    )[:RUNNING_PRIOR_CONTEXT_MAX_ITEMS]
    payload["recent_uncertainty"] = merge_string_lists(
        record.get("model_response", {}).get("uncertainty", []) for record in frame_results
    )[:RUNNING_PRIOR_CONTEXT_MAX_ITEMS]
    recent_frames: List[JsonObject] = []
    for record in frame_results[-RUNNING_PRIOR_CONTEXT_RECENT_FRAMES:]:
        recent_frames.append(build_frame_observation_record(record))
    payload["recent_sampled_frames"] = recent_frames
    return payload


def merge_observed_object_names(frame_results: List[JsonObject]) -> List[str]:
    names: List[str] = []
    seen = set()
    for record in frame_results:
        objects = record.get("model_response", {}).get("objects", [])
        if not isinstance(objects, list):
            continue
        for item in objects:
            if not isinstance(item, dict):
                continue
            name = item.get("object_name") or item.get("name")
            if not isinstance(name, str):
                continue
            normalized = " ".join(name.split()).lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                names.append(name.strip())
    return names


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
    summary_input, summary_image_paths = build_summary_endpoint_payload(preliminary_prior)
    prompt_values = {
        "task_name": episode.task_name,
        "stage_idx": skill.stage_idx,
        "skill_idx": skill.skill_idx,
        "skill_description": skill.skill_description,
        "object_id": skill.object_id,
        "manuipation_object_id": skill.manuipation_object_id,
        "frame_duration": list(skill.frame_duration),
        "prior_min_items": prior_min_items,
        "preliminary_prior_json": json.dumps(summary_input, ensure_ascii=False, indent=2),
        "universal_visual_rubric": prompt_catalog.render_optional("universal_visual_rubric", {}),
        "action_primitive_rubric": prompt_catalog.render_optional("action_primitive_rubric", {}),
        "prior_review_rubric": prompt_catalog.render_optional("prior_review_rubric", {}),
    }
    prompt = append_optional_prompt(
        prompt_catalog.render("subtask_prior_summary_user", prompt_values),
        prompt_catalog.render_optional("target_consistency_rules", prompt_values),
    )
    print(
        "[prior-subtask-summary] "
        f"stage_idx={skill.stage_idx} skill_idx={skill.skill_idx}",
        flush=True,
    )
    response, metadata = gemini_client.generate_json(
        system_instruction=system_instruction,
        prompt=prompt,
        image_paths=summary_image_paths,
        required_keys=SUBTASK_SUMMARY_KEYS,
    )
    response = sanitize_visual_guidance(response)

    consolidated = dict(preliminary_prior)
    for key in SUBTASK_SUMMARY_KEYS:
        consolidated[key] = response.get(key)
    consolidated["summary_model_response"] = response
    if metadata:
        consolidated["summary_google_response_metadata"] = metadata
    return sanitize_visual_guidance(consolidated)


def build_summary_endpoint_payload(preliminary_prior: JsonObject) -> Tuple[JsonObject, List[Path]]:
    frame_results = preliminary_prior.get("raw_frame_requests", [])
    if not isinstance(frame_results, list) or not frame_results:
        return preliminary_prior, []

    first_record = frame_results[0]
    last_record = frame_results[-1]
    first_observation = build_frame_observation_record(first_record)
    last_observation = build_frame_observation_record(last_record)
    summary_input: JsonObject = {
        "agent_type": preliminary_prior.get("agent_type", "subtask_prior_agent"),
        "task_name": preliminary_prior.get("task_name", ""),
        "stage_idx": preliminary_prior.get("stage_idx"),
        "skill_idx": preliminary_prior.get("skill_idx"),
        "skill_description": preliminary_prior.get("skill_description", ""),
        "object_id": preliminary_prior.get("object_id", ""),
        "manuipation_object_id": preliminary_prior.get("manuipation_object_id", ""),
        "frame_duration": preliminary_prior.get("frame_duration", []),
        "sample_count": preliminary_prior.get("sample_count", len(frame_results)),
        "summary_sampling_policy": (
            "Only the first and last sampled-frame observation responses are sent to the "
            "summary agent in this experimental branch."
        ),
        "first_observation": first_observation,
        "last_observation": last_observation,
    }

    image_paths: List[Path] = []
    for record in (first_record, last_record):
        image_path = record.get("image_path", "")
        if not image_path:
            continue
        path = Path(str(image_path))
        if path not in image_paths:
            image_paths.append(path)
    return summary_input, image_paths


def run_parent_prior(
    *,
    episode: EpisodeData,
    subtask_results: List[JsonObject],
    output_path: Path,
    prompt_catalog: PromptCatalog,
    gemini_client: GeminiClient,
    prior_min_items: int,
    parent_prior_attempts: int = DEFAULT_PARENT_PRIOR_ATTEMPTS,
) -> JsonObject:
    if parent_prior_attempts < 1:
        raise ValueError("parent_prior_attempts must be at least 1.")

    system_instruction = prompt_catalog.get("parent_prior_system")
    global_review_context = build_parent_review_global_context(episode, subtask_results)
    parent_reviews: List[JsonObject] = []
    parent_errors: List[JsonObject] = []
    metadata_by_skill: List[JsonObject] = []
    parent_attempt_count = 0
    current_request: Optional[JsonObject] = None

    for child_prior in subtask_results:
        stage_idx = child_prior.get("stage_idx")
        skill_idx = child_prior.get("skill_idx")
        current_request = {
            "phase": "parent_skill_prior_review",
            "stage_idx": stage_idx,
            "skill_idx": skill_idx,
            "parent_prior_attempts": parent_prior_attempts,
            "reviewed_subtask_count": len(parent_reviews),
            "subtask_prior_count": len(subtask_results),
        }
        try:
            review, metadata, review_errors, attempt_count = run_single_parent_skill_review(
                episode=episode,
                child_prior=child_prior,
                global_review_context=global_review_context,
                system_instruction=system_instruction,
                prompt_catalog=prompt_catalog,
                gemini_client=gemini_client,
                prior_min_items=prior_min_items,
                parent_prior_attempts=parent_prior_attempts,
            )
            parent_reviews.append(review)
            parent_attempt_count += attempt_count
            if review.get("review_status") == "review_failed_child_fallback":
                parent_errors.extend(review_errors)
            if metadata:
                metadata_by_skill.append(
                    {
                        "stage_idx": stage_idx,
                        "skill_idx": skill_idx,
                        "metadata": metadata,
                    }
                )
        except KeyboardInterrupt as exc:
            parent_checkpoint: JsonObject = {
                "status": checkpoint_status(exc),
                "agent_type": "per_skill_parent_review_agent",
                "task_name": episode.task_name,
                "annotation_json": str(episode.annotation_json),
                "image_root": str(episode.image_root),
                "prior_min_items": prior_min_items,
                "parent_prior_attempts": parent_prior_attempts,
                "current_request": current_request,
                "error": error_to_json(exc),
                "reviewed_subtask_count": len(parent_reviews),
                "parent_review_results": parent_reviews,
                "parent_errors": parent_errors,
                "subtask_prior_count": len(subtask_results),
                "subtask_priors": subtask_results,
            }
            save_checkpoint(output_path, parent_checkpoint)
            raise

    status = "complete" if not parent_errors else "partial_skill_review_fallback"
    response: JsonObject = {
        "task_summary": episode.task_name,
        "global_review_context": global_review_context,
        "skills": parent_reviews,
    }
    parent_prior: JsonObject = {
        "status": status,
        "usable_for_generation": True,
        "agent_type": "per_skill_parent_review_agent",
        "task_name": episode.task_name,
        "annotation_json": str(episode.annotation_json),
        "image_root": str(episode.image_root),
        "prior_min_items": prior_min_items,
        "parent_prior_attempts": parent_prior_attempts,
        "parent_attempt_count": parent_attempt_count,
        "reviewed_subtask_count": len(parent_reviews),
        "model_response": response,
        "subtask_prior_count": len(subtask_results),
        "subtask_priors": subtask_results,
    }
    if parent_errors:
        parent_prior["parent_errors"] = parent_errors
        parent_prior["parent_error"] = parent_errors[-1]
    if metadata_by_skill:
        parent_prior["google_response_metadata"] = {
            "skill_reviews": metadata_by_skill,
        }
    parent_prior = sanitize_visual_guidance(parent_prior)
    write_json(output_path, parent_prior)
    print(f"[saved] {output_path}", flush=True)
    return parent_prior


def run_single_parent_skill_review(
    *,
    episode: EpisodeData,
    child_prior: JsonObject,
    global_review_context: JsonObject,
    system_instruction: str,
    prompt_catalog: PromptCatalog,
    gemini_client: GeminiClient,
    prior_min_items: int,
    parent_prior_attempts: int,
) -> Tuple[JsonObject, JsonObject, List[JsonObject], int]:
    stage_idx = child_prior.get("stage_idx")
    skill_idx = child_prior.get("skill_idx")
    prompt_values = {
        "task_name": episode.task_name,
        "annotation_json": str(episode.annotation_json),
        "image_root": str(episode.image_root),
        "prior_min_items": prior_min_items,
        "global_review_context_json": json.dumps(global_review_context, ensure_ascii=False, indent=2),
        "child_prior_json": json.dumps(
            compact_child_prior_for_parent_review(child_prior),
            ensure_ascii=False,
            indent=2,
        ),
        "universal_visual_rubric": prompt_catalog.render_optional("universal_visual_rubric", {}),
        "action_primitive_rubric": prompt_catalog.render_optional("action_primitive_rubric", {}),
        "prior_review_rubric": prompt_catalog.render_optional("prior_review_rubric", {}),
    }
    prompt = append_optional_prompt(
        prompt_catalog.render("parent_prior_user", prompt_values),
        prompt_catalog.render_optional("target_consistency_rules", prompt_values),
    )

    review_errors: List[JsonObject] = []
    for parent_attempt in range(1, parent_prior_attempts + 1):
        print(
            "[prior-parent-skill] "
            f"stage_idx={stage_idx} skill_idx={skill_idx} "
            f"attempt={parent_attempt}/{parent_prior_attempts}",
            flush=True,
        )
        try:
            response, metadata = gemini_client.generate_json(
                system_instruction=system_instruction,
                prompt=prompt,
                required_keys=PARENT_REVIEW_KEYS,
            )
            return build_parent_review_skill(response, child_prior), metadata, review_errors, parent_attempt
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            error = error_to_json(exc)
            error["stage_idx"] = stage_idx
            error["skill_idx"] = skill_idx
            error["parent_attempt"] = parent_attempt
            review_errors.append(error)
            print(
                "[retry] parent_skill_review_failed "
                f"stage_idx={stage_idx} skill_idx={skill_idx} "
                f"attempt={parent_attempt}/{parent_prior_attempts} "
                f"error={error['type']}: {error['message']}",
                flush=True,
            )
            if parent_attempt < parent_prior_attempts:
                time.sleep(min(3.0, float(parent_attempt)))

    return (
        build_parent_skill_review_fallback(child_prior, review_errors[-1] if review_errors else {}),
        {},
        review_errors,
        parent_prior_attempts,
    )


def build_parent_review_global_context(
    episode: EpisodeData,
    subtask_results: List[JsonObject],
) -> JsonObject:
    return {
        "task_name": episode.task_name,
        "annotation_json": str(episode.annotation_json),
        "image_root": str(episode.image_root),
        "skill_order": [compact_child_prior_for_context(child) for child in subtask_results],
    }


def compact_child_prior_for_context(child_prior: JsonObject) -> JsonObject:
    keys = (
        "stage_idx",
        "skill_idx",
        "skill_description",
        "object_id",
        "manuipation_object_id",
        "subtask_name",
        "skill_type_hypothesis",
        "target_binding",
    )
    return {
        key: child_prior.get(key)
        for key in keys
        if has_prompt_value(child_prior.get(key))
    }


def compact_child_prior_for_parent_review(child_prior: JsonObject) -> JsonObject:
    keys = (
        "stage_idx",
        "skill_idx",
        "skill_description",
        "object_id",
        "manuipation_object_id",
        "frame_duration",
        "sample_count",
        "summary_sampling_policy",
        *CHILD_PRIOR_SNAPSHOT_KEYS,
    )
    return {
        key: child_prior.get(key)
        for key in keys
        if has_prompt_value(child_prior.get(key))
    }


def build_parent_skill_review_fallback(child_prior: JsonObject, error: JsonObject) -> JsonObject:
    review = build_parent_review_skill(
        {
            "review_status": "review_failed_child_fallback",
            "review_notes": [
                "The per-skill parent review request failed or returned invalid JSON. "
                "Use the attached child_prior_snapshot for generation and inspect parent_errors for details."
            ],
            "parent_review_guidance": (
                "Parent review was not available for this skill; child prior fields remain the "
                "primary generation guidance."
            ),
        },
        child_prior,
    )
    if error:
        review["review_error"] = error
    return sanitize_visual_guidance(review)


def has_prompt_value(value: object) -> bool:
    return value not in (None, "", [], {})


def build_parent_review_skill(parent_skill: JsonObject, child_prior: JsonObject) -> JsonObject:
    merged: JsonObject = {}
    for key in PARENT_REVIEW_KEYS:
        parent_value = parent_skill.get(key)
        child_value = child_prior.get(key)
        if key in PARENT_REVIEW_LIST_FIELDS:
            value = merge_string_lists([parent_value])
        elif key in PARENT_REVIEW_CHILD_IDENTITY_FIELDS:
            value = child_value if has_prompt_value(child_value) else parent_value
        else:
            value = parent_value if has_prompt_value(parent_value) else child_value
        if has_prompt_value(value):
            merged[key] = value
    child_snapshot = {
        key: child_prior.get(key)
        for key in CHILD_PRIOR_SNAPSHOT_KEYS
        if has_prompt_value(child_prior.get(key))
    }
    if child_snapshot:
        merged["child_prior_snapshot"] = child_snapshot
    return sanitize_visual_guidance(merged)


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


def append_optional_prompt(prompt: str, optional_prompt: str) -> str:
    if not optional_prompt.strip():
        return prompt
    return f"{prompt}\n\n{optional_prompt.strip()}"
