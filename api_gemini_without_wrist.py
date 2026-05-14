import argparse
import copy
import json
import os
import re
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

DEFAULT_MODEL = "gemini-3.1-pro-preview"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
FRAME_SELECTION = "valid_duration_stride"
DEFAULT_FRAME_STRIDE = 80
SUBTASK_IMAGE_DIR_PREFIXES = ("stage", "skill")
DEFAULT_MAX_RETRIES = 8
DEFAULT_RETRY_INITIAL_DELAY = 10.0
DEFAULT_RETRY_MAX_DELAY = 120.0
DEFAULT_MAX_RESPONSE_RETRIES = 3
NO_FOR_SURE_STATUS = "no_for_sure"
VALID_CURRENT_SKILL_STATUSES = {
    "not_started",
    "in_progress",
    "completed",
    "completed_and_transitioning",
    NO_FOR_SURE_STATUS,
}
REQUIRED_RESPONSE_KEYS = {
    "reasoning",
    "new_memory",
    "subtask",
    "current_skill_status",
    "visible_transition",
    "is_subtask_completed",
}
PROMPT_LEAK_MARKERS = (
    "current annotated skill",
    "current skill is",
    "because the current skill",
    "candidate skill",
    "skill label",
    "the annotation",
    "annotation boundary",
    "according to the prompt",
    "according to the instruction",
    "frame range",
    "frame timing",
    "sampled frame",
    "next skill",
    "completion gate",
    "scheduled observation",
    "final observation",
    "final scheduled",
    "last request",
    "final request",
    "not the final",
    "downstream program",
    "previous response",
    "previous answer",
    "retry",
)
IMAGE_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

JsonObject = Dict[str, Any]


class ModelResponseFormatError(ValueError):
    def __init__(self, message: str, content: str):
        super().__init__(f"{message}: {content}")
        self.content = content


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def format_annotation_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def get_skills(annotation: JsonObject) -> List[JsonObject]:
    skills = annotation.get("skills")
    if skills is None:
        skills = annotation.get("skill_annotation")

    if not isinstance(skills, list):
        raise ValueError("Annotation JSON must contain a skills or skill_annotation list.")

    for skill in skills:
        if not isinstance(skill, dict):
            raise ValueError("Every skill item must be a JSON object.")

    return skills


def get_subtask_image_dir_candidates(image_root: Path, stage_idx: int) -> List[Path]:
    return [image_root / f"{prefix}_{stage_idx:02d}" for prefix in SUBTASK_IMAGE_DIR_PREFIXES]


def find_stage_images(image_root: Path, stage_idx: int) -> List[Path]:
    candidates = get_subtask_image_dir_candidates(image_root, stage_idx)
    stage_dir = next((path for path in candidates if path.is_dir()), None)
    if stage_dir is None:
        expected_dirs = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Missing subtask image directory. Expected one of: {expected_dirs}")

    images = sorted(
        path
        for path in stage_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No image files found in subtask image directory: {stage_dir}")

    return images


def parse_frame_number(image_path: Path) -> Optional[int]:
    match = re.fullmatch(r"frame_(\d{6})", image_path.stem)
    if not match:
        return None
    return int(match.group(1))


def get_frame_duration(skill: JsonObject, stage_idx: int) -> List[int]:
    frame_duration = skill.get("frame_duration")
    if (
        not isinstance(frame_duration, list)
        or len(frame_duration) != 2
        or not all(isinstance(frame, int) for frame in frame_duration)
    ):
        raise ValueError(
            f"Skill at index {stage_idx} must contain frame_duration as [start_frame, end_frame)."
        )

    start_frame, end_frame = frame_duration
    if start_frame >= end_frame:
        raise ValueError(
            f"Skill at index {stage_idx} has invalid half-open frame_duration: {frame_duration}."
        )

    return frame_duration


def build_frame_image_map(images: List[Path]) -> Dict[int, Path]:
    frame_images: Dict[int, Path] = {}
    for image_path in images:
        frame_number = parse_frame_number(image_path)
        if frame_number is None:
            continue
        frame_images.setdefault(frame_number, image_path)
    return frame_images


def build_frame_selection_label(frame_stride: int) -> str:
    return f"{FRAME_SELECTION}_{frame_stride}"


def get_valid_duration(annotation: JsonObject) -> List[int]:
    valid_duration = annotation.get("valid_duration")
    if valid_duration is None:
        meta_data = annotation.get("meta_data")
        if isinstance(meta_data, dict):
            valid_duration = meta_data.get("valid_duration")

    if (
        not isinstance(valid_duration, list)
        or len(valid_duration) != 2
        or not all(isinstance(frame, int) for frame in valid_duration)
    ):
        raise ValueError(
            "Annotation JSON must contain valid_duration as [start_frame, end_frame), "
            "either at the top level or under meta_data."
        )

    start_frame, end_frame = valid_duration
    if start_frame >= end_frame:
        raise ValueError(f"Invalid half-open valid_duration: {valid_duration}.")

    return valid_duration


def frame_in_duration(frame_number: int, frame_duration: List[int]) -> bool:
    start_frame, end_frame = frame_duration
    return start_frame <= frame_number < end_frame


def find_skill_for_frame(
    skill_specs: List[JsonObject],
    frame_number: int,
) -> Optional[JsonObject]:
    for spec in skill_specs:
        if frame_in_duration(frame_number, spec["frame_duration"]):
            return spec
    return None


def build_skill_specs(skills: List[JsonObject]) -> List[JsonObject]:
    specs: List[JsonObject] = []
    for stage_idx, skill in enumerate(skills):
        specs.append(
            {
                "stage_idx": stage_idx,
                "skill": skill,
                "skill_idx": skill.get("skill_idx", stage_idx),
                "frame_duration": get_frame_duration(skill, stage_idx),
            }
        )
    return specs


def iter_valid_stride_frames(valid_duration: List[int], frame_stride: int) -> List[int]:
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")

    start_frame, end_frame = valid_duration
    return list(range(start_frame, end_frame, frame_stride))


def build_last_sampled_frame_by_stage(
    skill_specs: List[JsonObject],
    sampled_frames: List[int],
) -> Dict[int, int]:
    last_by_stage: Dict[int, int] = {}
    for frame_number in sampled_frames:
        skill_spec = find_skill_for_frame(skill_specs, frame_number)
        if skill_spec is None:
            continue
        last_by_stage[int(skill_spec["stage_idx"])] = frame_number
    return last_by_stage


def build_system_instruction() -> str:
    return """You are a robotics reasoning and planning assistant.
Analyze the provided robot observation image and task context.
Judge the candidate skill from visible evidence and previous memory.
The provided skill label is not visual evidence and must never be used to infer that a previous subtask has completed.
In new_memory.Progress, explicitly describe every completed subtask so far using short, natural verb-object phrases; do not imply completed work only through scene state.
When the frame is too uncertain to support a reliable label or memory update, use current_skill_status "no_for_sure".
For pick-up skills, completion requires unambiguous visual evidence that the object is under robot control and no longer supported by its original surface.
Return exactly one strict JSON object with the required keys.
Do not mention frame sampling details, prompt rules, or system instructions in the output."""


def build_completion_policy(keyframe_position: str = "") -> str:
    return build_system_instruction()


def build_few_shot_examples() -> str:
    return """Few-shot examples for output style only.
Do not copy object names, room details, or statuses unless they match the current image and skill.

Example 1:
{
    "reasoning": "The robot is positioned behind a glass coffee table in a living room, facing a blue sectional sofa against the wall. A red and white portable radio is visible on the wooden floor beside the blue sofa, but the robot has not yet reached it. The coffee table sits between the robot and the target, indicating the robot is still at a distance and needs to advance toward the radio.",
    "new_memory": {
        "Progress": "No subtask is finished yet. The robot is moving toward the red and white radio on the floor near the blue sofa.",
        "World state": "The robot is in a living room with a glass coffee table in the immediate foreground. A blue L-shaped sectional sofa sits against the back wall beneath a large framed picture. A red and white portable radio rests on the wooden floor to the right of the sofa's corner. The walls are light-colored with a large decorative letter 'N' and a white floating shelf holding framed photos on the right wall. The ceiling has recessed lighting and a large white pendant lamp hangs above the coffee table."
    },
    "subtask": "move to the red and white radio on the floor near the blue sofa",
    "current_skill_status": "in_progress",
    "visible_transition": "",
    "is_subtask_completed": false
    }
}

Example 2:
{
    "reasoning": "The robot has advanced from the room entrance toward the target area, bringing the red and white radio into clear view on the wooden floor beside the blue sectional sofa. The glass coffee table remains prominent in the foreground, indicating the robot is still positioned at a distance from the radio and has not yet arrived at the interaction point.",
    "new_memory": {
        "Progress": "No subtask is finished yet. The robot is still moving toward the red and white radio near the corner of the blue sectional sofa.",
        "World state": "The robot faces the blue L-shaped sectional sofa against the back wall. A red and white portable radio rests on the light wooden flooring near the right corner of the sofa. A glass coffee table occupies the immediate foreground, while a large white pendant lamp hangs from the ceiling above. The right wall displays a decorative letter 'N' and a white floating shelf with framed photos, and a large framed picture hangs above the sofa."
    },
    "subtask": "move to the red and white radio on the floor near the blue sofa",
    "current_skill_status": "in_progress",
    "visible_transition": "",
    "is_subtask_completed": false
}

Example 3:
{
    "reasoning": "The robot has moved from its previous position to a vantage point directly facing the red and white radio. The radio is now prominently positioned in the center of the view on the wooden floor, just beyond the glass coffee table and near the corner of the blue sectional sofa. Compared to the earlier state where the robot was still approaching, the radio is now clearly in close proximity and accessible, indicating the approach phase has concluded.",
    "new_memory": {
        "Progress": "Moved to the red and white radio on the floor near the blue sectional sofa.",
        "World state": "The robot is positioned near the red and white portable radio resting on the light wooden flooring at the corner of the blue L-shaped sectional sofa. A glass coffee table occupies the immediate foreground, a large white pendant lamp hangs overhead, and the right wall displays a decorative letter 'N' with a white floating shelf."
    },
    "subtask": "move to the red and white radio on the floor near the blue sectional sofa",
    "current_skill_status": "completed",
    "visible_transition": "",
    "is_subtask_completed": true
}

Example 4:
{
    "reasoning": "The robot is positioned at the glass coffee table with the red and white radio clearly visible resting on its reflective surface within immediate reach. This visible pose and object proximity show that the navigation-to-target subtask has reached its postcondition. The gripper is not yet visible around the radio, so the pick-up motion has not yet produced a visible grasp.",
    "new_memory": {
        "Progress": "Moved to the glass coffee table near the red and white radio. The robot is trying to pick up the radio from the table, but the grasp is not established yet.",
        "World state": "The red and white portable radio rests on the reflective glass coffee table in the foreground. Behind the table is a blue L-shaped sectional sofa against the back wall, with a large abstract painting above it. A large white pendant lamp hangs from the ceiling, and a decorative letter 'N' with a white floating shelf is mounted on the right wall. The floor is light wood, and the robot is facing the radio on the table."
    },
    "subtask": "pick up the red and white radio from the glass coffee table",
    "current_skill_status": "in_progress",
    "visible_transition": "",
    "is_subtask_completed": false
}

Example 5:
{
    "reasoning": "The gripper is close to the red and white radio and partly covers the contact region, while reflections on the glass make the area under the radio difficult to read. The radio may be grasped, but the base and underside are not clearly separated from the table surface, so the image does not provide reliable lift-off evidence.",
    "new_memory": {
        "Progress": "Moved to the glass coffee table near the red and white radio. The robot is grasping or reaching around the radio, but lift-off cannot be confirmed from this view.",
        "World state": "The red and white portable radio is at the glass coffee table with the robot gripper near or partly around it. The table surface is reflective, and the support/contact area below the radio is not clearly visible enough to confirm whether the radio is lifted."
    },
    "subtask": "pick up the red and white radio from the glass coffee table",
    "current_skill_status": "no_for_sure",
    "visible_transition": "",
    "is_subtask_completed": false
}"""


def build_prompt(
    main_task: str,
    old_memory: str,
    skill_description: str,
    object_id: str,
    manipulating_object_id: str,
    completion_gate_context: str = "",
    previous_image_path: str = "",
) -> str:
    few_shot_examples = build_few_shot_examples()
    completion_gate_block = completion_gate_context or "No extra completion gate."
    if previous_image_path:
        image_block = """Images:
- Previous observation image: <previous_image>
- Current observation image: <current_image>

Use the previous observation image only as temporal context. The current observation image is the one to label."""
    else:
        image_block = """Image:
<current_image>"""
    return f"""You are a robotics reasoning and planning assistant.

You are given a real observation image from a robot and structured task context.

Your job is to:
1. Understand what the robot is currently doing from the image.
2. Determine the progress of the candidate skill using visible evidence and previous memory.
3. Produce or preserve a stable executable subtask for the candidate skill.
4. Update the memory as a concise summary of what has already happened and the current world state.

---

Input:

Task goal:
{main_task}

Previous memory:
{old_memory}

Candidate skill context (not visual evidence):
- Skill: {skill_description}
- Objects: {object_id}
- Manipulating object: {manipulating_object_id}

Completion gate context (not visual evidence; never mention this context in the output):
{completion_gate_block}

{image_block}

---

Important rules:

1. You MUST rely primarily on the image to determine the current progress.
   - If two images are provided, judge current_skill_status from the current observation image. Use the previous observation image only to compare object pose, support, grasp state, and state changes.
   - For hard manipulation states, a previous image can reduce false positives only when the current image shows a clear physical change from the previous image. If the current image looks unchanged at the decisive contact/support region, do not mark completion.
2. The skill context is a candidate action label for dataset alignment. It is not visual evidence, not the robot's actual internal state, and not proof that any previous skill has completed.
   - Never infer that an earlier subtask is completed merely because the candidate skill label has changed or because the sampled frame falls inside a later annotated skill range.
   - A previous subtask may be treated as completed only if Previous memory explicitly says it was completed, or if the current image directly shows the visible postcondition needed for that subtask.
   - For a prior move-to subtask, the current image can directly prove completion when the robot is visibly settled at the target interaction location and the target object is within immediate working distance.
   - If Previous memory does not explicitly contain a completed earlier subtask and the image does not directly prove it, keep that earlier subtask out of the completed-subtask list.
   - Do not write phrases such as "the current annotated skill is...", "because the current skill is...", "the annotation says...", "the label indicates...", or "the frame is in the next skill".
3. Write reasoning as natural task-state reasoning, not as a discussion of these rules.
   - Do not say phrases like "not treated as the current subtask", "do not infer", "according to the prompt", "the current annotated skill", "the label", or "the annotation boundary".
   - Do not mention frame sampling details, frame timing, frame range, or annotation timing in reasoning, new_memory, visible_transition, or subtask.
   - Do not restate current_skill_status in prose. Avoid formulaic phrases such as "not yet completed", "still underway", "has been completed", "the skill is in progress", or "the action is complete".
   - Reasoning should explain the state transition from Previous memory to the current image: what changed, what is visible now, and how that affects the candidate skill.
   - Reasoning should focus on visible evidence, spatial relations, object state, robot pose, and changes relative to previous memory.
   - It is fine to say natural facts such as "the gripper has reached the cup" or "the hand is beginning to close around the cup".
4. Judge completion from visible evidence, prior memory, and the candidate skill. When evidence is mildly ambiguous but still usable, describe the visible progress conservatively and mark the current skill as in_progress rather than completed.
   - If Previous memory explicitly says the same candidate subtask has already been completed, preserve that completed state unless the image clearly shows failure, release, or loss of control. Temporary lowering, small pose changes, or partial occlusion should not undo a completed state.
   - If the current frame is too uncertain to be trusted for memory update or label generation, set current_skill_status to "no_for_sure". Use this when key evidence is occluded, outside the image, too blurry, color is uncertain, contact/separation cannot be judged, or a required state change cannot be confidently seen.
   - For hard-to-judge pick-up, lift, place, press, or switch states, use "no_for_sure" instead of "completed" when the decisive region is hidden, reflective, low-resolution, or interpretable in multiple ways.
   - For "no_for_sure", do not invent progress. Keep is_subtask_completed as false, visible_transition as "", and write new_memory as a conservative unchanged or minimally updated memory.
   - Follow the Completion gate context as an output constraint, but never refer to it in reasoning, new_memory, visible_transition, or subtask.
5. The "subtask" field must describe the candidate skill, not a future skill.
6. The subtask must be executable by a robot (VLA), not a passive description.
7. The subtask must contain ONE core skill (e.g., pick, move, press, spray).
8. The subtask must include a clear referring expression, not only a bare object category.
   - Use visible attributes such as color, position, size, shape, object state, relation to the robot, or relation to nearby objects.
   - Good examples: "pick up the red apple on the left side of the counter", "grasp the plate closest to the robot", "move to the small cup beside the bowl".
   - Bad examples: "pick up the apple", "move to the cup", "grasp the object".
9. You may mention visible transition after the current skill only in "visible_transition", using natural observation language.
10. Do NOT infer or invent a future skill from the image.
   - Do NOT write predicted future tasks into new_memory.
11. Treat memory as a Markov state update:
    - new_memory must be updated from Previous memory, the current image, the candidate skill, and the task goal.
    - Do not rely on unstated history outside Previous memory.
    - Do not include future action plans or predicted next skills.
12. new_memory.Progress must be a natural paragraph describing task progress up to the current moment.
    - It MUST explicitly name every subtask that has already been completed, including completed subtasks carried from Previous memory and the candidate skill if it is now completed.
    - Write completed subtasks as concise, natural action phrases such as "Moved to the glass coffee table; picked up the red and white radio; now pressing the top button." Avoid repetitive templates like "completed the X subtask by doing Y" unless that wording is genuinely needed for clarity.
    - Do not add a previous subtask to the completed list just because the candidate skill is a later action. Completion history must come from Previous memory or direct visual evidence.
    - Do not imply a completed subtask only through scene state. For example, do not rely on "the cup is on the table" to imply "the robot placed the cup on the table"; state the completed place subtask explicitly when it has been completed.
    - If no subtask has been completed yet, say that briefly, for example "No subtask is finished yet."
    - It should update the previous Progress using the current image while preserving the explicit list of completed subtasks.
    - It should summarize what has already happened and what is happening now for the current task.
    - When the current skill starts after earlier skills, carry forward a concise explicit summary of all earlier completed subtasks before describing the current activity.
    - If quantities or remaining task state are important, mention them naturally; do not force labels such as "Past", "Current", or "Remaining".
13. new_memory.World state must be an objective current-scene snapshot from the image.
    - Include relevant counts, colors, materials, locations, spatial relations, robot pose, gripper state, and object state when visible.
    - Prefer concrete scene details over vague descriptions: name how many relevant objects are visible, where they are, what they look like, and how the robot or gripper is positioned relative to them.
    - Do not include task bookkeeping, progress summaries, future plans, lessons, or instructions.
14. Memory must be compressed and structured, not a full log, and must follow this format:

Progress: ...
World state: ...

15. Determine whether the candidate skill is completed, not whether a future skill can start.
16. Use visible postconditions as guidance:
    - For "move to", target visibility, proximity, close-up view, or accessibility is not enough unless the robot appears settled at the target pose.
    - For "move to", if the Completion gate context says completion is not allowed for this observation, set current_skill_status to "in_progress" even when the robot appears close to the target. Describe only the visible approach or settling state; do not claim the move-to subtask is finished.
    - For "move to", if the Completion gate context says completion is allowed, output completed only when the robot is visibly settled at the target interaction location and the target object is within immediate working distance.
    - For "pick up", the object does not need to be raised high in the air, but approaching the object, hovering near it, touching it from the side, grasping a handle while the object still rests on the table, or partially covering it is not enough for completion.
    - Treat pick-up as completed only when the current image shows BOTH required facts: the target object is visibly under robot control, and the target object is clearly no longer supported by the table or original surface.
    - Evidence for robot control means the gripper visibly encloses, supports, pinches, or carries the object itself. A gripper merely near the object, beside it, above it, or touching only an ambiguous handle/edge is not enough.
    - Evidence for lift-off must be physical and unambiguous: the base/underside is visibly separated from the original support, the object is being carried away from the support, or a consistent clear gap can be seen along the support/contact region.
    - A tiny apparent gap, one bright highlight, a dark shadow line, glare, reflection, perspective distortion, or partial occlusion is not enough to conclude lift-off.
    - For glass or reflective tables, be extra conservative: do not treat tabletop reflections, lamp projections, shadows, white highlights, duplicated object reflections, or glare as a gap, gripper, or lifted underside.
    - If the object is still clearly resting on the table or original support, keep pick-up in_progress.
    - If the object may be grasped but contact/separation from the original support cannot be confidently judged, return "no_for_sure" rather than guessing completion.
    - If two images are provided and the object's base/contact region looks unchanged from the previous observation, do not mark pick-up completed; use in_progress if it is clearly still supported, otherwise use no_for_sure.
    - For the radio on the glass coffee table, if the radio body/base is still touching or resting on the glass table, keep pick-up in_progress even if the gripper has closed on the handle or side.
    - For the radio on the glass coffee table, only mark pick-up completed when the current image clearly shows the radio body itself lifted or carried, not merely the handle or top edge obscured by the gripper.
    - If the gripper is still open, merely approaching, or only adjacent to the object, keep pick-up in_progress.
    - If a later frame shows the held object lower than before but still under robot control, keep the pick-up completed.
    - For "place", the object must be visibly released and resting at the target location.
    - For "open", "close", "press", "turn on", or "turn off", the resulting state must be visibly changed.
    - For this radio task, pressing or turning on the radio is completed only when the button or indicator is clearly and unambiguously visible as green after the press. A gripper retracting away from the button is not enough evidence by itself.
    - If the button/indicator is occluded by the gripper, only partly visible, darkened by shadow, or ambiguous between red/black/dark green, return "no_for_sure" rather than guessing the color. Do not describe an unclear or occluded button as green just to satisfy the completion condition.
    - If the robot is actively pressing and the color change is not clearly visible yet, return "no_for_sure" when the frame cannot reliably support a label; otherwise treat it as in_progress.
17. Set current_skill_status to exactly one of:
    - "not_started"
    - "in_progress"
    - "completed"
    - "completed_and_transitioning"
    - "no_for_sure"
18. Set is_subtask_completed to true only when current_skill_status is "completed" or "completed_and_transitioning". Set it to false when current_skill_status is "no_for_sure".
19. Let current_skill_status and is_subtask_completed carry the explicit completion label. In reasoning and new_memory.Progress, describe the visual evidence and progress state without using repetitive completion-label phrasing.

---

{few_shot_examples}

---

Output format (strict JSON, no markdown):

{{
  "reasoning": "...",
  "new_memory": {{
    "Progress": "...",
    "World state": "..."
  }},
  "subtask": "...",
  "current_skill_status": "in_progress",
  "visible_transition": "",
  "is_subtask_completed": true or false
}}

Return ONLY valid JSON.
"""


def build_request_input(
    image_path: str,
    main_task: str,
    old_memory: str,
    skill_description: str,
    object_id: str,
    manipulating_object_id: str,
    completion_gate_context: str = "",
    previous_image_path: str = "",
) -> JsonObject:
    request_input = {
        "image_path": image_path,
        "main_task": main_task,
        "old_memory": old_memory,
        "skill_description": skill_description,
        "object_id": object_id,
        "manipulating_object_id": manipulating_object_id,
    }
    if previous_image_path:
        request_input["previous_image_path"] = previous_image_path
    if completion_gate_context:
        request_input["completion_gate_context"] = completion_gate_context
    return request_input


def build_error_record(
    *,
    skill_idx: Any,
    stage_idx: int,
    image_dir_name: str,
    image_path_text: str,
    original_image_idx: int,
    frame_number: int,
    frame_duration: List[int],
    keyframe_position: str,
    frame_stride: int,
    skill: JsonObject,
    request_input: JsonObject,
    exc: Exception,
) -> JsonObject:
    error: JsonObject = {
        "type": exc.__class__.__name__,
        "message": str(exc),
    }
    if isinstance(exc, ModelResponseFormatError):
        error["raw_response"] = exc.content

    return {
        "skill_idx": skill_idx,
        "stage_idx": stage_idx,
        "stage_dir": image_dir_name,
        "image_dir": image_dir_name,
        "image_path": image_path_text,
        "image_index_in_stage": original_image_idx - 1,
        "frame_number": frame_number,
        "frame_duration": frame_duration,
        "frame_selection": build_frame_selection_label(frame_stride),
        "frame_stride": frame_stride,
        "keyframe_position": keyframe_position,
        "frame_sample_label": keyframe_position,
        "skill_description": skill.get("skill_description", []),
        "subtask_description": skill.get("subtask_description", ""),
        "object_id": skill.get("object_id", []),
        "manipulating_object_id": skill.get("manipulating_object_id", []),
        "request_input": request_input,
        "model_response": None,
        "result_used": False,
        "error": error,
    }


def memory_to_text(memory: Any) -> str:
    if not memory:
        return ""
    if isinstance(memory, str):
        return memory
    return json.dumps(memory, ensure_ascii=False)


def is_completed_status(value: Any) -> bool:
    return value in {"completed", "completed_and_transitioning"}


def is_no_for_sure_status(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
    return normalized == NO_FOR_SURE_STATUS


def is_move_to_skill(
    *,
    skill_description: str,
    object_id: str = "",
    manipulating_object_id: str = "",
) -> bool:
    context = " ".join([skill_description, object_id, manipulating_object_id]).lower()
    return "move to" in context or "navigate to" in context


def should_preserve_completed_state_for_skill(
    *,
    skill_description: str,
    object_id: str,
    manipulating_object_id: str,
) -> bool:
    context = " ".join([skill_description, object_id, manipulating_object_id]).lower()
    return any(action in context for action in ("pick", "pick up", "grasp", "lift"))


def parse_memory_text(memory_text: str) -> JsonObject:
    if not memory_text or not memory_text.strip():
        return {}
    try:
        parsed = json.loads(memory_text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        return parsed

    memory: JsonObject = {}
    for line in memory_text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip().lower()
        if normalized_key == "progress":
            memory["Progress"] = value.strip()
        elif normalized_key == "world state":
            memory["World state"] = value.strip()
    return memory


def build_premature_move_to_memory(
    *,
    previous_memory_text: str,
    raw_result: JsonObject,
) -> JsonObject:
    previous_memory = parse_memory_text(previous_memory_text)
    raw_memory = raw_result.get("new_memory")
    if not isinstance(raw_memory, dict):
        raw_memory = {}

    progress = previous_memory.get("Progress")
    if not isinstance(progress, str) or not progress.strip():
        progress = "No subtask is finished yet."

    progress_lower = progress.lower()
    if "moving toward" not in progress_lower and "settling" not in progress_lower:
        progress = (
            progress.rstrip()
            + " The robot is still moving toward or settling at the target interaction location."
        )

    world_state = raw_memory.get("World state") or previous_memory.get("World state") or ""
    return {
        "Progress": progress,
        "World state": world_state,
    }


def suppress_premature_move_to_completion(
    result: JsonObject,
    *,
    previous_memory_text: str,
) -> Tuple[JsonObject, JsonObject]:
    guarded = copy.deepcopy(result)
    original_status = guarded.get("current_skill_status")
    original_is_completed = guarded.get("is_subtask_completed")

    guarded["reasoning"] = (
        "The robot is near the target and still settling at the interaction pose, "
        "so the move-to step remains active for this observation."
    )
    guarded["current_skill_status"] = "in_progress"
    guarded["is_subtask_completed"] = False
    guarded["visible_transition"] = ""
    guarded["new_memory"] = build_premature_move_to_memory(
        previous_memory_text=previous_memory_text,
        raw_result=result,
    )

    return guarded, {
        "type": "suppress_premature_move_to_completion",
        "original_current_skill_status": original_status,
        "original_is_subtask_completed": original_is_completed,
        "reason": (
            "For move-to skills, completed is only emitted on the last sampled request "
            "inside that skill's frame_duration."
        ),
    }


def get_move_to_target_phrase(result: JsonObject) -> str:
    subtask = result.get("subtask")
    if isinstance(subtask, str) and subtask.strip():
        text = subtask.strip().rstrip(".")
        lowered = text.lower()
        if lowered.startswith("move to "):
            return text[len("move to ") :].strip()
        if lowered.startswith("navigate to "):
            return text[len("navigate to ") :].strip()
        return text
    return "the target interaction location"


def build_move_to_completed_phrase(result: JsonObject) -> str:
    target = get_move_to_target_phrase(result)
    if target == "the target interaction location":
        return "Moved to the target interaction location."
    return f"Moved to {target}."


def remove_conflicting_move_to_progress(progress: str) -> str:
    if not isinstance(progress, str) or not progress.strip():
        return ""

    conflict_markers = (
        "no subtask is finished yet",
        "moving toward",
        "move toward",
        "still moving",
        "approaching",
        "still approaching",
        "settling at",
        "trying to move",
        "has not reached",
        "not yet reached",
        "not finished",
        "not complete",
        "in progress",
    )
    sentences = re.split(r"(?<=[.!?;])\s+", progress.strip())
    kept: List[str] = []
    for sentence in sentences:
        lowered = sentence.strip().lower()
        if not lowered:
            continue
        if any(marker in lowered for marker in conflict_markers):
            continue
        kept.append(sentence.strip())
    return " ".join(kept)


def build_completed_move_to_memory(
    *,
    previous_memory_text: str,
    raw_result: JsonObject,
) -> JsonObject:
    previous_memory = parse_memory_text(previous_memory_text)
    raw_memory = raw_result.get("new_memory")
    if not isinstance(raw_memory, dict):
        raw_memory = {}

    completed_phrase = build_move_to_completed_phrase(raw_result)

    previous_progress = previous_memory.get("Progress")
    if not isinstance(previous_progress, str) or not previous_progress.strip():
        previous_progress = ""
    previous_progress = remove_conflicting_move_to_progress(previous_progress)

    progress_lower = previous_progress.lower()
    completed_lower = completed_phrase.lower().rstrip(".")
    if completed_lower in progress_lower:
        progress = previous_progress
    elif not previous_progress or previous_progress.lower().startswith("no subtask is finished yet"):
        progress = completed_phrase.rstrip(".") + "."
    else:
        progress = previous_progress.rstrip()
        if not progress.endswith((".", ";")):
            progress += "."
        progress += " " + completed_phrase.rstrip(".") + "."

    world_state = raw_memory.get("World state") or previous_memory.get("World state") or ""
    return {
        "Progress": progress,
        "World state": world_state,
    }


def build_completed_move_to_reasoning(result: JsonObject) -> str:
    target = get_move_to_target_phrase(result)
    if target == "the target interaction location":
        return (
            "The robot is positioned at the target interaction location, with the workspace "
            "ready for the next manipulation step. This satisfies the move-to postcondition."
        )
    return (
        f"The robot is positioned at the target interaction location for {target}, "
        "with the target reachable for the next manipulation step. This satisfies the move-to postcondition."
    )


def force_final_move_to_completion(
    result: JsonObject,
    *,
    previous_memory_text: str,
) -> Tuple[JsonObject, Optional[JsonObject]]:
    if is_completed_status(result.get("current_skill_status")):
        return result, None

    forced = copy.deepcopy(result)
    original_status = forced.get("current_skill_status")
    original_is_completed = forced.get("is_subtask_completed")

    forced["reasoning"] = build_completed_move_to_reasoning(result)
    forced["current_skill_status"] = "completed"
    forced["is_subtask_completed"] = True
    forced["visible_transition"] = ""
    forced["new_memory"] = build_completed_move_to_memory(
        previous_memory_text=previous_memory_text,
        raw_result=result,
    )

    return forced, {
        "type": "force_final_move_to_completion",
        "original_current_skill_status": original_status,
        "original_is_subtask_completed": original_is_completed,
        "reason": (
            "For move-to skills, the final sampled request inside the skill's frame_duration "
            "is forced to completed so only the last request emits completion."
        ),
    }


def build_move_to_completion_gate_context(
    *,
    is_move_to: bool,
    is_last_sampled_request_for_skill: bool,
) -> str:
    if not is_move_to:
        return ""
    if is_last_sampled_request_for_skill:
        return (
            "Move-to completion is required for this observation. For the candidate move-to "
            "skill, set current_skill_status to completed and is_subtask_completed to true. "
            "Describe the robot as having reached the target interaction pose using natural "
            "visual language. The reasoning and new_memory.Progress must be consistent with "
            "completion: do not say the robot is still approaching, still moving toward, "
            "not yet at the target, or that no subtask is finished."
        )
    return (
        "Move-to completion is not allowed for this observation. For the candidate move-to "
        "skill, keep current_skill_status as in_progress unless the frame is too uncertain "
        "and should be no_for_sure. Do not output completed or completed_and_transitioning."
    )


def get_move_to_gate_violation(
    result: JsonObject,
    *,
    is_move_to: bool,
    is_last_sampled_request_for_skill: bool,
) -> Optional[str]:
    if not is_move_to:
        return None
    if is_no_for_sure_status(result.get("current_skill_status")):
        if is_last_sampled_request_for_skill:
            return "missing_final_completed"
        return None
    is_completed = is_completed_status(result.get("current_skill_status"))
    if not is_last_sampled_request_for_skill and is_completed:
        return "premature_completed"
    if is_last_sampled_request_for_skill and not is_completed:
        return "missing_final_completed"
    return None


def build_move_to_gate_retry_context(
    *,
    violation: str,
    is_last_sampled_request_for_skill: bool,
) -> str:
    base_context = build_move_to_completion_gate_context(
        is_move_to=True,
        is_last_sampled_request_for_skill=is_last_sampled_request_for_skill,
    )
    if violation == "premature_completed":
        correction = (
            "The previous answer violated the move-to completion gate by marking this "
            "non-final move-to observation as completed. Regenerate the full JSON from the image. "
            "Use current_skill_status in_progress unless the frame is too uncertain and should be no_for_sure. "
            "Make reasoning and new_memory.Progress consistent with the skill still being active."
        )
    elif violation == "missing_final_completed":
        correction = (
            "The previous answer violated the move-to completion gate by failing to mark this "
            "final move-to observation as completed. Regenerate the full JSON from the image. "
            "Use current_skill_status completed and is_subtask_completed true. "
            "Make reasoning and new_memory.Progress consistent with the robot having reached the target."
        )
    else:
        correction = "Regenerate the full JSON and follow the move-to completion gate exactly."
    return f"{base_context}\n{correction}"


def has_pickup_incomplete_evidence(result: JsonObject) -> bool:
    response_text = collect_response_text(result).lower()
    incomplete_markers = (
        "still resting on the table",
        "still resting on the glass",
        "resting on the table",
        "resting on the glass",
        "remains on the table",
        "remains on the glass",
        "on the table surface",
        "gripper is not visible",
        "gripper is not contacting",
        "not contacting",
        "not yet grasp",
        "not yet been grasp",
        "grasp is not established",
        "has not yet produced a visible grasp",
        "has not yet been successfully executed",
        "not lifted",
        "has not been lifted",
        "not separated",
        "approaching",
        "still approaching",
        "merely adjacent",
    )
    return any(marker in response_text for marker in incomplete_markers)


def preserve_completed_state(
    result: JsonObject,
    *,
    already_completed: bool,
) -> Tuple[JsonObject, Optional[JsonObject]]:
    if not already_completed or is_completed_status(result.get("current_skill_status")):
        return result, None
    if has_pickup_incomplete_evidence(result):
        return result, None

    smoothed = copy.deepcopy(result)
    original_status = smoothed.get("current_skill_status")
    original_is_completed = smoothed.get("is_subtask_completed")
    smoothed["current_skill_status"] = "completed"
    smoothed["is_subtask_completed"] = True

    reasoning = smoothed.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        smoothed["reasoning"] = (
            "The object was already picked up earlier and remains under robot control; "
            "a lower pose or partial occlusion does not undo the pick-up. "
            + reasoning
        )

    new_memory = smoothed.get("new_memory")
    if isinstance(new_memory, dict):
        progress = new_memory.get("Progress")
        if isinstance(progress, str) and progress.strip():
            new_memory["Progress"] = (
                "Picked up the object; it remains under robot control despite the lower pose. "
                + progress
            )

    return smoothed, {
        "type": "preserve_completed_state",
        "original_current_skill_status": original_status,
        "original_is_subtask_completed": original_is_completed,
        "reason": (
            "A previous frame for this pick/grasp/lift skill was completed; later frames are "
            "not allowed to regress unless there is clear release, loss of control, or failure."
        ),
    }


def build_episode_output(
    annotation_json: Path,
    image_root: Path,
    task_name: str,
    records: List[JsonObject],
    *,
    frame_selection: str = FRAME_SELECTION,
    interrupted: bool = False,
) -> JsonObject:
    return {
        "annotation_json": str(annotation_json),
        "image_root": str(image_root),
        "task_name": task_name,
        "frame_selection": frame_selection,
        "interrupted": interrupted,
        "processed_count": len(records),
        "used_count": sum(1 for record in records if record.get("result_used") is True),
        "discarded_count": sum(1 for record in records if record.get("result_used") is False),
        "no_for_sure_count": sum(
            1 for record in records if record.get("discard_reason") == NO_FOR_SURE_STATUS
        ),
        "move_to_guard_count": sum(1 for record in records if record.get("move_to_completion_guard")),
        "error_count": sum(1 for record in records if record.get("error")),
        "results": records,
    }


def save_episode_output(output_path: Optional[Path], output: JsonObject) -> bool:
    if output_path is None:
        return False

    write_json(output_path, output)
    print(f"[saved] {output_path} ({output['processed_count']} result(s))", flush=True)
    return True


def has_subtask_images_root(path: Path) -> bool:
    return any((path / f"{prefix}_00").is_dir() for prefix in SUBTASK_IMAGE_DIR_PREFIXES)


def find_annotation_jsons(annotation_path: Path) -> List[Path]:
    if annotation_path.is_file():
        return [annotation_path]
    if not annotation_path.is_dir():
        raise FileNotFoundError(f"Annotation JSON path does not exist: {annotation_path}")

    annotation_jsons = sorted(
        path for path in annotation_path.glob("episode_*.json") if path.is_file()
    )
    if not annotation_jsons:
        raise FileNotFoundError(f"No episode_*.json files found in: {annotation_path}")
    return annotation_jsons


def resolve_episode_image_root(
    annotation_json: Path,
    image_root: Path,
    *,
    multiple_episodes: bool,
) -> Path:
    if not multiple_episodes:
        return image_root

    episode_image_root = image_root / annotation_json.stem
    if has_subtask_images_root(episode_image_root):
        return episode_image_root

    if has_subtask_images_root(image_root):
        raise ValueError(
            "Directory annotation mode needs --image-root to contain one image folder per "
            f"episode, for example: {image_root / annotation_json.stem / 'stage_00'} "
            f"or {image_root / annotation_json.stem / 'skill_00'}"
        )

    raise FileNotFoundError(
        f"Could not locate image root for {annotation_json.name}. Expected: "
        f"{episode_image_root / 'stage_00'} or {episode_image_root / 'skill_00'}"
    )


def build_annotation_directory_output(
    annotation_path: Path,
    image_root: Path,
    episodes: List[JsonObject],
    *,
    interrupted: bool = False,
) -> JsonObject:
    return {
        "annotation_json": str(annotation_path),
        "image_root": str(image_root),
        "interrupted": interrupted,
        "episode_count": len(episodes),
        "processed_count": sum(int(episode.get("processed_count", 0)) for episode in episodes),
        "used_count": sum(int(episode.get("used_count", 0)) for episode in episodes),
        "discarded_count": sum(int(episode.get("discarded_count", 0)) for episode in episodes),
        "no_for_sure_count": sum(int(episode.get("no_for_sure_count", 0)) for episode in episodes),
        "move_to_guard_count": sum(
            int(episode.get("move_to_guard_count", 0)) for episode in episodes
        ),
        "error_count": sum(int(episode.get("error_count", 0)) for episode in episodes),
        "episodes": episodes,
    }


def get_episode_output_path(output_path: Optional[Path], annotation_json: Path) -> Optional[Path]:
    if output_path is None or output_path.suffix.lower() == ".json":
        return None
    return output_path / f"{annotation_json.stem}.json"


def is_resume_complete_episode_output(output: Any) -> bool:
    if not isinstance(output, dict):
        return False
    try:
        error_count = int(output.get("error_count", 0))
    except (TypeError, ValueError):
        return False
    return not output.get("interrupted") and error_count == 0


def can_resume_episode_output(output_path: Path) -> bool:
    try:
        output = read_json(output_path)
    except (OSError, json.JSONDecodeError):
        return False
    return is_resume_complete_episode_output(output)


def load_resume_aggregate_outputs(aggregate_output: Optional[Path]) -> Dict[str, JsonObject]:
    if aggregate_output is None or not aggregate_output.exists():
        return {}

    try:
        aggregate = read_json(aggregate_output)
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(aggregate, dict):
        return {}

    episodes = aggregate.get("episodes")
    if not isinstance(episodes, list):
        return {}

    resumable: Dict[str, JsonObject] = {}
    for episode in episodes:
        if not is_resume_complete_episode_output(episode):
            continue

        annotation_json = episode.get("annotation_json")
        if not annotation_json:
            continue
        resumable[Path(str(annotation_json)).name] = episode

    return resumable


class _LocalBase:
    pass


base = _LocalBase()
base.FRAME_SELECTION = FRAME_SELECTION
base.ModelResponseFormatError = ModelResponseFormatError
base.read_json = read_json
base.write_json = write_json
base.format_annotation_value = format_annotation_value
base.get_skills = get_skills
base.get_frame_duration = get_frame_duration
base.get_valid_duration = get_valid_duration
base.find_stage_images = find_stage_images
base.build_frame_image_map = build_frame_image_map
base.build_frame_selection_label = build_frame_selection_label
base.build_skill_specs = build_skill_specs
base.find_skill_for_frame = find_skill_for_frame
base.iter_valid_stride_frames = iter_valid_stride_frames
base.build_last_sampled_frame_by_stage = build_last_sampled_frame_by_stage
base.build_system_instruction = build_system_instruction
base.build_completion_policy = build_completion_policy
base.build_prompt = build_prompt
base.build_request_input = build_request_input
base.build_error_record = build_error_record
base.memory_to_text = memory_to_text
base.is_completed_status = is_completed_status
base.is_no_for_sure_status = is_no_for_sure_status
base.is_move_to_skill = is_move_to_skill
base.should_preserve_completed_state_for_skill = should_preserve_completed_state_for_skill
base.suppress_premature_move_to_completion = suppress_premature_move_to_completion
base.force_final_move_to_completion = force_final_move_to_completion
base.build_move_to_completion_gate_context = build_move_to_completion_gate_context
base.get_move_to_gate_violation = get_move_to_gate_violation
base.build_move_to_gate_retry_context = build_move_to_gate_retry_context
base.preserve_completed_state = preserve_completed_state
base.build_episode_output = build_episode_output
base.save_episode_output = save_episode_output
base.find_annotation_jsons = find_annotation_jsons
base.resolve_episode_image_root = resolve_episode_image_root
base.get_episode_output_path = get_episode_output_path
base.can_resume_episode_output = can_resume_episode_output
base.load_resume_aggregate_outputs = load_resume_aggregate_outputs
base.build_annotation_directory_output = build_annotation_directory_output


def make_client() -> Any:
    if genai is None:
        raise ImportError(
            "Missing google-genai package. Install it with: pip install google-genai"
        )

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing environment variable: GEMINI_API_KEY or GOOGLE_API_KEY")

    return genai.Client(api_key=api_key)


def image_to_part(image_path: str) -> Any:
    if types is None:
        raise ImportError(
            "Missing google-genai package. Install it with: pip install google-genai"
        )

    path = Path(image_path)
    mime_type = IMAGE_MIME_BY_EXT.get(path.suffix.lower(), "image/jpeg")
    with path.open("rb") as f:
        image_bytes = f.read()

    return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)


def strip_json_fence(content: str) -> str:
    stripped = content.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def parse_json_response(content: str) -> JsonObject:
    if not content or not content.strip():
        raise ModelResponseFormatError("Model returned an empty response", content)

    try:
        parsed = json.loads(strip_json_fence(content))
    except json.JSONDecodeError as exc:
        raise ModelResponseFormatError("Model returned invalid JSON", content) from exc

    if not isinstance(parsed, dict):
        raise ModelResponseFormatError("Model returned JSON that is not an object", content)

    missing_keys = sorted(REQUIRED_RESPONSE_KEYS - parsed.keys())
    if missing_keys:
        raise ModelResponseFormatError(
            f"Model returned JSON missing required keys {missing_keys}",
            content,
        )

    if is_no_for_sure_status(parsed.get("current_skill_status")):
        parsed["current_skill_status"] = NO_FOR_SURE_STATUS
        parsed["is_subtask_completed"] = False
    elif parsed.get("current_skill_status") not in VALID_CURRENT_SKILL_STATUSES:
        raise ModelResponseFormatError(
            "Model returned invalid current_skill_status",
            content,
        )

    prompt_leak = find_prompt_leak(parsed)
    if prompt_leak is not None:
        raise ModelResponseFormatError(
            f"Model response mentioned prompt or annotation metadata ({prompt_leak})",
            content,
        )

    return parsed


def collect_response_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(collect_response_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(collect_response_text(item) for item in value)
    return ""


def find_prompt_leak(parsed: JsonObject) -> Optional[str]:
    checked_text = " ".join(
        collect_response_text(parsed.get(key))
        for key in ("reasoning", "new_memory", "visible_transition")
    ).lower()
    for marker in PROMPT_LEAK_MARKERS:
        if marker in checked_text:
            return marker
    return None


def build_generation_config(
    *,
    system_instruction: str,
    temperature: float,
    max_output_tokens: Optional[int],
    json_mode: bool,
    include_thoughts: bool,
    thinking_level: Optional[str],
    thinking_budget: Optional[int],
) -> Any:
    if types is None:
        raise ImportError(
            "Missing google-genai package. Install it with: pip install google-genai"
        )

    config_kwargs: JsonObject = {
        "system_instruction": system_instruction,
        "temperature": temperature,
    }
    if max_output_tokens is not None:
        config_kwargs["max_output_tokens"] = max_output_tokens
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    thinking_kwargs: JsonObject = {}
    if include_thoughts:
        thinking_kwargs["include_thoughts"] = True
    if thinking_level:
        thinking_kwargs["thinking_level"] = thinking_level
    if thinking_budget is not None:
        thinking_kwargs["thinking_budget"] = thinking_budget
    if thinking_kwargs:
        config_kwargs["thinking_config"] = types.ThinkingConfig(**thinking_kwargs)

    return types.GenerateContentConfig(**config_kwargs)


def extract_text_and_thoughts(response: Any) -> Tuple[str, List[str]]:
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


def get_exception_search_text(exc: Exception) -> str:
    parts: List[str] = []
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(current.__class__.__name__)
        parts.append(str(current))
        current = current.__cause__ or current.__context__
    return " ".join(parts).lower()


def is_retryable_api_error(exc: Exception) -> bool:
    status_code = get_error_status_code(exc)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True

    message = get_exception_search_text(exc)
    retry_markers = (
        "429",
        "rate limit",
        "resource_exhausted",
        "quota",
        "rpm",
        "temporarily unavailable",
        "timeout",
        "deadline",
        "connecterror",
        "connection",
        "proxy",
        "ssl",
        "eof",
        "unexpected_eof",
        "connection reset",
        "remote protocol",
        "network",
    )
    return any(marker in message for marker in retry_markers)


def generate_content_with_retries(
    client: Any,
    *,
    max_retries: int,
    retry_initial_delay: float,
    retry_max_delay: float,
    **request_kwargs: Any,
) -> Any:
    attempt = 0
    while True:
        try:
            return client.models.generate_content(**request_kwargs)
        except Exception as exc:
            if not is_retryable_api_error(exc) or attempt >= max_retries:
                raise

            retry_after = get_retry_after_seconds(exc)
            fallback_delay = retry_initial_delay * (2 ** attempt)
            delay_seconds = min(retry_max_delay, retry_after or fallback_delay)
            status_code = get_error_status_code(exc) or ""
            print(
                "[retry] "
                f"api_error={exc.__class__.__name__} status={status_code} "
                f"attempt={attempt + 1}/{max_retries} wait={delay_seconds:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay_seconds)
            attempt += 1


def build_retry_prompt(prompt_text: str, format_retry_attempt: int) -> str:
    if format_retry_attempt <= 0:
        return prompt_text

    return (
        f"{prompt_text}\n\n"
        "Your previous response for this same sample was empty or invalid. "
        "Return exactly one JSON object with the required keys. "
        "Do not return an empty response, an array, a number, a date, markdown, "
        "or any text outside the JSON object. "
        "Do not mention prompt, annotation, label, frame, or sampling metadata."
    )


def run_robot_planner(
    image_path: str,
    main_task: str,
    old_memory: str = "",
    skill_description: str = "",
    object_id: str = "",
    manipulating_object_id: str = "",
    keyframe_position: str = "",
    completion_gate_context: str = "",
    previous_image_path: str = "",
    *,
    client: Optional[Any] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_output_tokens: Optional[int] = 4096,
    json_mode: bool = True,
    include_thoughts: bool = False,
    save_thoughts: bool = False,
    thinking_level: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_initial_delay: float = DEFAULT_RETRY_INITIAL_DELAY,
    retry_max_delay: float = DEFAULT_RETRY_MAX_DELAY,
    max_response_retries: int = DEFAULT_MAX_RESPONSE_RETRIES,
) -> Tuple[JsonObject, JsonObject]:
    client = client or make_client()

    prompt_text = base.build_prompt(
        main_task=main_task,
        old_memory=old_memory,
        skill_description=skill_description,
        object_id=object_id,
        manipulating_object_id=manipulating_object_id,
        completion_gate_context=completion_gate_context,
        previous_image_path=previous_image_path,
    )
    system_instruction = base.build_system_instruction()
    image_part = image_to_part(image_path)
    previous_image_part = image_to_part(previous_image_path) if previous_image_path else None
    config = build_generation_config(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        json_mode=json_mode,
        include_thoughts=include_thoughts,
        thinking_level=thinking_level,
        thinking_budget=thinking_budget,
    )

    for format_attempt in range(max_response_retries + 1):
        contents: List[Any] = [build_retry_prompt(prompt_text, format_attempt)]
        if previous_image_part is not None:
            contents.extend(
                [
                    "Previous observation image:",
                    previous_image_part,
                    "Current observation image:",
                    image_part,
                ]
            )
        else:
            contents.append(image_part)

        response = generate_content_with_retries(
            client,
            max_retries=max_retries,
            retry_initial_delay=retry_initial_delay,
            retry_max_delay=retry_max_delay,
            model=model,
            contents=contents,
            config=config,
        )
        response_text, thought_summaries = extract_text_and_thoughts(response)
        try:
            result = parse_json_response(response_text)
        except ModelResponseFormatError as exc:
            if format_attempt >= max_response_retries:
                raise

            preview = exc.content.replace("\n", " ")[:120]
            print(
                "[retry] "
                f"invalid_or_empty_model_response "
                f"attempt={format_attempt + 1}/{max_response_retries} "
                f"content={preview!r}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(min(10.0, 1.0 + format_attempt))
            continue

        metadata: JsonObject = {}
        if save_thoughts and thought_summaries:
            metadata["thought_summaries"] = thought_summaries

        return result, metadata

    raise RuntimeError("unreachable")


def run_episode_planner(
    annotation_json: Path,
    image_root: Path,
    *,
    output_path: Optional[Path] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    max_output_tokens: Optional[int] = 4096,
    json_mode: bool = True,
    include_thoughts: bool = False,
    save_thoughts: bool = False,
    thinking_level: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    request_delay: float = 0.0,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_initial_delay: float = DEFAULT_RETRY_INITIAL_DELAY,
    retry_max_delay: float = DEFAULT_RETRY_MAX_DELAY,
    max_response_retries: int = DEFAULT_MAX_RESPONSE_RETRIES,
    continue_on_model_error: bool = True,
    include_previous_image: bool = False,
) -> JsonObject:
    annotation = base.read_json(annotation_json)
    if not isinstance(annotation, dict):
        raise ValueError(f"Annotation JSON is not an object: {annotation_json}")

    task_name = base.format_annotation_value(annotation.get("task_name"))
    skills = base.get_skills(annotation)
    valid_duration = base.get_valid_duration(annotation)
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")
    frame_selection = base.build_frame_selection_label(frame_stride)
    skill_specs = base.build_skill_specs(skills)
    sampled_frames = base.iter_valid_stride_frames(valid_duration, frame_stride)
    last_sampled_frame_by_stage = base.build_last_sampled_frame_by_stage(
        skill_specs,
        sampled_frames,
    )
    client = make_client()

    old_memory = ""
    records: List[JsonObject] = []
    image_cache: Dict[int, Tuple[List[Path], Dict[int, Path]]] = {}
    completed_stage_indices: set[int] = set()
    previous_request_image_path = ""

    try:
        for sampled_idx, frame_number in enumerate(sampled_frames, start=1):
            skill_spec = base.find_skill_for_frame(skill_specs, frame_number)
            if skill_spec is None:
                print(
                    "[skip] "
                    f"frame={frame_number} has no matching skill frame_duration",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            stage_idx = int(skill_spec["stage_idx"])
            skill = skill_spec["skill"]
            skill_idx = skill_spec["skill_idx"]
            frame_duration = skill_spec["frame_duration"]
            skill_description = base.format_annotation_value(skill.get("skill_description"))
            object_id = base.format_annotation_value(skill.get("object_id"))
            manipulating_object_id = base.format_annotation_value(
                skill.get("manipulating_object_id")
            )
            is_move_to = base.is_move_to_skill(
                skill_description=skill_description,
                object_id=object_id,
                manipulating_object_id=manipulating_object_id,
            )
            is_last_sampled_request_for_skill = (
                frame_number == last_sampled_frame_by_stage.get(stage_idx)
            )
            completion_gate_context = base.build_move_to_completion_gate_context(
                is_move_to=is_move_to,
                is_last_sampled_request_for_skill=is_last_sampled_request_for_skill,
            )

            if stage_idx not in image_cache:
                images = base.find_stage_images(image_root, stage_idx)
                frame_images = base.build_frame_image_map(images)
                if not frame_images:
                    stage_dir = images[0].parent if images else "<unknown>"
                    raise FileNotFoundError(
                        f"No frame_000000-style image files found in subtask image directory: {stage_dir}"
                    )
                image_cache[stage_idx] = (images, frame_images)
            images, frame_images = image_cache[stage_idx]

            image_path = frame_images.get(frame_number)
            if image_path is None:
                stage_dir = images[0].parent if images else image_root / f"skill_{stage_idx:02d}"
                expected_stem = f"frame_{frame_number:06d}"
                raise FileNotFoundError(
                    f"Missing sampled frame image {expected_stem}.* in {stage_dir}"
                )

            original_image_idx = images.index(image_path) + 1
            frame_sample_label = f"stride_{sampled_idx:06d}"
            image_path_text = str(image_path)
            previous_image_path = previous_request_image_path if include_previous_image else ""
            image_dir_name = image_path.parent.name
            request_old_memory = old_memory
            request_input = base.build_request_input(
                image_path=image_path_text,
                main_task=task_name,
                old_memory=request_old_memory,
                skill_description=skill_description,
                object_id=object_id,
                manipulating_object_id=manipulating_object_id,
                completion_gate_context=completion_gate_context,
                previous_image_path=previous_image_path,
            )
            print(
                "[processing] "
                f"skill_idx={skill_idx} image_dir={image_dir_name} "
                f"sample={sampled_idx}/{len(sampled_frames)} stride={frame_stride} "
                f"frame={frame_number} image={original_image_idx}/{len(images)} path={image_path}",
                flush=True,
            )

            try:
                raw_result, response_metadata = run_robot_planner(
                    image_path=image_path_text,
                    main_task=task_name,
                    old_memory=request_old_memory,
                    skill_description=skill_description,
                    object_id=object_id,
                    manipulating_object_id=manipulating_object_id,
                    keyframe_position=frame_sample_label,
                    completion_gate_context=completion_gate_context,
                    previous_image_path=previous_image_path,
                    client=client,
                    model=model,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    json_mode=json_mode,
                    include_thoughts=include_thoughts,
                    save_thoughts=save_thoughts,
                    thinking_level=thinking_level,
                    thinking_budget=thinking_budget,
                    max_retries=max_retries,
                    retry_initial_delay=retry_initial_delay,
                    retry_max_delay=retry_max_delay,
                    max_response_retries=max_response_retries,
                )
            except ModelResponseFormatError as exc:
                if not continue_on_model_error:
                    raise

                record = base.build_error_record(
                    skill_idx=skill_idx,
                    stage_idx=stage_idx,
                    image_dir_name=image_dir_name,
                    image_path_text=image_path_text,
                    original_image_idx=original_image_idx,
                    frame_number=frame_number,
                    frame_duration=frame_duration,
                    keyframe_position=frame_sample_label,
                    frame_stride=frame_stride,
                    skill=skill,
                    request_input=request_input,
                    exc=exc,
                )
                records.append(record)
                previous_request_image_path = image_path_text
                print(
                    "[sample-error] "
                    + json.dumps(
                        {
                            "skill_idx": skill_idx,
                            "stage_dir": record["stage_dir"],
                            "image_path": image_path_text,
                            "frame_number": frame_number,
                            "frame_sample_label": frame_sample_label,
                            "error": record["error"],
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                if request_delay > 0:
                    print(f"[delay] wait={request_delay:.1f}s", flush=True)
                    time.sleep(request_delay)
                continue

            move_to_gate_retry_metadata: Optional[JsonObject] = None
            move_to_gate_violation = base.get_move_to_gate_violation(
                raw_result,
                is_move_to=is_move_to,
                is_last_sampled_request_for_skill=is_last_sampled_request_for_skill,
            )
            if move_to_gate_violation is not None:
                original_raw_result = raw_result
                retry_completion_gate_context = base.build_move_to_gate_retry_context(
                    violation=move_to_gate_violation,
                    is_last_sampled_request_for_skill=is_last_sampled_request_for_skill,
                )
                print(
                    "[move-to-retry] "
                    + json.dumps(
                        {
                            "skill_idx": skill_idx,
                            "stage_dir": image_dir_name,
                            "frame_number": frame_number,
                            "frame_sample_label": frame_sample_label,
                            "violation": move_to_gate_violation,
                            "original_current_skill_status": original_raw_result.get(
                                "current_skill_status"
                            ),
                            "original_is_subtask_completed": original_raw_result.get(
                                "is_subtask_completed"
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                try:
                    retry_result, retry_response_metadata = run_robot_planner(
                        image_path=image_path_text,
                        main_task=task_name,
                        old_memory=request_old_memory,
                        skill_description=skill_description,
                        object_id=object_id,
                        manipulating_object_id=manipulating_object_id,
                        keyframe_position=frame_sample_label,
                        completion_gate_context=retry_completion_gate_context,
                        previous_image_path=previous_image_path,
                        client=client,
                        model=model,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        json_mode=json_mode,
                        include_thoughts=include_thoughts,
                        save_thoughts=save_thoughts,
                        thinking_level=thinking_level,
                        thinking_budget=thinking_budget,
                        max_retries=max_retries,
                        retry_initial_delay=retry_initial_delay,
                        retry_max_delay=retry_max_delay,
                        max_response_retries=max_response_retries,
                    )
                    retry_violation = base.get_move_to_gate_violation(
                        retry_result,
                        is_move_to=is_move_to,
                        is_last_sampled_request_for_skill=is_last_sampled_request_for_skill,
                    )
                    raw_result = retry_result
                    response_metadata = retry_response_metadata
                    move_to_gate_retry_metadata = {
                        "type": "move_to_gate_retry",
                        "violation": move_to_gate_violation,
                        "retry_violation": retry_violation,
                        "retry_resolved": retry_violation is None,
                        "original_current_skill_status": original_raw_result.get(
                            "current_skill_status"
                        ),
                        "retry_current_skill_status": retry_result.get("current_skill_status"),
                    }
                    if retry_violation is not None:
                        move_to_gate_retry_metadata["fallback_required"] = True
                except ModelResponseFormatError as exc:
                    if not continue_on_model_error:
                        raise
                    move_to_gate_retry_metadata = {
                        "type": "move_to_gate_retry",
                        "violation": move_to_gate_violation,
                        "retry_resolved": False,
                        "fallback_required": True,
                        "retry_error": {
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                            "raw_response": exc.content,
                        },
                    }
                    raw_result = original_raw_result

            if base.is_no_for_sure_status(raw_result.get("current_skill_status")):
                record = {
                    "skill_idx": skill_idx,
                    "stage_idx": stage_idx,
                    "stage_dir": image_dir_name,
                    "image_dir": image_dir_name,
                    "image_path": image_path_text,
                    "image_index_in_stage": original_image_idx - 1,
                    "frame_number": frame_number,
                    "frame_duration": frame_duration,
                    "valid_duration": valid_duration,
                    "frame_selection": frame_selection,
                    "frame_stride": frame_stride,
                    "keyframe_position": frame_sample_label,
                    "frame_sample_label": frame_sample_label,
                    "frame_sample_index": sampled_idx,
                    "frame_sample_count": len(sampled_frames),
                    "skill_description": skill.get("skill_description", []),
                    "subtask_description": skill.get("subtask_description", ""),
                    "object_id": skill.get("object_id", []),
                    "manipulating_object_id": skill.get("manipulating_object_id", []),
                    "request_input": request_input,
                    "model_response": raw_result,
                    "result_used": False,
                    "discard_reason": NO_FOR_SURE_STATUS,
                }
                if response_metadata:
                    record["google_response_metadata"] = response_metadata
                if move_to_gate_retry_metadata:
                    record["move_to_gate_retry"] = move_to_gate_retry_metadata
                    record["raw_model_response_before_move_to_gate_retry"] = original_raw_result
                records.append(record)
                previous_request_image_path = image_path_text

                print(
                    "[discard] "
                    + json.dumps(
                        {
                            "skill_idx": skill_idx,
                            "stage_dir": record["stage_dir"],
                            "image_path": str(image_path),
                            "image_index_in_stage": record["image_index_in_stage"],
                            "frame_number": record["frame_number"],
                            "frame_sample_label": record["frame_sample_label"],
                            "current_skill_status": raw_result.get("current_skill_status"),
                            "result_used": False,
                            "reason": NO_FOR_SURE_STATUS,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

                if request_delay > 0:
                    print(f"[delay] wait={request_delay:.1f}s", flush=True)
                    time.sleep(request_delay)
                continue

            model_result = raw_result
            move_to_guard_metadata: Optional[JsonObject] = None
            if (
                base.is_move_to_skill(
                    skill_description=skill_description,
                    object_id=object_id,
                    manipulating_object_id=manipulating_object_id,
                )
                and base.is_completed_status(model_result.get("current_skill_status"))
                and not is_last_sampled_request_for_skill
            ):
                model_result, move_to_guard_metadata = base.suppress_premature_move_to_completion(
                    model_result,
                    previous_memory_text=request_old_memory,
                )
            elif (
                is_move_to
                and is_last_sampled_request_for_skill
                and not base.is_completed_status(model_result.get("current_skill_status"))
            ):
                model_result, move_to_guard_metadata = base.force_final_move_to_completion(
                    model_result,
                    previous_memory_text=request_old_memory,
                )

            model_result, smoothing_metadata = base.preserve_completed_state(
                model_result,
                already_completed=(
                    stage_idx in completed_stage_indices
                    and base.should_preserve_completed_state_for_skill(
                        skill_description=skill_description,
                        object_id=object_id,
                        manipulating_object_id=manipulating_object_id,
                    )
                ),
            )
            if base.is_completed_status(model_result.get("current_skill_status")):
                completed_stage_indices.add(stage_idx)

            record = {
                "skill_idx": skill_idx,
                "stage_idx": stage_idx,
                "stage_dir": image_dir_name,
                "image_dir": image_dir_name,
                "image_path": image_path_text,
                "image_index_in_stage": original_image_idx - 1,
                "frame_number": frame_number,
                "frame_duration": frame_duration,
                "valid_duration": valid_duration,
                "frame_selection": frame_selection,
                "frame_stride": frame_stride,
                "keyframe_position": frame_sample_label,
                "frame_sample_label": frame_sample_label,
                "frame_sample_index": sampled_idx,
                "frame_sample_count": len(sampled_frames),
                "skill_description": skill.get("skill_description", []),
                "subtask_description": skill.get("subtask_description", ""),
                "object_id": skill.get("object_id", []),
                "manipulating_object_id": skill.get("manipulating_object_id", []),
                "request_input": request_input,
                "model_response": model_result,
                "result_used": True,
            }
            if response_metadata:
                record["google_response_metadata"] = response_metadata
            if move_to_gate_retry_metadata:
                record["move_to_gate_retry"] = move_to_gate_retry_metadata
                record["raw_model_response_before_move_to_gate_retry"] = original_raw_result
            if move_to_guard_metadata:
                record["move_to_completion_guard"] = move_to_guard_metadata
                record["raw_model_response_before_move_to_guard"] = raw_result
            if smoothing_metadata:
                record["state_smoothing"] = smoothing_metadata
                record["raw_model_response_before_state_smoothing"] = raw_result
            records.append(record)

            print(
                "[result] "
                + json.dumps(
                    {
                        "skill_idx": skill_idx,
                        "stage_dir": record["stage_dir"],
                        "image_path": str(image_path),
                        "image_index_in_stage": record["image_index_in_stage"],
                        "frame_number": record["frame_number"],
                        "frame_sample_label": record["frame_sample_label"],
                        "subtask": model_result.get("subtask"),
                        "current_skill_status": model_result.get("current_skill_status"),
                        "is_subtask_completed": model_result.get("is_subtask_completed"),
                        "result_used": True,
                        "move_to_guarded": bool(move_to_guard_metadata),
                        "state_smoothed": bool(smoothing_metadata),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

            new_memory = model_result.get("new_memory")
            if new_memory:
                old_memory = base.memory_to_text(new_memory)
            previous_request_image_path = image_path_text
            if request_delay > 0:
                print(f"[delay] wait={request_delay:.1f}s", flush=True)
                time.sleep(request_delay)
    except KeyboardInterrupt:
        output = base.build_episode_output(
            annotation_json=annotation_json,
            image_root=image_root,
            task_name=task_name,
            records=records,
            frame_selection=frame_selection,
            interrupted=True,
        )
        if base.save_episode_output(output_path, output):
            print("[interrupted] Ctrl+C received; partial results saved.", file=sys.stderr)
        else:
            print("[interrupted] Ctrl+C received; no output path was provided.", file=sys.stderr)
        raise
    except Exception:
        output = base.build_episode_output(
            annotation_json=annotation_json,
            image_root=image_root,
            task_name=task_name,
            records=records,
            frame_selection=frame_selection,
            interrupted=True,
        )
        if base.save_episode_output(output_path, output):
            print("[error] Partial results saved before re-raising the error.", file=sys.stderr)
        raise

    output = base.build_episode_output(
        annotation_json=annotation_json,
        image_root=image_root,
        task_name=task_name,
        records=records,
        frame_selection=frame_selection,
    )
    base.save_episode_output(output_path, output)
    return output


def run_annotation_directory_planner(
    annotation_path: Path,
    image_root: Path,
    *,
    output_path: Optional[Path] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    max_output_tokens: Optional[int] = 4096,
    json_mode: bool = True,
    include_thoughts: bool = False,
    save_thoughts: bool = False,
    thinking_level: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    request_delay: float = 0.0,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_initial_delay: float = DEFAULT_RETRY_INITIAL_DELAY,
    retry_max_delay: float = DEFAULT_RETRY_MAX_DELAY,
    max_response_retries: int = DEFAULT_MAX_RESPONSE_RETRIES,
    continue_on_model_error: bool = True,
    resume: bool = False,
    episode_workers: int = 1,
    include_previous_image: bool = False,
) -> JsonObject:
    annotation_jsons = base.find_annotation_jsons(annotation_path)
    multiple_episodes = len(annotation_jsons) > 1 or annotation_path.is_dir()
    if episode_workers < 1:
        raise ValueError("--episode-workers must be >= 1")

    if (
        resume
        and annotation_path.is_dir()
        and output_path
        and output_path.suffix.lower() == ".json"
        and re.fullmatch(r"episode_.*\.json", output_path.name)
    ):
        raise ValueError(
            "In annotation directory mode, --resume cannot use one episode output file as -o: "
            f"{output_path}. Pass the output directory instead, for example: {output_path.parent}"
        )

    aggregate_output = output_path if output_path and output_path.suffix.lower() == ".json" else None
    resume_aggregate_outputs = (
        base.load_resume_aggregate_outputs(aggregate_output) if resume else {}
    )
    episode_outputs: List[Optional[JsonObject]] = [None] * len(annotation_jsons)

    def completed_episode_outputs() -> List[JsonObject]:
        return [episode for episode in episode_outputs if episode is not None]

    def run_one_episode(
        episode_number: int,
        annotation_json: Path,
        episode_image_root: Path,
        episode_output_path: Optional[Path],
    ) -> JsonObject:
        print(
            "[episode] "
            f"{episode_number}/{len(annotation_jsons)} annotation={annotation_json} "
            f"image_root={episode_image_root} memory=reset",
            flush=True,
        )
        return run_episode_planner(
            annotation_json=annotation_json,
            image_root=episode_image_root,
            output_path=episode_output_path,
            model=model,
            temperature=temperature,
            frame_stride=frame_stride,
            max_output_tokens=max_output_tokens,
            json_mode=json_mode,
            include_thoughts=include_thoughts,
            save_thoughts=save_thoughts,
            thinking_level=thinking_level,
            thinking_budget=thinking_budget,
            request_delay=request_delay,
            max_retries=max_retries,
            retry_initial_delay=retry_initial_delay,
            retry_max_delay=retry_max_delay,
            max_response_retries=max_response_retries,
            continue_on_model_error=continue_on_model_error,
            include_previous_image=include_previous_image,
        )

    if resume:
        if output_path is None:
            print(
                "[resume] --resume needs --output to find previous episode results; "
                "starting from the beginning.",
                file=sys.stderr,
                flush=True,
            )
        elif aggregate_output and resume_aggregate_outputs:
            print(
                f"[resume] loaded {len(resume_aggregate_outputs)} completed episode(s) "
                f"from {aggregate_output}",
                flush=True,
            )
        elif aggregate_output:
            print(
                f"[resume] no completed episodes found in aggregate output: {aggregate_output}",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(f"[resume] scanning episode output directory: {output_path}", flush=True)

    try:
        pending_jobs: List[Tuple[int, int, Path, Path, Optional[Path]]] = []
        for output_idx, annotation_json in enumerate(annotation_jsons):
            episode_number = output_idx + 1
            episode_image_root = base.resolve_episode_image_root(
                annotation_json,
                image_root,
                multiple_episodes=multiple_episodes,
            )
            episode_output_path = base.get_episode_output_path(output_path, annotation_json)
            if resume and episode_output_path and episode_output_path.exists():
                if base.can_resume_episode_output(episode_output_path):
                    print(
                        "[skip] "
                        f"{episode_number}/{len(annotation_jsons)} existing={episode_output_path}",
                        flush=True,
                    )
                    episode_outputs[output_idx] = base.read_json(episode_output_path)
                    continue

                print(
                    "[rerun] "
                    f"{episode_number}/{len(annotation_jsons)} incomplete_or_error={episode_output_path}",
                    flush=True,
                )
            elif resume and aggregate_output:
                existing_episode = resume_aggregate_outputs.get(annotation_json.name)
                if existing_episode:
                    print(
                        "[skip] "
                        f"{episode_number}/{len(annotation_jsons)} aggregate={aggregate_output} "
                        f"annotation={annotation_json.name}",
                        flush=True,
                    )
                    episode_outputs[output_idx] = existing_episode
                    continue

            pending_jobs.append(
                (output_idx, episode_number, annotation_json, episode_image_root, episode_output_path)
            )

        if episode_workers > 1 and pending_jobs:
            workers = min(episode_workers, len(pending_jobs))
            print(
                f"[concurrency] episode_workers={workers} pending_episode_count={len(pending_jobs)}",
                flush=True,
            )
            executor = ThreadPoolExecutor(max_workers=workers)
            futures: Dict[Future, int] = {}
            try:
                for output_idx, episode_number, annotation_json, episode_image_root, episode_output_path in pending_jobs:
                    future = executor.submit(
                        run_one_episode,
                        episode_number,
                        annotation_json,
                        episode_image_root,
                        episode_output_path,
                    )
                    futures[future] = output_idx

                for future in as_completed(futures):
                    output_idx = futures[future]
                    episode_outputs[output_idx] = future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)
        else:
            for output_idx, episode_number, annotation_json, episode_image_root, episode_output_path in pending_jobs:
                episode_outputs[output_idx] = run_one_episode(
                    episode_number,
                    annotation_json,
                    episode_image_root,
                    episode_output_path,
                )
    except KeyboardInterrupt:
        output = base.build_annotation_directory_output(
            annotation_path=annotation_path,
            image_root=image_root,
            episodes=completed_episode_outputs(),
            interrupted=True,
        )
        if aggregate_output:
            base.write_json(aggregate_output, output)
            print(
                f"[saved] {aggregate_output} ({output['episode_count']} completed episode(s))",
                flush=True,
            )
        raise
    except Exception:
        output = base.build_annotation_directory_output(
            annotation_path=annotation_path,
            image_root=image_root,
            episodes=completed_episode_outputs(),
            interrupted=True,
        )
        if aggregate_output:
            base.write_json(aggregate_output, output)
            print(
                f"[saved] {aggregate_output} ({output['episode_count']} completed episode(s))",
                flush=True,
            )
        raise

    output = base.build_annotation_directory_output(
        annotation_path=annotation_path,
        image_root=image_root,
        episodes=completed_episode_outputs(),
    )
    if aggregate_output:
        base.write_json(aggregate_output, output)
        print(
            f"[saved] {aggregate_output} ({output['episode_count']} episode(s), "
            f"{output['processed_count']} result(s))",
            flush=True,
        )

    return output


def iter_records_from_output(data: Any) -> List[JsonObject]:
    if not isinstance(data, dict):
        return []

    if isinstance(data.get("model_response"), dict) or isinstance(data.get("new_memory"), dict):
        return [data]

    records = data.get("results")
    if isinstance(records, list):
        return [record for record in records if isinstance(record, dict)]

    episodes = data.get("episodes")
    if isinstance(episodes, list):
        flattened: List[JsonObject] = []
        for episode in episodes:
            flattened.extend(iter_records_from_output(episode))
        return flattened

    return []


def select_previous_record(data: Any, record_index: Optional[int] = None) -> JsonObject:
    records = iter_records_from_output(data)
    if not records:
        raise ValueError(
            "Previous record JSON must be a single record, an episode output with results, "
            "or an aggregate output with episodes."
        )

    if record_index is not None:
        try:
            return records[record_index]
        except IndexError as exc:
            raise ValueError(
                f"Previous record index {record_index} is out of range for {len(records)} records."
            ) from exc

    for record in reversed(records):
        if record.get("result_used") is False:
            continue
        model_response = record.get("model_response")
        if isinstance(model_response, dict) and model_response.get("new_memory"):
            return record
        if record.get("new_memory"):
            return record

    raise ValueError("No usable previous record with model_response.new_memory was found.")


def load_previous_record(path: Path, record_index: Optional[int] = None) -> JsonObject:
    return select_previous_record(read_json(path), record_index=record_index)


def get_previous_record_memory(record: JsonObject) -> str:
    if record.get("result_used") is False:
        raise ValueError(
            "The selected previous record has result_used=false. It should not be used as memory."
        )

    model_response = record.get("model_response")
    if not isinstance(model_response, dict):
        model_response = record

    if is_no_for_sure_status(model_response.get("current_skill_status")):
        raise ValueError(
            "The selected previous record is no_for_sure. It should not be used as memory."
        )

    new_memory = model_response.get("new_memory")
    if not new_memory:
        raise ValueError("The selected previous record does not contain model_response.new_memory.")
    return memory_to_text(new_memory)


def get_previous_record_image_path(record: Optional[JsonObject]) -> str:
    if not isinstance(record, dict):
        return ""

    image_path = record.get("image_path")
    if image_path:
        return str(image_path)

    request_input = record.get("request_input")
    if isinstance(request_input, dict):
        image_path = request_input.get("image_path")
        if image_path:
            return str(image_path)

    return ""


def get_previous_request_value(record: JsonObject, key: str) -> str:
    request_input = record.get("request_input")
    if not isinstance(request_input, dict):
        return ""
    value = request_input.get(key)
    if value is None:
        return ""
    return str(value)


def get_case_value(case: Optional[JsonObject], key: str) -> str:
    if not isinstance(case, dict):
        return ""
    value = case.get(key)
    if value is None:
        request_input = case.get("request_input")
        if isinstance(request_input, dict):
            value = request_input.get(key)
    return format_annotation_value(value)


def get_case_raw_value(case: Optional[JsonObject], key: str) -> Any:
    if not isinstance(case, dict):
        return None
    if key in case:
        return case.get(key)
    request_input = case.get("request_input")
    if isinstance(request_input, dict):
        return request_input.get(key)
    return None


def parse_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"Expected boolean value, got: {value!r}")


def get_case_path_value(case: Optional[JsonObject], *keys: str) -> str:
    if not isinstance(case, dict):
        return ""
    for key in keys:
        value = case.get(key)
        if value:
            return str(value)
    request_input = case.get("request_input")
    if isinstance(request_input, dict):
        for key in keys:
            value = request_input.get(key)
            if value:
                return str(value)
    return ""


def get_case_previous_record(case: Optional[JsonObject]) -> Optional[JsonObject]:
    if not isinstance(case, dict):
        return None
    previous_record = case.get("previous_record")
    if previous_record is None:
        previous_record = case.get("previous_result")
    if previous_record is None:
        return None
    if not isinstance(previous_record, dict):
        raise ValueError("single-step JSON previous_record must be a JSON object.")
    return previous_record


def build_single_image_record(
    *,
    image_path: str,
    previous_image_path: str = "",
    request_input: JsonObject,
    model_response: JsonObject,
    metadata: JsonObject,
    previous_record_json: Optional[Path] = None,
    previous_record_index: Optional[int] = None,
    previous_record: Optional[JsonObject] = None,
) -> JsonObject:
    result_used = not is_no_for_sure_status(model_response.get("current_skill_status"))
    record: JsonObject = {
        "image_path": image_path,
        "request_input": request_input,
        "model_response": model_response,
        "result_used": result_used,
    }
    if previous_image_path:
        record["previous_image_path"] = previous_image_path
    if not result_used:
        record["discard_reason"] = NO_FOR_SURE_STATUS
    if metadata:
        record["google_response_metadata"] = metadata
    if previous_record_json is not None:
        record["previous_record_json"] = str(previous_record_json)
        record["previous_record_index"] = previous_record_index
    if previous_record is not None:
        record["previous_model_response"] = previous_record.get("model_response", previous_record)
    return record


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run robot planner requests through the official Google Gemini API."
    )
    parser.add_argument("--image-path", help="Image path for one request.")
    parser.add_argument(
        "--previous-image-path",
        default="",
        help=(
            "Optional previous observation image for single-image mode. When provided, "
            "the request sends both previous and current images for temporal comparison."
        ),
    )
    parser.add_argument("--main-task", help="Main task for one request.")
    parser.add_argument("--old-memory", default="", help="Previous memory.")
    parser.add_argument(
        "--previous-record-json",
        help=(
            "Single-step test helper. Load old_memory from a previous record JSON, "
            "episode output JSON, or aggregate output JSON."
        ),
    )
    parser.add_argument(
        "--single-step-json",
        help=(
            "Single-step test case JSON containing previous_record and next_image_path. "
            "Optional fields: main_task, skill_description, object_id, manipulating_object_id."
        ),
    )
    parser.add_argument(
        "--previous-record-index",
        type=int,
        help=(
            "Record index to use from --previous-record-json. Supports negative indexes. "
            "When omitted, the last usable record with model_response.new_memory is used."
        ),
    )
    parser.add_argument(
        "--include-previous-image",
        action="store_true",
        help=(
            "Episode mode: send the previous sampled request image together with the current "
            "image. Single-image mode: derive the previous image from --previous-record-json "
            "when --previous-image-path is not provided."
        ),
    )
    parser.add_argument("--skill-description", default="", help="Skill annotation.")
    parser.add_argument(
        "--keyframe-position",
        default="",
        help="Deprecated and ignored. Frame position no longer changes the system prompt.",
    )
    parser.add_argument(
        "--completion-gate-context",
        default="",
        help=(
            "Optional single-image testing gate text. Normally produced automatically in "
            "episode mode for move-to skills."
        ),
    )
    parser.add_argument("--object-id", default="", help="Object annotation.")
    parser.add_argument(
        "--manipulating-object-id",
        default="",
        help="Manipulating object annotation.",
    )
    parser.add_argument(
        "--annotation-json",
        help="Episode annotation JSON file, or a directory containing episode_*.json files.",
    )
    parser.add_argument(
        "--image-root",
        help=(
            "Episode image root containing stage_00/stage_01 or skill_00/skill_01 directories. "
            "For annotation directory mode, use a root containing episode_name/stage_00 "
            "or episode_name/skill_00."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output JSON path for single episode or aggregate directory mode. "
            "In annotation directory mode, a non-.json path is treated as an output directory."
        ),
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=DEFAULT_FRAME_STRIDE,
        help=(
            "Sample one frame every N frames across episode valid_duration, "
            "starting at the first valid frame."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument(
        "--disable-json-mode",
        action="store_true",
        help="Do not set response_mime_type=application/json.",
    )
    parser.add_argument(
        "--include-thoughts",
        action="store_true",
        help="Request Gemini thought summaries with thinking_config.include_thoughts=true.",
    )
    parser.add_argument(
        "--save-thoughts",
        action="store_true",
        help="Save returned thought summaries when --include-thoughts is used.",
    )
    parser.add_argument(
        "--thinking-level",
        choices=["minimal", "low", "medium", "high"],
        help="Gemini 3 thinking level. Leave unset to use the model default.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        help="Gemini 2.5 thinking token budget. Leave unset for model default.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="Seconds to wait after each successful API request in episode modes.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retry attempts for retryable API errors such as 429 rate limits.",
    )
    parser.add_argument(
        "--retry-initial-delay",
        type=float,
        default=DEFAULT_RETRY_INITIAL_DELAY,
        help="Initial retry delay in seconds for retryable API errors.",
    )
    parser.add_argument(
        "--retry-max-delay",
        type=float,
        default=DEFAULT_RETRY_MAX_DELAY,
        help="Maximum retry delay in seconds for retryable API errors.",
    )
    parser.add_argument(
        "--max-response-retries",
        type=int,
        default=DEFAULT_MAX_RESPONSE_RETRIES,
        help="Maximum retry attempts when Gemini returns empty, invalid, or incomplete JSON.",
    )
    parser.add_argument(
        "--fail-on-model-error",
        action="store_true",
        help="Stop the batch if response-format retries still fail for a sample.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "In annotation directory mode, skip complete and error-free episodes from an "
            "existing output directory or aggregate output JSON."
        ),
    )
    parser.add_argument(
        "--episode-workers",
        type=int,
        default=1,
        help=(
            "Number of episodes to process concurrently in annotation directory mode. "
            "Each episode still processes its own selected frames sequentially."
        ),
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        if args.annotation_json or args.image_root:
            if not args.annotation_json or not args.image_root:
                parser.error("--annotation-json and --image-root must be used together.")

            annotation_path = Path(args.annotation_json)
            image_root = Path(args.image_root)
            output_path = Path(args.output) if args.output else None
            planner_kwargs = {
                "model": args.model,
                "temperature": args.temperature,
                "frame_stride": args.frame_stride,
                "max_output_tokens": args.max_output_tokens,
                "json_mode": not args.disable_json_mode,
                "include_thoughts": args.include_thoughts,
                "save_thoughts": args.save_thoughts,
                "thinking_level": args.thinking_level,
                "thinking_budget": args.thinking_budget,
                "request_delay": args.request_delay,
                "max_retries": args.max_retries,
                "retry_initial_delay": args.retry_initial_delay,
                "retry_max_delay": args.retry_max_delay,
                "max_response_retries": args.max_response_retries,
                "continue_on_model_error": not args.fail_on_model_error,
                "include_previous_image": args.include_previous_image,
            }
            if annotation_path.is_dir():
                result = run_annotation_directory_planner(
                    annotation_path=annotation_path,
                    image_root=image_root,
                    output_path=output_path,
                    resume=args.resume,
                    episode_workers=args.episode_workers,
                    **planner_kwargs,
                )
            else:
                result = run_episode_planner(
                    annotation_json=annotation_path,
                    image_root=image_root,
                    output_path=output_path,
                    **planner_kwargs,
                )
        else:
            single_step_case: Optional[JsonObject] = None
            if args.single_step_json:
                loaded_case = read_json(Path(args.single_step_json))
                if not isinstance(loaded_case, dict):
                    raise ValueError("--single-step-json must contain a JSON object.")
                single_step_case = loaded_case

            previous_record_path_text = args.previous_record_json or get_case_path_value(
                single_step_case,
                "previous_record_json",
            )
            previous_record_path = Path(previous_record_path_text) if previous_record_path_text else None
            previous_record: Optional[JsonObject] = None
            previous_memory = ""
            embedded_previous_record = get_case_previous_record(single_step_case)
            if embedded_previous_record is not None:
                previous_record = select_previous_record(
                    embedded_previous_record,
                    record_index=args.previous_record_index,
                )
            elif previous_record_path is not None:
                previous_record = load_previous_record(
                    previous_record_path,
                    record_index=args.previous_record_index,
                )
            if previous_record is not None:
                previous_memory = get_previous_record_memory(previous_record)

            image_path = args.image_path or get_case_path_value(
                single_step_case,
                "next_image_path",
                "image_path",
            )
            previous_image_path = args.previous_image_path or get_case_path_value(
                single_step_case,
                "previous_image_path",
                "previous_request_image_path",
            )
            if not previous_image_path and args.include_previous_image:
                previous_image_path = get_previous_record_image_path(previous_record)
            main_task = args.main_task or get_case_value(single_step_case, "main_task") or (
                get_previous_request_value(previous_record, "main_task")
                if previous_record is not None
                else ""
            )
            old_memory = args.old_memory or get_case_value(single_step_case, "old_memory") or previous_memory
            skill_description = args.skill_description or get_case_value(
                single_step_case,
                "skill_description",
            ) or (
                get_previous_request_value(previous_record, "skill_description")
                if previous_record is not None
                else ""
            )
            object_id = args.object_id or get_case_value(single_step_case, "object_id") or (
                get_previous_request_value(previous_record, "object_id")
                if previous_record is not None
                else ""
            )
            manipulating_object_id = args.manipulating_object_id or get_case_value(
                single_step_case,
                "manipulating_object_id",
            ) or (
                get_previous_request_value(previous_record, "manipulating_object_id")
                if previous_record is not None
                else ""
            )
            is_move_to = is_move_to_skill(
                skill_description=skill_description,
                object_id=object_id,
                manipulating_object_id=manipulating_object_id,
            )
            case_is_last_move_to_request = parse_optional_bool(
                get_case_raw_value(single_step_case, "is_last_move_to_request")
            )
            completion_gate_context = (
                args.completion_gate_context
                or get_case_value(single_step_case, "completion_gate_context")
            )
            if not completion_gate_context and is_move_to and case_is_last_move_to_request is not None:
                completion_gate_context = build_move_to_completion_gate_context(
                    is_move_to=True,
                    is_last_sampled_request_for_skill=case_is_last_move_to_request,
                )

            if not image_path or not main_task:
                parser.error(
                    "--image-path and --main-task are required in single-image mode, "
                    "unless --single-step-json or --previous-record-json provides them."
                )

            request_input = base.build_request_input(
                image_path=image_path,
                main_task=main_task,
                old_memory=old_memory,
                skill_description=skill_description,
                object_id=object_id,
                manipulating_object_id=manipulating_object_id,
                completion_gate_context=completion_gate_context,
                previous_image_path=previous_image_path,
            )

            result, metadata = run_robot_planner(
                image_path=image_path,
                main_task=main_task,
                old_memory=old_memory,
                skill_description=skill_description,
                object_id=object_id,
                manipulating_object_id=manipulating_object_id,
                keyframe_position=args.keyframe_position,
                completion_gate_context=completion_gate_context,
                previous_image_path=previous_image_path,
                model=args.model,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                json_mode=not args.disable_json_mode,
                include_thoughts=args.include_thoughts,
                save_thoughts=args.save_thoughts,
                thinking_level=args.thinking_level,
                thinking_budget=args.thinking_budget,
                max_retries=args.max_retries,
                retry_initial_delay=args.retry_initial_delay,
                retry_max_delay=args.retry_max_delay,
                max_response_retries=args.max_response_retries,
            )
            result = build_single_image_record(
                image_path=image_path,
                previous_image_path=previous_image_path,
                request_input=request_input,
                model_response=result,
                metadata=metadata,
                previous_record_json=previous_record_path,
                previous_record_index=args.previous_record_index,
                previous_record=previous_record,
            )
            if args.output:
                write_json(Path(args.output), result)
                print(f"[saved] {args.output}", flush=True)
    except KeyboardInterrupt:
        return 130

    if not (args.annotation_json or args.image_root) and not args.output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
