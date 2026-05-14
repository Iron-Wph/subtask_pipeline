import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .io_utils import JsonObject, format_annotation_value, read_json, require_json_object

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
STAGE_DIR_PREFIXES = ("stage", "skill")


@dataclass(frozen=True)
class SkillSpec:
    stage_idx: int
    skill_idx: Any
    skill: JsonObject
    skill_description: str
    object_id: str
    manuipation_object_id: str
    frame_duration: Tuple[int, int]


@dataclass(frozen=True)
class SampledFrame:
    frame_number: int
    image_path: Path
    image_index_in_stage: int


@dataclass(frozen=True)
class EpisodeData:
    annotation_json: Path
    image_root: Path
    annotation: JsonObject
    task_name: str
    skills: List[SkillSpec]


def load_episode(annotation_json: Path, image_root: Path) -> EpisodeData:
    annotation = require_json_object(read_json(annotation_json), str(annotation_json))
    task_name = get_task_name(annotation)
    skills = build_skill_specs(annotation)
    return EpisodeData(
        annotation_json=annotation_json,
        image_root=image_root,
        annotation=annotation,
        task_name=task_name,
        skills=skills,
    )


def get_task_name(annotation: JsonObject) -> str:
    for key in ("task_name", "main_task", "task_description"):
        value = annotation.get(key)
        if value:
            return format_annotation_value(value)
    meta_data = annotation.get("meta_data")
    if isinstance(meta_data, dict):
        for key in ("task_name", "main_task", "task_description"):
            value = meta_data.get(key)
            if value:
                return format_annotation_value(value)
    return ""


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


def get_frame_duration(skill: JsonObject, stage_idx: int) -> Tuple[int, int]:
    frame_duration = skill.get("frame_duration")
    if (
        not isinstance(frame_duration, list)
        or len(frame_duration) != 2
        or not all(isinstance(frame, int) for frame in frame_duration)
    ):
        raise ValueError(
            f"Skill at index {stage_idx} must contain frame_duration as [start_frame, end_frame]."
        )
    start_frame, end_frame = frame_duration
    if start_frame >= end_frame:
        raise ValueError(f"Skill at index {stage_idx} has invalid frame_duration: {frame_duration}.")
    return start_frame, end_frame


def get_valid_duration(annotation: JsonObject) -> Optional[Tuple[int, int]]:
    valid_duration = annotation.get("valid_duration")
    if valid_duration is None:
        meta_data = annotation.get("meta_data")
        if isinstance(meta_data, dict):
            valid_duration = meta_data.get("valid_duration")
    if valid_duration is None:
        return None
    if (
        not isinstance(valid_duration, list)
        or len(valid_duration) != 2
        or not all(isinstance(frame, int) for frame in valid_duration)
    ):
        raise ValueError("valid_duration must be [start_frame, end_frame] when present.")
    start_frame, end_frame = valid_duration
    if start_frame >= end_frame:
        raise ValueError(f"Invalid valid_duration: {valid_duration}.")
    return start_frame, end_frame


def get_manuipation_object_id(skill: JsonObject) -> str:
    if "manuipation_object_id" in skill:
        return format_annotation_value(skill.get("manuipation_object_id"))
    return format_annotation_value(skill.get("manipulating_object_id"))


def build_skill_specs(annotation: JsonObject) -> List[SkillSpec]:
    specs: List[SkillSpec] = []
    for stage_idx, skill in enumerate(get_skills(annotation)):
        specs.append(
            SkillSpec(
                stage_idx=stage_idx,
                skill_idx=skill.get("skill_idx", stage_idx),
                skill=skill,
                skill_description=format_annotation_value(skill.get("skill_description")),
                object_id=format_annotation_value(skill.get("object_id")),
                manuipation_object_id=get_manuipation_object_id(skill),
                frame_duration=get_frame_duration(skill, stage_idx),
            )
        )
    return specs


def parse_frame_number(image_path: Path) -> Optional[int]:
    match = re.fullmatch(r"frame_(\d+)", image_path.stem)
    if match is None:
        return None
    return int(match.group(1))


def has_stage_image_root(path: Path) -> bool:
    return any((path / f"{prefix}_00").is_dir() for prefix in STAGE_DIR_PREFIXES)


def find_annotation_jsons(annotation_path: Path) -> List[Path]:
    if annotation_path.is_file():
        return [annotation_path]
    if not annotation_path.is_dir():
        raise FileNotFoundError(f"Annotation path does not exist: {annotation_path}")
    episode_jsons = sorted(path for path in annotation_path.glob("episode_*.json") if path.is_file())
    if episode_jsons:
        return episode_jsons
    jsons = sorted(path for path in annotation_path.glob("*.json") if path.is_file())
    if not jsons:
        raise FileNotFoundError(f"No annotation JSON files found in: {annotation_path}")
    return jsons


def resolve_episode_image_root(
    annotation_json: Path,
    image_root: Path,
    *,
    multiple_episodes: bool,
) -> Path:
    if not multiple_episodes:
        return image_root
    direct = image_root / annotation_json.stem
    if has_stage_image_root(direct):
        return direct
    if has_stage_image_root(image_root):
        raise ValueError(
            "Directory annotation mode needs --image-root to contain one image folder per episode, "
            f"for example: {image_root / annotation_json.stem / 'stage_00'}"
        )
    raise FileNotFoundError(
        f"Could not locate image root for {annotation_json.name}. Expected: {direct / 'stage_00'}"
    )


def find_stage_images(image_root: Path, stage_idx: int) -> List[Path]:
    candidates = [image_root / f"{prefix}_{stage_idx:02d}" for prefix in STAGE_DIR_PREFIXES]
    stage_dir = next((path for path in candidates if path.is_dir()), None)
    if stage_dir is None:
        expected = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Missing subtask image directory. Expected one of: {expected}")
    images = sorted(
        path for path in stage_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No image files found in subtask image directory: {stage_dir}")
    return images


def build_frame_image_map(images: Iterable[Path]) -> Dict[int, Path]:
    frame_images: Dict[int, Path] = {}
    for image_path in images:
        frame_number = parse_frame_number(image_path)
        if frame_number is None:
            continue
        frame_images.setdefault(frame_number, image_path)
    return frame_images


def sample_uniform_internal_frames(
    frame_duration: Tuple[int, int],
    available_frame_numbers: Sequence[int],
    k: int,
) -> List[int]:
    if k < 1:
        raise ValueError("k must be >= 1")
    start_frame, end_frame = frame_duration
    candidates = sorted(frame for frame in available_frame_numbers if start_frame < frame < end_frame)
    if not candidates:
        raise FileNotFoundError(
            f"No available internal frames for duration {list(frame_duration)}. "
            "Sampling excludes both boundary frames."
        )
    if len(candidates) <= k:
        return candidates

    n = len(candidates)
    indices: List[int] = []
    used = set()
    for i in range(k):
        idx = round((i + 1) * (n - 1) / (k + 1))
        while idx in used and idx + 1 < n:
            idx += 1
        while idx in used and idx - 1 >= 0:
            idx -= 1
        used.add(idx)
        indices.append(idx)
    return [candidates[idx] for idx in sorted(indices)]


def sample_subtask_images(image_root: Path, skill: SkillSpec, k: int) -> List[SampledFrame]:
    images = find_stage_images(image_root, skill.stage_idx)
    frame_image_map = build_frame_image_map(images)
    if not frame_image_map:
        raise FileNotFoundError(f"No frame_000123-style images found in {images[0].parent}")
    sampled_frames = sample_uniform_internal_frames(
        skill.frame_duration,
        sorted(frame_image_map.keys()),
        k,
    )
    samples: List[SampledFrame] = []
    for frame_number in sampled_frames:
        image_path = frame_image_map[frame_number]
        samples.append(
            SampledFrame(
                frame_number=frame_number,
                image_path=image_path,
                image_index_in_stage=images.index(image_path),
            )
        )
    return samples


def iter_stride_frames(annotation: JsonObject, image_root: Path, frame_stride: int) -> List[Tuple[SkillSpec, SampledFrame]]:
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")
    valid_duration = get_valid_duration(annotation)
    skills = build_skill_specs(annotation)
    if valid_duration is None:
        first = min(spec.frame_duration[0] for spec in skills)
        last = max(spec.frame_duration[1] for spec in skills)
        valid_duration = (first, last)

    image_cache: Dict[int, Tuple[List[Path], Dict[int, Path]]] = {}
    samples: List[Tuple[SkillSpec, SampledFrame]] = []
    start_frame, end_frame = valid_duration
    for frame_number in range(start_frame, end_frame, frame_stride):
        skill = next(
            (spec for spec in skills if spec.frame_duration[0] <= frame_number < spec.frame_duration[1]),
            None,
        )
        if skill is None:
            continue
        if skill.stage_idx not in image_cache:
            images = find_stage_images(image_root, skill.stage_idx)
            image_cache[skill.stage_idx] = (images, build_frame_image_map(images))
        images, frame_image_map = image_cache[skill.stage_idx]
        image_path = frame_image_map.get(frame_number)
        if image_path is None:
            continue
        samples.append(
            (
                skill,
                SampledFrame(
                    frame_number=frame_number,
                    image_path=image_path,
                    image_index_in_stage=images.index(image_path),
                ),
            )
        )
    return samples
