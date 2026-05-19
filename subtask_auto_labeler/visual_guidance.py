from typing import Any


INDIRECT_EVIDENCE_REPLACEMENTS = (
    (
        "use shadows and reflections to accurately judge physical contact and lift-off gaps",
        "rely on clear, direct visual evidence to judge physical contact and lift-off gaps",
    ),
    (
        "use shadows and reflections to judge physical contact and lift-off gaps",
        "rely on clear, direct visual evidence to judge physical contact and lift-off gaps",
    ),
    (
        "use shadows or reflections to judge physical contact and lift-off gaps",
        "rely on clear, direct visual evidence to judge physical contact and lift-off gaps",
    ),
)


def sanitize_visual_guidance_text(text: str) -> str:
    cleaned = text.strip()
    for old, new in INDIRECT_EVIDENCE_REPLACEMENTS:
        cleaned = cleaned.replace(old, new)
        cleaned = cleaned.replace(old.capitalize(), new.capitalize())
    return cleaned


def sanitize_visual_guidance(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_visual_guidance_text(value)
    if isinstance(value, list):
        return [sanitize_visual_guidance(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_visual_guidance(item) for key, item in value.items()}
    return value
