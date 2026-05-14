import json
from pathlib import Path
from typing import Any, Dict

JsonObject = Dict[str, Any]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def require_json_object(value: Any, source: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a JSON object.")
    return value


def format_annotation_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)
