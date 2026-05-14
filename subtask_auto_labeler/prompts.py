from pathlib import Path
from typing import Any, Dict

from .io_utils import JsonObject, read_json, require_json_object


class PromptCatalog:
    def __init__(self, data: JsonObject):
        self._data = data

    @classmethod
    def from_file(cls, path: Path) -> "PromptCatalog":
        return cls(require_json_object(read_json(path), str(path)))

    def get(self, name: str) -> str:
        value = self._data.get(name)
        if not isinstance(value, str):
            raise KeyError(f"Prompt config is missing string prompt: {name}")
        return value

    def render(self, name: str, values: Dict[str, Any]) -> str:
        text = self.get(name)
        rendered = text
        for key, value in values.items():
            rendered = rendered.replace("{" + key + "}", value if isinstance(value, str) else str(value))
        return rendered
