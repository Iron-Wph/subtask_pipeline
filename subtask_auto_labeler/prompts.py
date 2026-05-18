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
        return value.replace("\\n", "\n")

    def render(self, name: str, values: Dict[str, Any]) -> str:
        text = self.get(name)
        rendered = text
        for key, value in values.items():
            rendered = rendered.replace("{" + key + "}", value if isinstance(value, str) else str(value))
        return rendered

    def render_optional(self, name: str, values: Dict[str, Any]) -> str:
        value = self._data.get(name)
        if value is None:
            return ""
        if not isinstance(value, str):
            raise KeyError(f"Prompt config must define string prompt when present: {name}")
        rendered = value.replace("\\n", "\n")
        for key, item in values.items():
            rendered = rendered.replace("{" + key + "}", item if isinstance(item, str) else str(item))
        return rendered
