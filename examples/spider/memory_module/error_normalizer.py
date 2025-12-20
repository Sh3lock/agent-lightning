from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class NormalizedError:
    error_type: str
    entities: Dict[str, List[str]]
    raw: str


_PATTERNS = [
    ("MissingColumn", re.compile(r"no such column: ([\\w\\.]+)", re.IGNORECASE)),
    ("MissingTable", re.compile(r"no such table: ([\\w\\.]+)", re.IGNORECASE)),
    ("AmbiguousColumn", re.compile(r"ambiguous column name: ([\\w\\.]+)", re.IGNORECASE)),
    ("SyntaxError", re.compile(r'near "([^"]+)": syntax error', re.IGNORECASE)),
    ("NoSuchFunction", re.compile(r"no such function: ([\\w\\.]+)", re.IGNORECASE)),
    ("GroupByError", re.compile(r"(not in group by|misuse of aggregate|group by)", re.IGNORECASE)),
    ("TypeMismatch", re.compile(r"(datatype mismatch|type mismatch)", re.IGNORECASE)),
]


def normalize_error(raw: str, dialect: str = "sqlite") -> NormalizedError:
    text = raw or ""
    for error_type, pattern in _PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        entities: Dict[str, List[str]] = {}
        if error_type in {"MissingColumn", "AmbiguousColumn"}:
            entities["columns"] = [match.group(1)]
        elif error_type == "MissingTable":
            entities["tables"] = [match.group(1)]
        elif error_type == "SyntaxError":
            entities["tokens"] = [match.group(1)]
        elif error_type == "NoSuchFunction":
            entities["functions"] = [match.group(1)]
        else:
            entities["tokens"] = [match.group(1)]
        return NormalizedError(error_type=error_type, entities=entities, raw=text)

    return NormalizedError(error_type="Other", entities={}, raw=text)
