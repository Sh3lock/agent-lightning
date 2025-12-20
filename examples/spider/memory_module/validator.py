from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import sqlglot
    from sqlglot import exp
except Exception:  # pragma: no cover
    sqlglot = None  # type: ignore[assignment]
    exp = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    error_type: str
    message: str
    entities: Dict[str, List[str]]


class StaticValidator:
    def __init__(self, default_dialect: str = "sqlite") -> None:
        self.default_dialect = default_dialect

    def validate(self, sql: str, table_info: str, dialect: Optional[str] = None) -> ValidationResult:
        if sqlglot is None or exp is None:
            return ValidationResult(
                ok=True,
                error_type="Unavailable",
                message="sqlglot unavailable",
                entities={},
            )

        tables, columns_by_table = _parse_table_info(table_info)
        if not tables:
            return ValidationResult(ok=True, error_type="SchemaUnavailable", message="schema unavailable", entities={})

        read_dialect = dialect or self.default_dialect
        try:
            tree = sqlglot.parse_one(sql, read=read_dialect)
        except Exception as exc:
            return ValidationResult(
                ok=False,
                error_type="SyntaxError",
                message=f"Failed to parse SQL: {exc}",
                entities={},
            )

        alias_map: Dict[str, str] = {}
        referenced_tables: Set[str] = set()
        for table in tree.find_all(exp.Table):
            name = table.name
            if name:
                referenced_tables.add(name)
            alias = table.args.get("alias")
            if alias and alias.this and name:
                alias_map[alias.this.name] = name

        for table in referenced_tables:
            if table not in tables:
                return ValidationResult(
                    ok=False,
                    error_type="MissingTable",
                    message=f"Table not in schema: {table}",
                    entities={"tables": [table]},
                )

        for col in tree.find_all(exp.Column):
            col_name = col.name
            if not col_name or col_name == "*":
                continue
            table_ref = col.table
            if table_ref:
                table_name = alias_map.get(table_ref, table_ref)
                if table_name not in tables:
                    return ValidationResult(
                        ok=False,
                        error_type="MissingTable",
                        message=f"Table not in schema: {table_name}",
                        entities={"tables": [table_name]},
                    )
                if col_name not in columns_by_table.get(table_name, set()):
                    return ValidationResult(
                        ok=False,
                        error_type="MissingColumn",
                        message=f"Column not in schema: {table_name}.{col_name}",
                        entities={"tables": [table_name], "columns": [col_name]},
                    )
                continue

            candidate_tables = [
                table for table in referenced_tables if col_name in columns_by_table.get(table, set())
            ]
            if not candidate_tables:
                return ValidationResult(
                    ok=False,
                    error_type="MissingColumn",
                    message=f"Column not in schema: {col_name}",
                    entities={"columns": [col_name]},
                )
            if len(candidate_tables) > 1 and len(referenced_tables) > 1:
                return ValidationResult(
                    ok=False,
                    error_type="AmbiguousColumn",
                    message=f"Ambiguous column reference: {col_name}",
                    entities={"columns": [col_name], "tables": candidate_tables},
                )

        return ValidationResult(ok=True, error_type="OK", message="", entities={})


def _parse_table_info(table_info: str) -> Tuple[Set[str], Dict[str, Set[str]]]:
    tables: Set[str] = set()
    columns_by_table: Dict[str, Set[str]] = {}

    create_table_pattern = re.compile(
        r"CREATE\\s+TABLE\\s+[`\\\"]?(\\w+)[`\\\"]?\\s*\\((.*?)\\)\\s*;?",
        re.IGNORECASE | re.DOTALL,
    )
    for match in create_table_pattern.finditer(table_info):
        table = _strip_identifier(match.group(1))
        tables.add(table)
        columns_by_table.setdefault(table, set())
        columns_blob = match.group(2)
        for raw in columns_blob.split(","):
            token = raw.strip()
            if not token:
                continue
            first = _strip_identifier(token.split()[0])
            if first.upper() in {"PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT"}:
                continue
            columns_by_table[table].add(first)

    current_table = None
    for line in table_info.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("table:"):
            current_table = _strip_identifier(stripped.split(":", 1)[1].strip())
            tables.add(current_table)
            columns_by_table.setdefault(current_table, set())
            continue
        if "columns:" in lower:
            parts = stripped.split(":", 1)
            if len(parts) != 2:
                continue
            cols = [
                _strip_identifier(col.strip())
                for col in parts[1].split(",")
                if col.strip()
            ]
            if current_table:
                columns_by_table.setdefault(current_table, set()).update(cols)

    return tables, columns_by_table


def _strip_identifier(value: str) -> str:
    return value.strip().strip("`\"[]")
