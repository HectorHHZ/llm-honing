"""A small, standard-library interpreter of the JSON Schema 2020-12 keywords the Exp 0 v0 schemas use.

The release verifier must run in a fresh Python with no installed packages, so it cannot import
``jsonschema``. The schema files in ``schemas/`` are ordinary JSON Schema 2020-12 documents; this module
checks an instance against them using only the keywords listed in ``SUPPORTED``. A schema that uses any
other keyword is refused when it is loaded, so the schema files cannot silently rely on a keyword that
this interpreter would ignore.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

SUPPORTED = {
    "$schema",
    "$id",
    "$defs",
    "$ref",
    "$comment",
    "title",
    "description",
    "examples",
    "default",
    "type",
    "enum",
    "const",
    "properties",
    "required",
    "additionalProperties",
    "propertyNames",
    "items",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "anyOf",
    "oneOf",
    "allOf",
    "not",
    "if",
    "then",
    "else",
    "minProperties",
    "maxProperties",
}
_SUBSCHEMA_MAPS = ("$defs", "properties")
_SUBSCHEMA_LISTS = ("anyOf", "oneOf", "allOf")
_SUBSCHEMA_ONE = ("additionalProperties", "items", "not", "if", "then", "else", "propertyNames")


class SchemaError(ValueError):
    """The schema itself uses a keyword this interpreter does not implement, or is malformed."""


def _check_keywords(schema: Any, where: str) -> None:
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict):
        raise SchemaError(f"{where}: a schema is an object or a boolean")
    unknown = set(schema) - SUPPORTED
    if unknown:
        raise SchemaError(f"{where}: unsupported schema keywords {sorted(unknown)}")
    for key in _SUBSCHEMA_MAPS:
        for name, sub in (schema.get(key) or {}).items():
            _check_keywords(sub, f"{where}/{key}/{name}")
    for key in _SUBSCHEMA_LISTS:
        for number, sub in enumerate(schema.get(key) or ()):
            _check_keywords(sub, f"{where}/{key}/{number}")
    for key in _SUBSCHEMA_ONE:
        if key in schema:
            _check_keywords(schema[key], f"{where}/{key}")


def load_schema(path: Path) -> dict[str, Any]:
    schema = json.loads(Path(path).read_text(encoding="utf-8"))
    _check_keywords(schema, Path(path).name)
    return schema


def _type_ok(value: Any, name: str) -> bool:
    if name == "null":
        return value is None
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if name == "string":
        return isinstance(value, str)
    if name == "array":
        return isinstance(value, list)
    if name == "object":
        return isinstance(value, dict)
    raise SchemaError(f"unknown type {name!r}")


def _resolve(root: dict[str, Any], ref: str) -> Any:
    if not ref.startswith("#/"):
        raise SchemaError(f"only local references are supported, got {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return node


def errors(instance: Any, schema: Any, root: dict[str, Any] | None = None, where: str = "$") -> list[str]:
    """Every violation of ``schema`` by ``instance``, as ``path: message`` strings (empty when valid)."""
    root = schema if root is None else root
    if schema is True:
        return []
    if schema is False:
        return [f"{where}: no value is allowed here"]
    found: list[str] = []
    if "$ref" in schema:
        found += errors(instance, _resolve(root, schema["$ref"]), root, where)
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_type_ok(instance, name) for name in names):
            return found + [f"{where}: expected {'/'.join(names)}, got {type(instance).__name__}"]  # noqa: RUF005
    if "enum" in schema and not any(_same(instance, option) for option in schema["enum"]):
        found.append(f"{where}: {instance!r} is not one of {schema['enum']}")
    if "const" in schema and not _same(instance, schema["const"]):
        found.append(f"{where}: expected {schema['const']!r}")
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            found.append(f"{where}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            found.append(f"{where}: longer than {schema['maxLength']}")
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            found.append(f"{where}: {instance!r} does not match {schema['pattern']!r}")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            found.append(f"{where}: below {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            found.append(f"{where}: above {schema['maximum']}")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            found.append(f"{where}: not above {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
            found.append(f"{where}: not below {schema['exclusiveMaximum']}")
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            found.append(f"{where}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            found.append(f"{where}: more than {schema['maxItems']} items")
        if schema.get("uniqueItems"):
            seen = [json.dumps(item, sort_keys=True) for item in instance]
            if len(set(seen)) != len(seen):
                found.append(f"{where}: items are not unique")
        if "items" in schema:
            for number, item in enumerate(instance):
                found += errors(item, schema["items"], root, f"{where}[{number}]")
    if isinstance(instance, dict):
        for name in schema.get("required", ()):
            if name not in instance:
                found.append(f"{where}: missing required field {name!r}")
        if "minProperties" in schema and len(instance) < schema["minProperties"]:
            found.append(f"{where}: fewer than {schema['minProperties']} fields")
        if "maxProperties" in schema and len(instance) > schema["maxProperties"]:
            found.append(f"{where}: more than {schema['maxProperties']} fields")
        properties = schema.get("properties", {})
        for name, value in instance.items():
            if name in properties:
                found += errors(value, properties[name], root, f"{where}.{name}")
            elif "additionalProperties" in schema:
                found += errors(value, schema["additionalProperties"], root, f"{where}.{name}")
            if "propertyNames" in schema:
                found += errors(name, schema["propertyNames"], root, f"{where}<key {name!r}>")
    for sub in schema.get("allOf", ()):
        found += errors(instance, sub, root, where)
    if "anyOf" in schema and all(errors(instance, sub, root, where) for sub in schema["anyOf"]):
        found.append(f"{where}: matches none of the anyOf alternatives")
    if "oneOf" in schema:
        matches = sum(not errors(instance, sub, root, where) for sub in schema["oneOf"])
        if matches != 1:
            found.append(f"{where}: matches {matches} of the oneOf alternatives, expected exactly 1")
    if "not" in schema and not errors(instance, schema["not"], root, where):
        found.append(f"{where}: matches a schema it must not match")
    if "if" in schema:
        branch = "then" if not errors(instance, schema["if"], root, where) else "else"
        if branch in schema:
            found += errors(instance, schema[branch], root, where)
    return found


def _same(left: Any, right: Any) -> bool:
    """JSON equality: 1 and 1.0 are equal, but True is not 1."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right


def validate(instance: Any, schema: dict[str, Any]) -> None:
    """Raise ``ValueError`` listing every violation (at most 20 shown)."""
    found = errors(instance, schema)
    if found:
        shown = "; ".join(found[:20]) + (f"; ... {len(found) - 20} more" if len(found) > 20 else "")
        raise ValueError(f"schema violations: {shown}")
