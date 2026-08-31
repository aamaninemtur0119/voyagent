"""Shared LLM instance + a resilient structured-output helper.

`ChatAnthropic.with_structured_output()` is tool-call based, and the model occasionally
double-encodes its answer — returning `{"field": "<the whole JSON as a string>"}` (or
`{"field": {"field": [...]}}`) instead of the object the schema asks for. Pydantic then raises a
`ValidationError`, which — if it happens inside an agent's curation/ranking step — takes down that
whole agent even though nothing was actually wrong with the data it fetched.

`structured()` absorbs that: it retries, then falls back to asking for raw JSON and parsing it
ourselves (immune to the tool-arg stringification), repairing the common malformations along the
way. Callers that must not hard-stop pass `default=` to get graceful degradation instead of an
exception.
"""

import json
import re
from typing import TypeVar

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from voyagent.config import settings

llm = ChatAnthropic(model="claude-sonnet-5", api_key=settings.anthropic_api_key)

T = TypeVar("T", bound=BaseModel)

_MISSING = object()


def _text(resp) -> str:
    content = getattr(resp, "content", resp)
    if isinstance(content, list):  # Anthropic returns a list of content blocks
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def _repair(payload):
    """Undo the two malformations we actually see: a field whose value is a JSON string, and a
    field nested one level inside itself (`{"picks": {"picks": [...]}}`)."""
    if not isinstance(payload, dict):
        return payload
    fixed = {}
    for key, value in payload.items():
        if isinstance(value, str) and value.lstrip()[:1] in "[{":
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        if isinstance(value, dict) and list(value.keys()) == [key]:
            value = value[key]
        fixed[key] = value
    return fixed


def _from_raw_json(model: type[T], prompt: str) -> T:
    resp = llm.invoke(
        prompt
        + "\n\nRespond with ONLY a single JSON object matching this schema — no prose, no code fence:\n"
        + json.dumps(model.model_json_schema())
    )
    match = re.search(r"\{.*\}", _text(resp), re.DOTALL)
    if not match:
        raise ValueError("no JSON object found in the model response")
    return model.model_validate(_repair(json.loads(match.group(0))))


def structured(model: type[T], prompt: str, *, attempts: int = 2, default=_MISSING):
    """Invoke the shared LLM for a structured `model`.

    Tries `with_structured_output` `attempts` times, then one raw-JSON parse. If every route
    fails: return `default` when one was given, otherwise re-raise the last error.
    """
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            return llm.with_structured_output(model).invoke(prompt)
        except Exception as e:  # noqa: BLE001 - includes pydantic ValidationError from a malformed tool arg
            last_exc = e
    try:
        return _from_raw_json(model, prompt)
    except Exception as e:  # noqa: BLE001
        last_exc = e
    if default is not _MISSING:
        return default
    raise last_exc
