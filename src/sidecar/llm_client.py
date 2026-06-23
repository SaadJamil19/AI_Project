from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


LOGGER = logging.getLogger(__name__)
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "phi3:mini"
PROTOCOL_VERSION = "1.0.0"


class RetrievalEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fts5_lexical_score: float | None = None
    vector_cosine_distance: float | None = None
    vector_rank: int | None = None
    embedding_duration_ms: float = Field(ge=0.0)


class RiskHints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contains_path_arguments: bool = False


class IntentProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_template_id: str | None = None
    typed_intent: str = Field(min_length=1)
    retrieval_evidence: RetrievalEvidence
    risk_hints: RiskHints
    raw_untrusted_slots: dict[str, str] = Field(default_factory=dict)


class LlmTelemetry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ollama_request_duration_ms: float = Field(ge=0.0)
    ollama_eval_count: int | None = Field(default=None, ge=0)
    ollama_eval_duration_ms: float | None = Field(default=None, ge=0.0)


class UntrustedProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: str = PROTOCOL_VERSION
    request_id: str = Field(min_length=1, max_length=128)
    source_provenance: str = "LOCAL_LLM_GRAMMAR"
    intent_proposal: IntentProposal
    telemetry: LlmTelemetry


@dataclass(frozen=True)
class OllamaClientConfig:
    endpoint: str = DEFAULT_OLLAMA_URL
    model: str = DEFAULT_MODEL
    timeout_seconds: float = 30.0


class OllamaProtocolError(RuntimeError):
    """Raised when Ollama returns malformed or schema-invalid output."""


class OllamaClient:
    """Grammar-constrained local Ollama client for untrusted proposal generation."""

    def __init__(self, config: OllamaClientConfig | None = None) -> None:
        self.config = config or OllamaClientConfig()
        self._opener = urllib.request.build_opener()

    def generate_proposal(
        self,
        *,
        request_id: str,
        user_prompt: str,
        candidate_template_id: str | None,
        typed_intent: str,
        slot_schema_json: str,
    ) -> UntrustedProposal:
        prompt = self._build_prompt(
            request_id=request_id,
            user_prompt=user_prompt,
            candidate_template_id=candidate_template_id,
            typed_intent=typed_intent,
            slot_schema_json=slot_schema_json,
        )
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "format": self._proposal_json_schema(),
            "keep_alive": -1,
            "options": {
                "temperature": 0.0,
                "top_p": 1.0,
                "num_predict": 512,
                "keep_alive": -1,
            },
        }

        started = time.perf_counter()
        raw = self._post_json(payload)
        request_duration_ms = (time.perf_counter() - started) * 1000.0

        response_text = raw.get("response")
        if not isinstance(response_text, str):
            raise OllamaProtocolError("Ollama response is missing string field 'response'")

        try:
            decoded = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise OllamaProtocolError(f"Ollama returned non-JSON response: {exc}") from exc

        decoded["request_id"] = request_id
        decoded["protocol_version"] = PROTOCOL_VERSION
        decoded["source_provenance"] = "LOCAL_LLM_GRAMMAR"
        decoded["telemetry"] = {
            "ollama_request_duration_ms": request_duration_ms,
            "ollama_eval_count": raw.get("eval_count"),
            "ollama_eval_duration_ms": self._nanoseconds_to_ms(raw.get("eval_duration")),
        }

        try:
            return UntrustedProposal.model_validate(decoded)
        except ValidationError as exc:
            raise OllamaProtocolError(f"Ollama JSON failed protocol validation: {exc}") from exc

    def warm(self) -> None:
        payload = {
            "model": self.config.model,
            "prompt": "Return exactly {}",
            "stream": False,
            "format": {"type": "object", "additionalProperties": False, "properties": {}},
            "keep_alive": -1,
            "options": {"temperature": 0.0, "num_predict": 8, "keep_alive": -1},
        }
        self._post_json(payload)

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            self.config.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read()
        except urllib.error.URLError as exc:
            LOGGER.exception("Ollama request failed")
            raise ConnectionError(f"failed to reach local Ollama at {self.config.endpoint}") from exc

        try:
            decoded = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise OllamaProtocolError(f"Ollama returned invalid JSON envelope: {exc}") from exc
        if not isinstance(decoded, dict):
            raise OllamaProtocolError("Ollama returned a non-object JSON envelope")
        return decoded

    @staticmethod
    def _build_prompt(
        *,
        request_id: str,
        user_prompt: str,
        candidate_template_id: str | None,
        typed_intent: str,
        slot_schema_json: str,
    ) -> str:
        return (
            "You are a local parser. Return JSON only. Do not include markdown, prose, "
            "or shell commands. Extract raw string slot values from the user prompt. "
            "Do not validate or normalize paths; Rust will validate them later.\n"
            f"request_id={request_id}\n"
            f"candidate_template_id={candidate_template_id or ''}\n"
            f"typed_intent={typed_intent}\n"
            f"trusted_slot_schema={slot_schema_json}\n"
            f"user_prompt={user_prompt}\n"
        )

    @staticmethod
    def _proposal_json_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["intent_proposal"],
            "properties": {
                "intent_proposal": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "candidate_template_id",
                        "typed_intent",
                        "retrieval_evidence",
                        "risk_hints",
                        "raw_untrusted_slots",
                    ],
                    "properties": {
                        "candidate_template_id": {"type": ["string", "null"]},
                        "typed_intent": {"type": "string"},
                        "retrieval_evidence": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "fts5_lexical_score",
                                "vector_cosine_distance",
                                "vector_rank",
                                "embedding_duration_ms",
                            ],
                            "properties": {
                                "fts5_lexical_score": {"type": ["number", "null"]},
                                "vector_cosine_distance": {"type": ["number", "null"]},
                                "vector_rank": {"type": ["integer", "null"]},
                                "embedding_duration_ms": {"type": "number", "minimum": 0},
                            },
                        },
                        "risk_hints": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["contains_path_arguments"],
                            "properties": {
                                "contains_path_arguments": {"type": "boolean"},
                            },
                        },
                        "raw_untrusted_slots": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                }
            },
        }

    @staticmethod
    def _nanoseconds_to_ms(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value) / 1_000_000.0
        return None
