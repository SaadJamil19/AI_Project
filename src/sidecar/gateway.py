from __future__ import annotations

import json
import logging
import re
import shlex
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict


LOGGER = logging.getLogger(__name__)
FAST_PATH_TRUST_STATES = {"STATIC_VERIFIED", "PROMOTED_FASTPATH"}
PLACEHOLDER_RE = re.compile(r"^(?:\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?|<([A-Za-z_][A-Za-z0-9_]*)>|:([A-Za-z_][A-Za-z0-9_]*))$")


class FastPathDecision(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    proposal: Any | None = None
    rejection_reason: str | None = None


class FastPathGateway:
    """Deterministic bypass gate for static, low-risk command templates only."""

    def evaluate(
        self,
        query: str,
        request_id: str,
        hits: Sequence[Any],
        evidence_factory: Callable[[Any], Any],
    ) -> FastPathDecision:
        tokens = self._command_tokens(query)
        if not tokens:
            return FastPathDecision(rejection_reason="prompt_is_not_command_like")

        for hit in hits:
            candidate = hit.candidate
            if not self._prefix_matches(tokens, candidate.binary_name, candidate.subcommand_path):
                continue
            if candidate.trust_state not in FAST_PATH_TRUST_STATES:
                return FastPathDecision(rejection_reason="candidate_trust_state_not_fast_path")
            if not self._policy_is_low_risk(candidate):
                return FastPathDecision(rejection_reason="candidate_policy_not_low_risk")

            slot_start = 1 + len(self._subcommand_tokens(candidate.subcommand_path))
            slots = self._extract_slots(candidate.template_argv_json, tokens[slot_start:])
            if slots is None:
                return FastPathDecision(rejection_reason="slot_extraction_failed")

            try:
                from .search import (
                    IntentProposal,
                    RiskHints,
                    SidecarTelemetry,
                    UntrustedProposal,
                )

                proposal = UntrustedProposal(
                    request_id=request_id,
                    source_provenance="LOCAL_FAST_PATH",
                    intent_proposal=IntentProposal(
                        candidate_template_id=candidate.doc_id,
                        typed_intent=candidate.typed_intent,
                        retrieval_evidence=evidence_factory(hit),
                        risk_hints=RiskHints(
                            contains_path_arguments=self._contains_path_argument(slots)
                        ),
                        raw_untrusted_slots=slots,
                    ),
                    telemetry=SidecarTelemetry(
                        faiss_matrix_scan_duration_ms=0.0,
                        python_scheduling_delay_ms=0.0,
                    ),
                )
                return FastPathDecision(proposal=proposal)
            except Exception:
                LOGGER.exception("failed to construct fast-path proposal")
                raise

        return FastPathDecision(rejection_reason="no_exact_prefix_match")

    @staticmethod
    def _command_tokens(query: str) -> list[str]:
        try:
            tokens = shlex.split(query, posix=True)
        except ValueError:
            tokens = query.split()
        return [token for token in tokens if token]

    @staticmethod
    def _subcommand_tokens(subcommand_path: str) -> list[str]:
        return [token for token in subcommand_path.split() if token]

    def _prefix_matches(
        self,
        tokens: Sequence[str],
        binary_name: str,
        subcommand_path: str,
    ) -> bool:
        if not tokens or tokens[0] != binary_name:
            return False
        subcommands = self._subcommand_tokens(subcommand_path)
        if not subcommands:
            return True
        return list(tokens[1 : 1 + len(subcommands)]) == subcommands

    @staticmethod
    def _policy_is_low_risk(candidate: Any) -> bool:
        if not bool(candidate.fast_path_allowed):
            return False
        if int(candidate.required_confirmation_count) > 1:
            return False

        risky_surfaces = (
            candidate.package_manager_risk_level,
            candidate.privilege_risk_level,
            candidate.destructive_recursive_level,
        )
        return all(str(value) == "BLOCK" for value in risky_surfaces)

    def _extract_slots(
        self,
        template_argv_json: str,
        remaining_tokens: Sequence[str],
    ) -> dict[str, str] | None:
        try:
            template = json.loads(template_argv_json)
        except json.JSONDecodeError:
            LOGGER.error("template_argv_json is malformed")
            return None
        if not isinstance(template, list) or not all(isinstance(item, str) for item in template):
            LOGGER.error("template_argv_json must be a JSON string array")
            return None

        slot_names: list[str] = []
        for item in template:
            match = PLACEHOLDER_RE.match(item)
            if match is None:
                continue
            name = next(group for group in match.groups() if group is not None)
            if name not in slot_names:
                slot_names.append(name)

        if len(remaining_tokens) < len(slot_names):
            return None
        if not slot_names and remaining_tokens:
            return {"argv_tail": " ".join(remaining_tokens)}

        return {
            name: str(remaining_tokens[index])
            for index, name in enumerate(slot_names)
        }

    @staticmethod
    def _contains_path_argument(slots: dict[str, str]) -> bool:
        return any(
            "/" in value.replace("\\", "/")
            or "." in value
            or value.startswith("~")
            for value in slots.values()
        )
