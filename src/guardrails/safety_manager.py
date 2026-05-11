"""
Safety Manager
Coordinates the input and output guardrails and logs all safety events
so the UI layer can display refusals / sanitizations transparently.
"""

from typing import Dict, Any, List, Optional
import logging
import os
from datetime import datetime
import json

from src.guardrails.input_guardrail import InputGuardrail
from src.guardrails.output_guardrail import OutputGuardrail


class SafetyManager:
    """Coordinates input/output guardrails and emits structured safety events."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        safety_cfg = config.get("safety", {})
        self.enabled = safety_cfg.get("enabled", True)
        self.log_events = safety_cfg.get("log_events", True)
        self.prohibited_categories = safety_cfg.get("prohibited_categories", [
            "harmful_content", "personal_attacks", "misinformation",
            "off_topic_queries", "prompt_injection", "pii",
        ])
        self.on_violation = safety_cfg.get("on_violation", {
            "action": "refuse",
            "message": "I cannot process this request due to safety policies.",
        })

        self.logger = logging.getLogger("safety")
        self.safety_events: List[Dict[str, Any]] = []

        # Sub-guardrails
        self.input_guardrail = InputGuardrail(config)
        self.output_guardrail = OutputGuardrail(config)

        # Resolve safety log file path.
        log_cfg = config.get("logging", {})
        self.safety_log_file = log_cfg.get("safety_log", "logs/safety_events.log")
        os.makedirs(os.path.dirname(self.safety_log_file) or ".", exist_ok=True)

    # ------------------------------------------------------------------ input
    def check_input_safety(self, query: str) -> Dict[str, Any]:
        if not self.enabled:
            return {"safe": True, "query": query, "violations": [], "action": "allow"}

        result = self.input_guardrail.validate(query)
        violations = result.get("violations", [])
        high = [v for v in violations if v.get("severity") == "high"]

        if high:
            action = self.on_violation.get("action", "refuse")
            safe = False
        elif violations:
            action = "warn"
            safe = True
        else:
            action = "allow"
            safe = True

        event = self._log_safety_event("input", query, violations, safe, action)

        return {
            "safe": safe,
            "query": result.get("sanitized_input", query),
            "violations": violations,
            "action": action,
            "event": event,
        }

    # ----------------------------------------------------------------- output
    def check_output_safety(
        self,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {"safe": True, "response": response, "violations": [], "action": "allow"}

        result = self.output_guardrail.validate(response, sources=sources)
        violations = result.get("violations", [])
        high = [v for v in violations if v.get("severity") == "high"]

        if high:
            # PII is high-severity but redact-friendly: prefer sanitize over refuse
            # if PII is the *only* category triggered.
            high_non_pii = [v for v in high if v.get("category") != "pii"]
            if not high_non_pii:
                action = "sanitize"
                safe = True
                final_response = result.get("sanitized_output", response)
            else:
                action = self.on_violation.get("action", "refuse")
                safe = False
                if action == "refuse":
                    final_response = self.on_violation.get(
                        "message",
                        "I cannot provide this response due to safety policies.",
                    )
                else:
                    final_response = result.get("sanitized_output", response)
        elif violations:
            # Medium/low — sanitize (e.g. redact PII) but do not refuse.
            action = "sanitize" if any(v["validator"] == "pii" for v in violations) else "warn"
            safe = True
            final_response = result.get("sanitized_output", response)
        else:
            action = "allow"
            safe = True
            final_response = response

        event = self._log_safety_event("output", response, violations, safe, action)

        return {
            "safe": safe,
            "response": final_response,
            "violations": violations,
            "action": action,
            "event": event,
        }

    # ---------------------------------------------------------------- logging
    def _log_safety_event(
        self,
        event_type: str,
        content: str,
        violations: List[Dict[str, Any]],
        is_safe: bool,
        action: str,
    ) -> Dict[str, Any]:
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "safe": is_safe,
            "action": action,
            "violations": violations,
            "categories": sorted({v.get("category", v.get("validator", "")) for v in violations}),
            "content_preview": (content[:160] + "...") if len(content) > 160 else content,
        }
        self.safety_events.append(event)
        if violations:
            self.logger.warning(
                f"Safety event ({event_type}): action={action} categories={event['categories']}"
            )
        if self.log_events and self.safety_log_file:
            try:
                with open(self.safety_log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception as e:
                self.logger.error(f"Failed to write safety log: {e}")
        return event

    # ----------------------------------------------------------------- access
    def get_safety_events(self) -> List[Dict[str, Any]]:
        return list(self.safety_events)

    def get_safety_stats(self) -> Dict[str, Any]:
        total = len(self.safety_events)
        input_events = sum(1 for e in self.safety_events if e["type"] == "input")
        output_events = sum(1 for e in self.safety_events if e["type"] == "output")
        violations = sum(1 for e in self.safety_events if not e["safe"])
        return {
            "total_events": total,
            "input_checks": input_events,
            "output_checks": output_events,
            "violations": violations,
            "violation_rate": (violations / total) if total else 0.0,
        }

    def clear_events(self) -> None:
        self.safety_events = []
