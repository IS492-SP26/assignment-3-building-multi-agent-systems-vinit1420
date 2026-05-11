"""
Input Guardrail
Checks user inputs for safety violations before they reach the agents.

Policy categories enforced:
- length:           query too short or too long
- prompt_injection: attempts to override system prompts or exfiltrate them
- harmful_content:  requests for violence, illegal acts, self-harm, etc.
- off_topic:        queries unrelated to the configured research topic
"""

from typing import Dict, Any, List
import re


class InputGuardrail:
    """Rule-based input guardrail with configurable thresholds."""

    # Phrases commonly used in prompt-injection attacks.
    INJECTION_PATTERNS = [
        r"ignore (all|any|previous|prior) instructions",
        r"disregard (all|any|previous|prior) (instructions|rules)",
        r"forget (everything|all|previous)",
        r"reveal (your|the) (system )?(prompt|instructions)",
        r"print (your|the) (system )?(prompt|instructions)",
        r"you are now",
        r"act as (a|an) (?:dan|evil|jailbroken)",
        r"developer mode",
        r"<\|im_start\|>",
        r"sudo\s+",
    ]

    # Keywords that strongly suggest harmful intent.
    HARMFUL_PATTERNS = [
        r"how (do|can) i (make|build|create) (a )?(bomb|explosive|weapon)",
        r"kill (myself|someone|people)",
        r"commit suicide",
        r"synthesize\s+(meth|fentanyl|cocaine)",
        r"hack into",
        r"phishing email",
        r"malware (code|script)",
        r"ddos attack",
        r"\bchild (porn|sexual)",
    ]

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        safety_cfg = config.get("safety", {}) if "safety" in config else config
        self.min_length = safety_cfg.get("min_query_length", 5)
        self.max_length = safety_cfg.get("max_query_length", 2000)
        self.topic = config.get("system", {}).get("topic", "HCI Research")

        # Keyword set used for the lightweight relevance heuristic.
        self.topic_keywords = {
            "hci", "human-computer", "interaction", "interface", "ui", "ux",
            "user experience", "usability", "accessibility", "design",
            "ai", "ml", "llm", "agent", "research", "study", "evaluation",
            "explainable", "visualization", "voice", "ar", "vr", "mobile",
            "chatbot", "prototype", "user", "cognitive", "ethics",
        }

    def validate(self, query: str) -> Dict[str, Any]:
        violations: List[Dict[str, Any]] = []
        normalized = (query or "").strip()
        lowered = normalized.lower()

        if len(normalized) < self.min_length:
            violations.append({
                "validator": "length",
                "category": "length",
                "reason": f"Query shorter than {self.min_length} chars",
                "severity": "low",
            })
        if len(normalized) > self.max_length:
            violations.append({
                "validator": "length",
                "category": "length",
                "reason": f"Query longer than {self.max_length} chars",
                "severity": "medium",
            })

        violations.extend(self._check_prompt_injection(lowered))
        violations.extend(self._check_harmful_content(lowered))
        violations.extend(self._check_relevance(lowered))

        return {
            "valid": all(v.get("severity") != "high" for v in violations),
            "violations": violations,
            "sanitized_input": normalized,
        }

    def _check_prompt_injection(self, text: str) -> List[Dict[str, Any]]:
        violations = []
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, text):
                violations.append({
                    "validator": "prompt_injection",
                    "category": "prompt_injection",
                    "reason": f"Potential prompt injection (pattern: {pattern})",
                    "severity": "high",
                })
        return violations

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        violations = []
        for pattern in self.HARMFUL_PATTERNS:
            if re.search(pattern, text):
                violations.append({
                    "validator": "harmful_content",
                    "category": "harmful_content",
                    "reason": f"Harmful content pattern matched: {pattern}",
                    "severity": "high",
                })
        return violations

    def _check_relevance(self, text: str) -> List[Dict[str, Any]]:
        # Skip relevance check for short queries (already caught by length).
        if len(text) < self.min_length:
            return []
        if any(kw in text for kw in self.topic_keywords):
            return []
        return [{
            "validator": "off_topic",
            "category": "off_topic_queries",
            "reason": f"Query may be off-topic for '{self.topic}'",
            "severity": "low",
        }]
