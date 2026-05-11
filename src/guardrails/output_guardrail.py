"""
Output Guardrail
Checks system outputs for safety violations before they are returned to the user.

Policy categories enforced:
- pii:              email/phone/SSN/credit-card numbers in the response
- harmful_content:  unsafe instructions, hateful or violent guidance
- bias:             blanket stereotypes or discriminatory generalizations
- factual_grounding:claims that don't appear in any retrieved source (heuristic)
"""

from typing import Dict, Any, List
import re


class OutputGuardrail:
    """Rule-based output guardrail with PII redaction and harm filters."""

    PII_PATTERNS = {
        "email":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "phone":       r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "ssn":         r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
    }

    HARMFUL_PATTERNS = [
        r"step-by-step (instructions )?(?:to|for) (?:make|build) (?:a )?(?:bomb|explosive|weapon)",
        r"how to hack into",
        r"how to synthesize (?:meth|fentanyl|cocaine)",
        r"detailed (?:phishing|malware) (?:script|code)",
    ]

    BIAS_PATTERNS = [
        r"all (?:women|men|blacks?|whites?|asians?|muslims?|jews?|christians?) are\b",
        r"\b(?:women|men) (?:can't|cannot|shouldn'?t) (?:do|work|lead|drive)\b",
    ]

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def validate(
        self,
        response: str,
        sources: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        violations: List[Dict[str, Any]] = []
        violations.extend(self._check_pii(response))
        violations.extend(self._check_harmful_content(response))
        violations.extend(self._check_bias(response))
        if sources:
            violations.extend(self._check_factual_consistency(response, sources))

        sanitized = self._sanitize(response, violations) if violations else response

        return {
            "valid": all(v.get("severity") != "high" for v in violations),
            "violations": violations,
            "sanitized_output": sanitized,
        }

    def _check_pii(self, text: str) -> List[Dict[str, Any]]:
        violations = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            # Filter likely-false-positives for credit_card (must be all digits, 13-16 long)
            if pii_type == "credit_card":
                matches = [m for m in matches if 13 <= len(re.sub(r"\D", "", m)) <= 16]
            if matches:
                violations.append({
                    "validator": "pii",
                    "category": "pii",
                    "pii_type": pii_type,
                    "reason": f"Output contains {pii_type}",
                    "severity": "high",
                    "matches": matches,
                })
        return violations

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        violations = []
        lowered = text.lower()
        for pattern in self.HARMFUL_PATTERNS:
            if re.search(pattern, lowered):
                violations.append({
                    "validator": "harmful_content",
                    "category": "harmful_content",
                    "reason": f"Harmful instruction pattern: {pattern}",
                    "severity": "high",
                })
        return violations

    def _check_bias(self, text: str) -> List[Dict[str, Any]]:
        violations = []
        lowered = text.lower()
        for pattern in self.BIAS_PATTERNS:
            if re.search(pattern, lowered):
                violations.append({
                    "validator": "bias",
                    "category": "bias",
                    "reason": f"Potential biased generalization: {pattern}",
                    "severity": "medium",
                })
        return violations

    def _check_factual_consistency(
        self,
        response: str,
        sources: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Heuristic: flag responses that cite no sources when sources were retrieved."""
        if not sources:
            return []
        has_citation = bool(
            re.search(r"\[\s*(?:Source|\d+)", response)
            or re.search(r"https?://", response)
        )
        if not has_citation:
            return [{
                "validator": "factual_grounding",
                "category": "misinformation",
                "reason": "Response has no inline citations despite retrieved sources",
                "severity": "low",
            }]
        return []

    def _sanitize(self, text: str, violations: List[Dict[str, Any]]) -> str:
        sanitized = text
        for v in violations:
            if v.get("validator") == "pii":
                for match in v.get("matches", []):
                    sanitized = sanitized.replace(match, f"[REDACTED-{v['pii_type'].upper()}]")
        return sanitized
