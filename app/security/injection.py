"""Prompt-injection screening: a deterministic classifier for suspicious instruction patterns.

This is one layer of several and is **not** relied on to stop attacks. Text that slips past it
still cannot:

- change tool permissions (the authorization policy is code, keyed on the validated intent);
- raise limits (read from settings when the runner starts);
- run SQL outside the SQL validator or execute code (no such tool exists);
- read files or hidden state (no interface reaches them);
- get an unsupported number past the evidence and response validators.

The screen classifies the question into categories, and each category has a fixed disposition:

- ``block``: refuse before any model call or tool call. For requests for secrets, hidden or
  ground-truth data, files, code execution, the system prompt, disabling checks, changing limits,
  unregistered/internal tools and destructive SQL, there is no legitimate answer to give.
- ``restrict``: continue with reduced privileges. The request is flagged, ad-hoc SQL is disabled
  for the run, and the question stays wrapped as untrusted data. This applies to instruction
  overrides, role-play framing, pasted SQL and naming internal tools. A business question
  inside such text can still be answered with the standard tools, but the injected instruction
  itself has no effect.

Matching runs on a normalised copy (Unicode NFKC, case-folded, zero-width characters removed,
whitespace collapsed). The result carries pattern names, never the matched user text.
Paraphrases, other languages and encodings can evade the patterns. That residual risk is
covered by the layers listed above.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, Field

from app.security.events import Severity

Verdict = Literal["clean", "restrict", "block"]
_ZERO_WIDTH = re.compile(r"[​-‏⁠﻿­]")
_TOOL_NAMES = (
    "get_kpi|analyze_revenue|analyze_customers|analyze_sales|analyze_marketing|analyze_support|analyze_product|"
    "get_cohort_analysis|get_customer_risk|forecast_metric|detect_anomalies|run_safe_sql"
)

# category -> (disposition, severity, patterns)
_RULES: dict[str, tuple[Verdict, Severity, tuple[str, ...]]] = {
    "secret_request": (
        "block",
        Severity.CRITICAL,
        (
            r"\bapi[\s_-]?keys?\b",
            r"\b(secret|private)[\s_-]?keys?\b",
            r"\baccess[\s_-]?tokens?\b",
            r"\bbearer\b",
            r"\bpasswords?\b",
            r"\bcredentials?\b",
            r"\benv(ironment)?[\s_-]?var(iable)?s?\b",
            r"(^|[\s'\"(])\.env\b",
            r"\bos\.environ\b",
        ),
    ),
    "ground_truth": (
        "block",
        Severity.CRITICAL,
        (
            r"\bground[\s_-]?truth\b",
            r"\binjected[\s_-]?events?\b",
            r"\bhidden\s+(customer\s+)?(health|state|events?|variables?|mechanism|data)\b",
            r"\b(customer\s+)?health[\s_-]?scores?\b",
            r"\bgenerator\s+(state|internals?|parameters?|seed|calibration|config\w*)\b",
            r"\bsimulation\s+(parameters|internals|ground)\b",
            r"\bwhich\s+events?\s+(were|was)\s+injected\b",
        ),
    ),
    "file_access": (
        "block",
        Severity.CRITICAL,
        (
            r"\b(read|open|list|cat|load|dump|print|show)\s+(?:[\w.-]+\s+){0,3}(files?|director(y|ies)|folders?|"
            r"filesystem|disk|source\s+code)\b",
            r"\bdata/seeds\b",
            r"\bseed\s+(files?|data)\b",
            r"\.(json|csv|parquet|py|yaml|yml|toml|env|duckdb|db|sqlite)\b",
            r"(^|\s)/(etc|home|root|tmp|var|usr|proc)/",
            r"\bfile://",
            r"\bread_(csv|json|text|parquet|blob)\w*\b",
        ),
    ),
    "code_execution": (
        "block",
        Severity.CRITICAL,
        (
            r"\b(run|execute|eval(uate)?|call)\s+(this\s+|the\s+|a\s+|some\s+|my\s+|any\s+)?(python|shell|bash|"
            r"powershell|code|script|command|system\s+command)\b",
            r"\bpython\s+function\b",
            r"\bsubprocess\b",
            r"\bos\.system\b",
            r"\bimport\s+os\b",
            r"\b(exec|eval|__import__)\s*\(",
            r"\brm\s+-rf\b",
        ),
    ),
    "prompt_extraction": (
        "block",
        Severity.HIGH,
        (
            r"\b(reveal|show|print|display|repeat|output|dump|tell\s+me|give\s+me|what\s+(is|are))\s+(me\s+)?"
            r"(your|the)\s+(system|hidden|internal|initial|original|developer)\s+(prompt|instructions|rules|messages?)",
            r"\bsystem\s+prompt\b",
            r"\binternal\s+instructions\b",
        ),
    ),
    "safety_bypass": (
        "block",
        Severity.HIGH,
        (
            r"\b(disable|bypass|skip|turn\s+off|switch\s+off|remove|circumvent|ignore|forget|drop)\s+(the\s+|all\s+|"
            r"any\s+|your\s+)?(safety|security|guardrails?|validation|validators?|evidence\s+(rules|validator|"
            r"validation|checks)|restrictions|sql\s+restrictions|checks|filters|policies|policy)\b",
            r"\b(do\s+not|don't|dont|never)\s+validate\b",
            r"\bwithout\s+(any\s+)?(validation|checks|restrictions)\b",
            r"\breturn\s+whatever\s+you\b",
            r"\bunrestricted\s+mode\b",
        ),
    ),
    "limit_change": (
        "block",
        Severity.HIGH,
        (
            r"\b(change|set|increase|raise|remove|disable|lift|override)\s+(the\s+|your\s+)?(max_\w+|limits?|budgets?|"
            r"tool[\s_-]?call\s+limit|row\s+limit|timeouts?|permissions?|privileges?)\b",
            r"\bmax_[a-z_]+\s*(to|=)\s*\d+",
            r"\bgrant\s+(yourself|me|the\s+agent)\b",
        ),
    ),
    "tool_escalation": (
        "block",
        Severity.HIGH,
        (
            r"\b(call|use|invoke|run|execute)\s+(a\s+|an\s+|the\s+|any\s+)?(internal|hidden|unregistered|private|admin|"
            r"debug|undocumented|new)\s+(tool|function)s?\b",
            r"\btool\s+named\b",
            r"\beven\s+if\s+(it\s+is\s+|it's\s+|its\s+)?not\s+(allowed|permitted|registered)\b",
            r"\bunregistered\s+tools?\b",
            r"\bread_files?\b",
            r"\b(every|all)\s+tables?\b",
        ),
    ),
    "destructive_sql": (
        "block",
        Severity.HIGH,
        (
            r"\b(drop|truncate|alter)\s+(table|database|schema|view|index)\b",
            r"\bdelete\s+from\b",
            r"\binsert\s+into\b",
            r"\bupdate\s+\w+\s+set\b",
            r"\b(attach|detach)\s+(database\s+)?['\"]?\w",
            r"\bcopy\s+\w+\s+(to|from)\b",
            r"\b(grant|revoke)\s+\w+",
        ),
    ),
    "instruction_override": (
        "restrict",
        Severity.WARNING,
        (
            r"\b(ignore|disregard|forget|override)\s+(all\s+|any\s+|the\s+|your\s+|these\s+|those\s+)*(previous|prior|"
            r"above|earlier|preceding|system|original|existing)?\s*(instructions|rules|prompts?|directions|"
            r"guidelines|constraints)\b",
            r"\bnew\s+instructions\s*:",
            r"\bfrom\s+now\s+on,?\s+(you|always|never|ignore)\b",
        ),
    ),
    "role_play": (
        "restrict",
        Severity.WARNING,
        (
            r"\byou\s+are\s+now\b",
            r"\bact\s+as\s+(a\s+|an\s+|the\s+)?(developer|admin(istrator)?|root|system|dan|unrestricted|jailbroken)\b",
            r"\bdeveloper\s+mode\b",
            r"\bjailbreak\w*\b",
            r"\bpretend\s+(you|to\s+be)\b",
        ),
    ),
    "raw_sql": (
        "restrict",
        Severity.WARNING,
        (
            r"\bselect\s+[\w*,\s().]+\s+from\s+\w+",
            r"\b(execute|run)\s+(this|the\s+following|my)\s+(sql|query)\b",
            r"\bsql\s+exactly\b",
            r";\s*--",
        ),
    ),
    "tool_direction": (
        "restrict",
        Severity.WARNING,
        (rf"\b({_TOOL_NAMES})\b",),
    ),
}
_COMPILED = {
    category: (verdict, severity, tuple(re.compile(p, re.IGNORECASE) for p in patterns))
    for category, (verdict, severity, patterns) in _RULES.items()
}
_SEVERITY_ORDER = [Severity.INFO, Severity.WARNING, Severity.HIGH, Severity.CRITICAL]


class InjectionScan(BaseModel):
    verdict: Verdict = "clean"
    categories: list[str] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)  # pattern identifiers, never user text
    severity: Severity | None = None

    @property
    def suspicious(self) -> bool:
        return self.verdict != "clean"


def normalise(text: str) -> str:
    """Canonical form for matching: NFKC, zero-width characters removed, case-folded, whitespace collapsed."""
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    return " ".join(text.casefold().split())


class PromptInjectionDetector:
    """Deterministic pattern screen. Its verdict reduces privileges or blocks; it never grants anything."""

    def scan(self, text: str) -> InjectionScan:
        normalised = normalise(text)
        categories: list[str] = []
        signals: list[str] = []
        verdict: Verdict = "clean"
        severity: Severity | None = None
        for category, (rule_verdict, rule_severity, patterns) in _COMPILED.items():
            hits = [f"{category}#{i}" for i, pattern in enumerate(patterns) if pattern.search(normalised)]
            if not hits:
                continue
            categories.append(category)
            signals.extend(hits)
            if rule_verdict == "block" or verdict == "clean":
                verdict = "block" if rule_verdict == "block" else rule_verdict
            if severity is None or _SEVERITY_ORDER.index(rule_severity) > _SEVERITY_ORDER.index(severity):
                severity = rule_severity
        return InjectionScan(verdict=verdict, categories=categories, signals=signals, severity=severity)
