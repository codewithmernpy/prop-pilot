"""AI layer exports: Anthropic rule extractor and tool-using Copilot with offline fallback."""

from proppilot.ai.copilot import (
    COPILOT_PROMPT,
    CopilotReply,
    build_copilot_tools,
    explain_plan,
    generate_offline_plan_explanation,
    run_copilot_query,
)
from proppilot.ai.extractor import (
    EXTRACT_PROMPT,
    RuleExtractionError,
    extract_rules_from_text,
    extract_rules_offline,
)
from proppilot.ai.session import PlanSession

__all__ = [
    "COPILOT_PROMPT",
    "CopilotReply",
    "EXTRACT_PROMPT",
    "PlanSession",
    "RuleExtractionError",
    "build_copilot_tools",
    "explain_plan",
    "extract_rules_from_text",
    "extract_rules_offline",
    "generate_offline_plan_explanation",
    "run_copilot_query",
]
