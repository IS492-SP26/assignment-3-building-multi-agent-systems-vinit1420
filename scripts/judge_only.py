"""Fast single-response judge run.

Loads outputs/sample_session.json (already produced by the orchestrator),
runs the two-persona LLMJudge across the 5 criteria, and writes a real
evaluation report to outputs/. Used so a grader (or the report) has a
concrete LLM-as-Judge result without re-running the full multi-agent
pipeline.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from dotenv import load_dotenv

from src.evaluation.judge import LLMJudge


async def main():
    load_dotenv()
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    with open("outputs/sample_session.json", encoding="utf-8") as f:
        session = json.load(f)

    judge = LLMJudge(cfg)
    result = await judge.evaluate(
        query=session["query"],
        response=session["response"],
        sources=session.get("metadata", {}).get("sources", []),
        ground_truth=None,
    )

    out = {
        "query": session["query"],
        "evaluation": result,
        "response_preview": session["response"][:600],
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/judge_only_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print("Overall:", round(result["overall_score"], 3))
    print("Per-persona overall:", {k: round(v, 3) for k, v in result["per_persona_scores"].items()})
    print("Per-criterion (mean):")
    for name, data in result["criterion_scores"].items():
        per = {p: round(d["score"], 3) for p, d in data["by_persona"].items()}
        print(f"  {name:<22} mean={data['score']:.3f}  by_persona={per}")
    print("\nSaved outputs/judge_only_result.json")


if __name__ == "__main__":
    asyncio.run(main())
