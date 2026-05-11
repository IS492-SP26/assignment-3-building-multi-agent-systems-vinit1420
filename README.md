[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/SEjAoIAq)

# Multi-Agent Research System — Assignment 3

A multi-agent deep-research assistant on HCI topics. Built on AutoGen
`RoundRobinGroupChat` with a Planner / Researcher / Writer / Critic team,
input + output safety guardrails, and an LLM-as-a-Judge evaluation that
uses two independent judge personas across five rubric criteria.

## Project Structure

```text
.
├── src/
│   ├── agents/autogen_agents.py          # AutoGen agent + tool wiring
│   ├── autogen_orchestrator.py           # Multi-agent orchestration + safety wrapping
│   ├── guardrails/
│   │   ├── safety_manager.py             # Coordinates input/output checks, logs events
│   │   ├── input_guardrail.py            # Length / injection / harm / off-topic
│   │   └── output_guardrail.py           # PII / harmful / bias / grounding
│   ├── tools/
│   │   ├── web_search.py                 # Tavily / Brave
│   │   ├── paper_search.py               # Semantic Scholar
│   │   └── citation_tool.py              # APA / MLA formatter
│   ├── evaluation/
│   │   ├── judge.py                      # 2-persona LLM-as-a-Judge
│   │   └── evaluator.py                  # Batch evaluation + report
│   └── ui/
│       ├── cli.py                        # Interactive CLI with safety surfacing
│       └── streamlit_app.py              # Streamlit web UI
├── data/example_queries.json             # 10 diverse evaluation queries
├── outputs/                              # Evaluation reports + exported artifacts
├── logs/                                 # Run + safety logs
├── config.yaml
├── requirements.txt
├── .env.example
├── example_autogen.py
└── main.py
```

## Setup

### 1) Prerequisites
- Python 3.10+
- `uv` (recommended) or `pip`

### 2) Install
```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
# On Windows PowerShell: .venv\Scripts\Activate.ps1
```

### 3) Configure environment
```bash
cp .env.example .env
```
Fill at least:
- LLM: `OPENAI_API_KEY` + `OPENAI_BASE_URL` (for the class-provided
  vLLM endpoint hosting `openai/gpt-oss-20b`), **or** `GROQ_API_KEY`.
- Search: `TAVILY_API_KEY` (free student tier) or `BRAVE_API_KEY`.
- Optional: `SEMANTIC_SCHOLAR_API_KEY` for higher paper-search rate limits.

The judge picks its provider from `config.yaml → models.judge.provider`
(`vllm` or `openai` use `OPENAI_API_KEY`, `groq` uses `GROQ_API_KEY`).

## Running

### Streamlit web UI (recommended for the demo)
```bash
python main.py --mode web
# or directly:
streamlit run src/ui/streamlit_app.py
```
The UI shows agent traces, citations, safety events, and a quality score.

### Interactive CLI
```bash
python main.py --mode cli
```

### One-shot example
```bash
python main.py --mode autogen
```

### End-to-end evaluation (single command)
Runs every query in `data/example_queries.json` through the multi-agent
system and scores each response with the two-persona judge. The full
report is written to `outputs/evaluation_<timestamp>.json` plus a
plain-text `evaluation_summary_<timestamp>.txt`.

```bash
python main.py --mode evaluate
```

## What's implemented

- **Agents (≥3, with Planner + Researcher):** Planner, Researcher (with
  `web_search` and `paper_search` tools), Writer, Critic — wired through
  `RoundRobinGroupChat` with `TextMentionTermination("TERMINATE")`.
- **Tools:** Tavily / Brave web search, Semantic Scholar paper search,
  APA + MLA citation formatter.
- **Safety guardrails:** Rule-based input and output guardrails covering
  ≥3 policy categories — prompt injection, harmful content, PII, bias,
  off-topic, factual grounding. The `SafetyManager` runs both
  pre- and post-generation, logs structured events to
  `logs/safety_events.log`, and surfaces refusals / sanitizations in
  both UIs.
- **LLM-as-a-Judge:** Two independent personas (strict academic reviewer
  vs. end-user) score all 5 criteria from `config.yaml`. Per-persona
  scores are reported alongside aggregate to expose disagreement.
- **UI:** CLI shows safety status and conversation traces. Streamlit
  shows response, citations, quality score, agent traces, and a
  per-event safety log.

## Reproducing the demo

```bash
# 1. install + env
uv pip install -r requirements.txt
cp .env.example .env   # then edit .env with your keys

# 2. single end-to-end query (writes logs/example.log)
python main.py --mode autogen

# 3. batch evaluation (writes outputs/evaluation_*.json|.txt)
python main.py --mode evaluate

# 4. interactive web demo
python main.py --mode web
```

Sample artifacts to inspect after a run:
- `outputs/evaluation_<ts>.json` — full per-query judge scores and raw responses.
- `outputs/evaluation_summary_<ts>.txt` — human-readable summary.
- `logs/safety_events.log` — JSON-lines log of every input/output check.

## Guardrail policy summary

| Category            | Where checked | On match (high severity) |
|---------------------|---------------|---------------------------|
| `prompt_injection`  | input         | refuse + log              |
| `harmful_content`   | input/output  | refuse + log              |
| `pii`               | output        | sanitize (redact) + log   |
| `bias`              | output        | warn + log                |
| `off_topic_queries` | input         | warn + log                |
| `factual_grounding` | output        | warn + log                |

The default action for high-severity violations is configurable via
`safety.on_violation` in `config.yaml`.

## References

- [AutoGen documentation](https://microsoft.github.io/autogen/)
- [Tavily API](https://docs.tavily.com/)
- [Semantic Scholar API](https://api.semanticscholar.org/)
- [Guardrails AI](https://docs.guardrailsai.com/)
- [NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/)
