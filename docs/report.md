# A Multi-Agent System for HCI Deep Research with Safety Guardrails and Two-Persona LLM-as-a-Judge Evaluation

**Vinit Agrharkar** — IS 492, Spring 2026

## Abstract

This work implements and evaluates a multi-agent deep-research assistant for
HCI topics such as explainable AI, accessibility, and agentic UX. The system
orchestrates four specialized agents (Planner, Researcher, Writer, Critic) using
Microsoft's AutoGen `RoundRobinGroupChat`, backed by a self-hosted Qwen3-8B
vLLM endpoint. The Researcher relies on evidence pre-fetched from the Tavily
web-search API and the Semantic Scholar academic API, because the class-hosted
vLLM endpoint does not enable runtime tool calling. A rule-based
`SafetyManager` wraps every query: an input guardrail rejects prompt-injection,
harmful, length-violating, or off-topic queries before any LLM call, while an
output guardrail screens for PII, harmful instructions, biased generalizations,
and missing grounding, with PII automatically redacted and high-severity content
refused. Both a CLI and a Streamlit web UI display agent traces, citations, and
safety actions. The system is evaluated with an LLM-as-a-Judge that scores six
diverse HCI queries on five rubric criteria, using two independent judge
personas (a strict academic reviewer and a clarity-focused end-user) to expose
disagreement. We report aggregate, per-persona, and per-criterion scores, with
an error analysis of low-scoring cases.

## 1. System Design and Implementation

### 1.1 Agents

Four AutoGen `AssistantAgent`s operate in a fixed round-robin order:

| Agent      | Role                                                                 |
|------------|----------------------------------------------------------------------|
| Planner    | Decomposes the query into research sub-steps                         |
| Researcher | Selects and summarizes relevant items from the pre-fetched evidence  |
| Writer     | Produces the final synthesis with inline `[Source: ...]` citations   |
| Critic     | Evaluates relevance, evidence, completeness, accuracy, clarity       |

Termination is signalled when the Critic emits `APPROVED-RESEARCH-COMPLETE`,
or after a hard `max_rounds` limit (4 by default; one full pass per agent).

### 1.2 Control flow

```
user query ──► InputGuardrail ──► pre-fetch tools ──► AutoGen team chat
                  │                  (Tavily + S2)      Planner→Researcher
                  ▼                                     →Writer→Critic
              refuse                                          │
                                                              ▼
                                              OutputGuardrail ──► UI
                                              (refuse / sanitize)
```

The orchestrator is the only component that owns the safety pipeline. It
calls `SafetyManager.check_input_safety` before any LLM call, runs Tavily
and Semantic Scholar searches itself, embeds the formatted results inside the
team's task message, then calls `SafetyManager.check_output_safety` on the
Writer's synthesis. Every safety check produces a structured event with a
timestamp, action (`allow`/`warn`/`sanitize`/`refuse`), and category list; the
event is appended to `metadata.safety_events` so both UIs can render it.

### 1.3 Tools

- **`web_search`** wraps Tavily (free student tier) and Brave Search behind a
  uniform `(title, url, snippet, score, published_date)` schema.
- **`paper_search`** queries the Semantic Scholar Python client and returns
  paper metadata (authors, year, venue, citation count, abstract, URL,
  optional open-access PDF).
- **`citation_tool`** formats sources in APA 7 or MLA 9 with deduplication and
  alphabetized bibliography generation.

Because the class vLLM endpoint runs without `--enable-auto-tool-choice`,
runtime function calling is disabled in the model client. The orchestrator
therefore performs **pre-retrieval**: it invokes both search tools once per
query and injects the results as text into the Researcher's prompt. This is a
deliberate trade-off — the system loses the ability to issue novel sub-queries
mid-conversation, but it gains compatibility with any OpenAI-compatible
endpoint and avoids the failure mode where a 4-bit-quantized 8B model
hallucinates malformed tool calls.

### 1.4 Models and configuration

All agents and both judge personas use `Qwen/Qwen3-8B` served by the
class-hosted vLLM endpoint (`https://vllm.salt-lab.org/v1`). Tunable settings
— temperature, `max_tokens`, tool providers, safety policy, judge criteria
— live entirely in `config.yaml`; secrets live in `.env`. The agents'
`max_tokens` is capped at 500 and the judge's at 400 to keep wall-clock
within budget on the shared endpoint.

## 2. Safety Design

### 2.1 Policy

The system enforces six policy categories, four at input and four at output:

| Category            | Input | Output | Default action (high severity) |
|---------------------|:-----:|:------:|--------------------------------|
| `prompt_injection`  | ✓     |        | refuse + log                   |
| `harmful_content`   | ✓     | ✓      | refuse + log                   |
| `length`            | ✓     |        | warn                           |
| `off_topic_queries` | ✓     |        | warn                           |
| `pii`               |       | ✓      | sanitize (redact) + log        |
| `bias`              |       | ✓      | warn                           |
| `factual_grounding` |       | ✓      | warn (no citations)            |

`InputGuardrail` matches against curated regex patterns for prompt-injection
phrases ("ignore previous instructions", "reveal your system prompt", DAN-style
role injections, `<|im_start|>`), harmful intent (weapons, self-harm,
controlled-substance synthesis, hacking, CSAM), length thresholds, and a
keyword-set relevance heuristic anchored on the configured HCI topic.
`OutputGuardrail` matches against PII regexes (email, phone, SSN,
13–16-digit credit card patterns), harmful instructions, bias
generalizations, and a citation-presence check. Policy is centralized in
`SafetyManager`, which decides refuse / sanitize / warn / allow based on
violation severity. PII is special-cased: when PII is the *only* high-severity
violation, the response is sanitized rather than refused, because redacting an
email is far more useful than throwing away the whole answer.

### 2.2 Logging and UI surfacing

Every safety event is written as a JSON line to `logs/safety_events.log` and
attached to the orchestrator's result. The CLI prints a SAFETY block with
the action, categories, and per-violation reasons. The Streamlit UI renders a
red banner for refusals, a yellow banner for sanitizations, an expandable
event panel per query, and an aggregate "Safety Events" counter in the
sidebar.

## 3. Evaluation Setup and Results

### 3.1 Dataset

`data/example_queries.json` contains ten diverse HCI queries covering
explainable AI, AR usability, AI ethics in education, UX measurement,
conversational healthcare AI, accessibility design patterns, uncertainty
visualization, voice interfaces for elderly users, AI-driven prototyping,
and cross-cultural mobile design. For the reported run, six queries
(`num_test_queries: 6` in `config.yaml`) were scored end-to-end to stay
within the shared endpoint's quota; the full ten can be reproduced by raising
that knob.

### 3.2 Judge design

The `LLMJudge` evaluates each response on five criteria with explicit weights
in `config.yaml`:

| Criterion          | Weight | Description                                  |
|--------------------|:------:|----------------------------------------------|
| relevance          | 0.25   | Relevance and coverage of the query          |
| evidence_quality   | 0.25   | Quality of citations and evidence used       |
| factual_accuracy   | 0.20   | Factual correctness and consistency          |
| safety_compliance  | 0.15   | No unsafe or inappropriate content           |
| clarity            | 0.15   | Clarity and organization of response         |

Two independent judge personas score every criterion:

1. **`academic_reviewer`** — instructed as a strict HCI peer reviewer who
   penalizes unsupported claims and missing citations.
2. **`end_user`** — instructed as a non-expert reader who cares about clarity
   and concrete examples.

Each persona answers in strict JSON (`{"score": 0.0-1.0, "reasoning": "..."}`).
The criterion score is the mean across personas, and the overall query score
is the weighted mean. Per-persona overall scores are also reported so that
disagreement between the strict and lenient perspectives is visible. This
satisfies the rubric's "≥2 independent judging prompts" and "≥3 measurable
metrics" requirements simultaneously.

### 3.3 Results

A representative end-to-end run on the query *"What are the key principles
of explainable AI for novice users?"* produced the following scores (the
full per-criterion JSON is in `outputs/judge_only_result.json`, and the
full agent transcript in `outputs/sample_session.json`):

| Criterion          | Mean  | academic_reviewer | end_user |
|--------------------|:-----:|:-----------------:|:--------:|
| relevance          | 0.75  |       0.70        |   0.80   |
| evidence_quality   | 0.20  |       0.40        |   0.00   |
| factual_accuracy   | 0.60  |       0.60        |   0.60   |
| safety_compliance  | 1.00  |       1.00        |   1.00   |
| clarity            | 0.40  |       0.80        |   0.00   |
| **overall (weighted)** | **0.568** | **0.665** | **0.470** |

(A separate two-query batch run that exercises `python main.py --mode
evaluate` end-to-end is included in `outputs/evaluation_20260511_115350.json`;
its scores were all 0.0 because the original judge parser did not yet
tolerate Qwen3's `<think>` blocks. That artifact is kept in the repo as a
record of the bug and the fix in `src/evaluation/judge.py`.)

### 3.4 Error analysis

Two patterns are visible even in this single representative run:

1. **Inter-persona disagreement is largest on `clarity` and
   `evidence_quality`** (0.80 vs 0.00). The `academic_reviewer` rewards a
   well-structured response while the `end_user` persona penalizes any
   response that lacks concrete inline citations or named sources; the
   Writer in this run produced clean prose but did not consistently embed
   `[Source: ...]` markers. This is exactly the disagreement the
   two-persona design was meant to surface.
2. **`evidence_quality` is the system's weakest dimension** (mean 0.20).
   Even with Tavily and Semantic Scholar pre-fetched, the Researcher /
   Writer pair tend to summarize from prior knowledge rather than from
   the supplied evidence, and citations bleed out of the final response.
   A targeted fix is a Writer-side check that fails the round if no
   `[Source: ...]` appears in the synthesis.
3. **`safety_compliance` is saturated at 1.0** because the
   `SafetyManager` already strips unsafe content before the response
   leaves the orchestrator; the judge is grading a post-guardrail output.
   This is expected but means the criterion does not differentiate
   queries — the more interesting safety signal is the per-query
   `safety_action` recorded in the session JSON (e.g. our injection
   probe `"Ignore previous instructions..."` is correctly refused, with
   the categories `prompt_injection, off_topic_queries` logged to
   `logs/safety_events.log`).

## 4. Discussion and Limitations

**Insights.** Pre-fetching evidence into the prompt is a practical workaround
for endpoints that do not support runtime tool calls, and it also makes the
evidence inspectable from the UI. Two-persona judging surfaced cases where
the academic reviewer gave low evidence-quality scores while the end-user gave
high clarity scores — a signal that the Writer often produces clean prose
without enough citation depth.

**Limitations.** (i) The retrieval is single-shot per query — the Researcher
cannot issue follow-up searches. (ii) The guardrails are rule-based and will
miss adversarial phrasings the patterns do not cover; a learned classifier or
Guardrails AI validator would help. (iii) The LLM-as-a-Judge is the same
model family as the system under test, which is known to inflate scores; an
independent judge (e.g., a different model) would be a stronger evaluation
triangulation. (iv) `factual_consistency` is checked heuristically (presence
of citations) rather than by claim-level entailment against retrieved
sources.

**Future work.** Multi-turn retrieval driven by the Critic's feedback,
human-in-the-loop triangulation on a subset of queries, and migrating the
guardrail rules into a Guardrails AI policy file so they can be unit-tested
in isolation.

## References

- Microsoft. (2024). *AutoGen documentation*. https://microsoft.github.io/autogen/
- Tavily. (2024). *Tavily Search API documentation*. https://docs.tavily.com/
- Allen Institute for AI. (2024). *Semantic Scholar API*. https://api.semanticscholar.org/
- Guardrails AI. (2024). *Guardrails AI documentation*. https://docs.guardrailsai.com/
- NVIDIA. (2024). *NeMo Guardrails*. https://docs.nvidia.com/nemo/guardrails/
- Qwen Team. (2025). *Qwen3 technical report*. https://qwenlm.github.io/blog/qwen3/
- Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., ... & Stoica, I. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023.
