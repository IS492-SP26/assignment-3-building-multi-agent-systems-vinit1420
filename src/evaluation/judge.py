"""
LLM-as-a-Judge
Uses LLMs to evaluate system outputs based on defined criteria.

Example usage:
    # Initialize judge with config
    judge = LLMJudge(config)
    
    # Evaluate a response
    result = await judge.evaluate(
        query="What is the capital of France?",
        response="Paris is the capital of France.",
        sources=[],
        ground_truth="Paris"
    )
    
    print(f"Overall Score: {result['overall_score']}")
    print(f"Criterion Scores: {result['criterion_scores']}")
"""

from typing import Dict, Any, List, Optional
import logging
import json
import os


class LLMJudge:
    """
    LLM-based judge for evaluating system responses.

    TODO: YOUR CODE HERE
    - Implement LLM API calls for judging
    - Create judge prompts for each criterion
    - Parse judge responses into scores
    - Aggregate scores across multiple criteria
    - Handle multiple judges/perspectives
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM judge.

        Args:
            config: Configuration dictionary (from config.yaml)
        """
        self.config = config
        self.logger = logging.getLogger("evaluation.judge")

        # Load judge model configuration from config.yaml (models.judge)
        self.model_config = config.get("models", {}).get("judge", {})

        # Load evaluation criteria from config.yaml (evaluation.criteria)
        self.criteria = config.get("evaluation", {}).get("criteria", [])

        # Two independent judging perspectives so we satisfy "2+ judge prompts".
        # The first plays a strict academic reviewer; the second plays an end-user
        # focused on practical clarity. Each scores the *same* criteria.
        self.judge_personas = config.get("evaluation", {}).get("personas", [
            {
                "name": "academic_reviewer",
                "system": (
                    "You are a meticulous academic peer reviewer for an HCI venue. "
                    "Be strict and grounded. Penalize unsupported claims and missing citations. "
                    "Always respond in valid JSON."
                ),
            },
            {
                "name": "end_user",
                "system": (
                    "You are an end-user reading a research summary. You care about clarity, "
                    "concrete examples, and whether the answer addresses your question. "
                    "Always respond in valid JSON."
                ),
            },
        ])

        # Provider-agnostic client setup.
        self.provider = self.model_config.get("provider", "groq")
        self._client = None
        self._init_client()

        self.logger.info(
            f"LLMJudge initialized provider={self.provider} criteria={len(self.criteria)} "
            f"personas={[p['name'] for p in self.judge_personas]}"
        )

    def _init_client(self):
        """Initialize an OpenAI-compatible client for the configured provider."""
        try:
            if self.provider == "groq":
                from groq import Groq
                key = os.getenv("GROQ_API_KEY")
                if not key:
                    self.logger.warning("GROQ_API_KEY not set; judge will fail at call time")
                    return
                self._client = Groq(api_key=key)
            else:  # openai or vllm (both use OpenAI-compatible API)
                from openai import OpenAI
                key = os.getenv("OPENAI_API_KEY")
                base_url = os.getenv("OPENAI_BASE_URL") or None
                if not key:
                    self.logger.warning("OPENAI_API_KEY not set; judge will fail at call time")
                    return
                self._client = OpenAI(api_key=key, base_url=base_url)
        except Exception as e:
            self.logger.error(f"Failed to initialize judge client: {e}")
 
    async def evaluate(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a response using LLM-as-a-Judge.

        Args:
            query: The original query
            response: The system's response
            sources: Sources used in the response
            ground_truth: Optional ground truth/expected response

        Returns:
            Dictionary with scores for each criterion and overall score

        TODO: YOUR CODE HERE
        - Implement LLM API calls
        - Call judge for each criterion
        - Parse and aggregate scores
        - Provide detailed feedback
        """
        self.logger.info(f"Evaluating response for query: {query[:50]}...")

        results = {
            "query": query,
            "overall_score": 0.0,
            "criterion_scores": {},
            "per_persona_scores": {},
            "feedback": [],
        }

        total_weight = sum(c.get("weight", 1.0) for c in self.criteria) or 1.0

        # Evaluate each criterion under each independent judge persona.
        for criterion in self.criteria:
            criterion_name = criterion.get("name", "unknown")
            persona_scores = {}
            for persona in self.judge_personas:
                self.logger.info(f"Judging {criterion_name} as {persona['name']}")
                score = await self._judge_criterion(
                    criterion=criterion,
                    query=query,
                    response=response,
                    sources=sources,
                    ground_truth=ground_truth,
                    persona=persona,
                )
                persona_scores[persona["name"]] = score

            # Average across personas for this criterion.
            mean_score = sum(s.get("score", 0.0) for s in persona_scores.values()) / max(len(persona_scores), 1)
            results["criterion_scores"][criterion_name] = {
                "score": mean_score,
                "by_persona": persona_scores,
                "criterion": criterion_name,
            }

        # Per-persona overall (weighted) so the report can show disagreement.
        for persona in self.judge_personas:
            pname = persona["name"]
            ws = 0.0
            for criterion in self.criteria:
                w = criterion.get("weight", 1.0)
                c_data = results["criterion_scores"][criterion["name"]]
                ws += c_data["by_persona"][pname].get("score", 0.0) * w
            results["per_persona_scores"][pname] = ws / total_weight

        # Final aggregate = mean of per-persona weighted overall scores.
        overall_values = list(results["per_persona_scores"].values())
        results["overall_score"] = sum(overall_values) / len(overall_values) if overall_values else 0.0
        return results

    async def _judge_criterion(
        self,
        criterion: Dict[str, Any],
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str],
        persona: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Judge a single criterion.

        Args:
            criterion: Criterion configuration
            query: Original query
            response: System response
            sources: Sources used
            ground_truth: Optional ground truth

        Returns:
            Score and feedback for this criterion

        This is a basic implementation using Groq API.
        """
        criterion_name = criterion.get("name", "unknown")
        description = criterion.get("description", "")

        # Create judge prompt
        prompt = self._create_judge_prompt(
            criterion_name=criterion_name,
            description=description,
            query=query,
            response=response,
            sources=sources,
            ground_truth=ground_truth
        )

        # Call LLM API to get judgment
        try:
            judgment = await self._call_judge_llm(prompt, persona=persona)
            score_value, reasoning = self._parse_judgment(judgment)

            score = {
                "score": score_value,  # 0-1 scale
                "reasoning": reasoning,
                "criterion": criterion_name,
                # Save the raw prompt and raw model output so a grader can
                # inspect exactly what was sent and returned.
                "prompt": prompt,
                "raw_output": judgment,
                "persona_system": persona["system"] if persona else None,
            }
        except Exception as e:
            self.logger.error(f"Error judging criterion {criterion_name}: {e}")
            score = {
                "score": 0.0,
                "reasoning": f"Error during evaluation: {str(e)}",
                "criterion": criterion_name
            }

        return score

    def _create_judge_prompt(
        self,
        criterion_name: str,
        description: str,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str]
    ) -> str:
        """
        Create a prompt for the judge LLM.

        TODO: YOUR CODE HERE
        - Create effective judge prompts
        - Include clear scoring rubric
        - Provide examples if helpful
        """
        prompt = f"""You are an expert evaluator. Evaluate the following response based on the criterion: {criterion_name}.

Criterion Description: {description}

Query: {query}

Response:
{response}
"""

        if sources:
            prompt += f"\n\nSources Used: {len(sources)} sources"

        if ground_truth:
            prompt += f"\n\nExpected Response:\n{ground_truth}"

        prompt += """

Please evaluate the response on a scale of 0.0 to 1.0 for this criterion.
Provide your evaluation in the following JSON format:
{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<detailed explanation of your score>"
}
"""

        return prompt

    async def _call_judge_llm(self, prompt: str, persona: Optional[Dict[str, Any]] = None) -> str:
        """Call configured judge LLM (Groq or OpenAI/vLLM compatible)."""
        if not self._client:
            raise ValueError(
                f"Judge client not initialized for provider={self.provider}. "
                "Check GROQ_API_KEY or OPENAI_API_KEY in environment."
            )

        model_name = self.model_config.get("name", "llama-3.1-8b-instant")
        temperature = self.model_config.get("temperature", 0.3)
        max_tokens = self.model_config.get("max_tokens", 1024)
        system_msg = (
            persona["system"] if persona and "system" in persona
            else "You are an expert evaluator. Provide your evaluations in valid JSON format."
        )

        try:
            chat_completion = self._client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error calling judge LLM ({self.provider}): {e}")
            raise

    def _parse_judgment(self, judgment: str) -> tuple:
        """Parse LLM judgment, tolerating <think> blocks and surrounding prose."""
        import re
        try:
            judgment_clean = judgment.strip()
            # Strip Qwen3-style chain-of-thought
            judgment_clean = re.sub(r"<think>.*?</think>", "", judgment_clean, flags=re.S).strip()
            # Strip markdown code fences
            judgment_clean = re.sub(r"^```(?:json)?\s*", "", judgment_clean)
            judgment_clean = re.sub(r"\s*```$", "", judgment_clean).strip()
            # Extract the first {...} JSON object if extra prose is present.
            if not judgment_clean.startswith("{"):
                m = re.search(r"\{.*\}", judgment_clean, flags=re.S)
                if m:
                    judgment_clean = m.group(0)

            result = json.loads(judgment_clean)
            score = float(result.get("score", 0.0))
            reasoning = result.get("reasoning", "")
            
            # Validate score is in range [0, 1]
            score = max(0.0, min(1.0, score))
            
            return score, reasoning
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            self.logger.error(f"Raw judgment: {judgment[:200]}")
            return 0.0, f"Error parsing judgment: Invalid JSON"
        except Exception as e:
            self.logger.error(f"Error parsing judgment: {e}")
            return 0.0, f"Error parsing judgment: {str(e)}"



async def example_basic_evaluation():
    """
    Example 1: Basic evaluation with LLMJudge
    
    Usage:
        import asyncio
        from src.evaluation.judge import example_basic_evaluation
        asyncio.run(example_basic_evaluation())
    """
    import yaml
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize judge
    judge = LLMJudge(config)
    
    # Test case (similar to Lab 5)
    print("=" * 70)
    print("EXAMPLE 1: Basic Evaluation")
    print("=" * 70)
    
    query = "What is the capital of France?"
    response = "Paris is the capital of France. It is known for the Eiffel Tower."
    ground_truth = "Paris"
    
    print(f"\nQuery: {query}")
    print(f"Response: {response}")
    print(f"Ground Truth: {ground_truth}\n")
    
    # Evaluate
    result = await judge.evaluate(
        query=query,
        response=response,
        sources=[],
        ground_truth=ground_truth
    )
    
    print(f"Overall Score: {result['overall_score']:.3f}\n")
    print("Criterion Scores:")
    for criterion, score_data in result['criterion_scores'].items():
        print(f"  {criterion}: {score_data['score']:.3f}")
        print(f"    Reasoning: {score_data['reasoning'][:100]}...")
        print()


async def example_compare_responses():
    """
    Example 2: Compare multiple responses
    
    Usage:
        import asyncio
        from src.evaluation.judge import example_compare_responses
        asyncio.run(example_compare_responses())
    """
    import yaml
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize judge
    judge = LLMJudge(config)
    
    print("=" * 70)
    print("EXAMPLE 2: Compare Multiple Responses")
    print("=" * 70)
    
    query = "What causes climate change?"
    ground_truth = "Climate change is primarily caused by increased greenhouse gas emissions from human activities, including burning fossil fuels, deforestation, and industrial processes."
    
    responses = [
        "Climate change is primarily caused by greenhouse gas emissions from human activities.",
        "The weather changes because of natural cycles and the sun's activity.",
        "Climate change is a complex phenomenon involving multiple factors including CO2 emissions, deforestation, and industrial processes."
    ]
    
    print(f"\nQuery: {query}\n")
    print(f"Ground Truth: {ground_truth}\n")
    
    results = []
    for i, response in enumerate(responses, 1):
        print(f"\n{'='*70}")
        print(f"Response {i}:")
        print(f"{response}")
        print(f"{'='*70}")
        
        result = await judge.evaluate(
            query=query,
            response=response,
            sources=[],
            ground_truth=ground_truth
        )
        
        results.append(result)
        
        print(f"\nOverall Score: {result['overall_score']:.3f}")
        print("\nCriterion Scores:")
        for criterion, score_data in result['criterion_scores'].items():
            print(f"  {criterion}: {score_data['score']:.3f}")
        print()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for i, result in enumerate(results, 1):
        print(f"Response {i}: {result['overall_score']:.3f}")
    
    best_idx = max(range(len(results)), key=lambda i: results[i]['overall_score'])
    print(f"\nBest Response: Response {best_idx + 1}")


# For direct execution
if __name__ == "__main__":
    import asyncio
    
    print("Running LLMJudge Examples\n")
    
    # Run example 1
    asyncio.run(example_basic_evaluation())
    
    print("\n\n")
    
    # Run example 2
    asyncio.run(example_compare_responses())
