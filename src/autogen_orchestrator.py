"""
AutoGen-Based Orchestrator

This orchestrator uses AutoGen's RoundRobinGroupChat to coordinate multiple agents
in a research workflow.

Workflow:
1. Planner: Breaks down the query into research steps
2. Researcher: Gathers evidence using web and paper search tools
3. Writer: Synthesizes findings into a coherent response
4. Critic: Evaluates quality and provides feedback
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional

from src.agents.autogen_agents import create_research_team
from src.guardrails.safety_manager import SafetyManager
from src.tools.web_search import WebSearchTool
from src.tools.paper_search import PaperSearchTool


class AutoGenOrchestrator:
    """
    Orchestrates multi-agent research using AutoGen's RoundRobinGroupChat.
    
    This orchestrator manages a team of specialized agents that work together
    to answer research queries. It uses AutoGen's built-in conversation
    management and tool execution capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AutoGen orchestrator.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.logger = logging.getLogger("autogen_orchestrator")
        
        # Safety manager runs input/output guardrails around the team.
        self.safety_manager = SafetyManager(config)

        # Create the research team
        self.logger.info("Creating research team...")
        self.team = create_research_team(config)

        self.logger.info("Research team created successfully")

        # Workflow trace for debugging and UI display
        self.workflow_trace: List[Dict[str, Any]] = []

    def process_query(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        """
        Process a research query through the multi-agent system.

        Args:
            query: The research question to answer
            max_rounds: Maximum number of conversation rounds

        Returns:
            Dictionary containing:
            - query: Original query
            - response: Final synthesized response
            - conversation_history: Full conversation between agents
            - metadata: Additional information about the process
        """
        self.logger.info(f"Processing query: {query}")

        # ----- Input guardrail -----
        input_check = self.safety_manager.check_input_safety(query)
        safety_events: List[Dict[str, Any]] = []
        if input_check.get("event"):
            safety_events.append(input_check["event"])
        if not input_check["safe"]:
            refusal = self.safety_manager.on_violation.get(
                "message",
                "I cannot process this request due to safety policies.",
            )
            self.logger.warning("Input refused by guardrail")
            return {
                "query": query,
                "response": refusal,
                "conversation_history": [],
                "metadata": {
                    "num_messages": 0,
                    "num_sources": 0,
                    "agents_involved": [],
                    "safety_events": safety_events,
                    "safety_action": input_check["action"],
                    "refused": True,
                },
            }

        try:
            # Run the async query processing
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run, 
                        self._process_query_async(query, max_rounds)
                    ).result()
            else:
                result = loop.run_until_complete(self._process_query_async(query, max_rounds))

            # ----- Output guardrail -----
            output_check = self.safety_manager.check_output_safety(
                result.get("response", ""),
                sources=result.get("metadata", {}).get("sources", []),
            )
            if output_check.get("event"):
                safety_events.append(output_check["event"])
            result["response"] = output_check["response"]
            md = result.setdefault("metadata", {})
            md["safety_events"] = safety_events
            md["safety_action"] = output_check["action"]
            md["refused"] = not output_check["safe"] and output_check["action"] == "refuse"

            self.logger.info("Query processing complete")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "response": f"An error occurred while processing your query: {str(e)}",
                "conversation_history": [],
                "metadata": {"error": True}
            }
    
    async def _process_query_async(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        """
        Async implementation of query processing.
        
        Args:
            query: The research question to answer
            max_rounds: Maximum number of conversation rounds
            
        Returns:
            Dictionary containing results
        """
        # Pre-fetch evidence (tools must be invoked here because the class
        # vLLM endpoint does not support automatic tool calling). We await
        # the async tools directly to avoid nested asyncio.run().
        tools_cfg = self.config.get("tools", {})
        web_max = tools_cfg.get("web_search", {}).get("max_results", 5)
        paper_max = tools_cfg.get("paper_search", {}).get("max_results", 5)
        web_provider = tools_cfg.get("web_search", {}).get("provider", "tavily")

        # Cap snippet/abstract length so the cumulative context stays under
        # the model's window (Qwen3-8B = 40,960 tokens).
        SNIPPET_MAX = 350
        ABSTRACT_MAX = 280

        web_evidence = "No web search results found."
        try:
            wtool = WebSearchTool(provider=web_provider, max_results=web_max)
            web_results = await wtool.search(query)
            if web_results:
                lines = [f"Found {len(web_results)} web results for '{query}':\n"]
                for i, r in enumerate(web_results, 1):
                    snippet = (r.get("snippet","") or "")[:SNIPPET_MAX]
                    lines.append(f"{i}. {r.get('title','')}\n   URL: {r.get('url','')}\n   {snippet}\n")
                web_evidence = "\n".join(lines)
        except Exception as e:
            self.logger.warning(f"web_search failed: {e}")

        paper_evidence = "No academic papers found."
        try:
            ptool = PaperSearchTool(max_results=paper_max)
            papers = await ptool.search(query)
            if papers:
                lines = [f"Found {len(papers)} academic papers for '{query}':\n"]
                for i, p in enumerate(papers, 1):
                    authors = ", ".join(a.get("name","") for a in p.get("authors", [])[:3])
                    abstract = (p.get("abstract","") or "")[:ABSTRACT_MAX]
                    lines.append(
                        f"{i}. {p.get('title','')}\n"
                        f"   Authors: {authors}\n"
                        f"   Year: {p.get('year','?')} | Citations: {p.get('citation_count',0)}\n"
                        f"   Abstract: {abstract}\n"
                        f"   URL: {p.get('url','')}\n"
                    )
                paper_evidence = "\n".join(lines)
        except Exception as e:
            self.logger.warning(f"paper_search failed: {e}")

        task_message = f"""Research Query: {query}

Pre-fetched WEB EVIDENCE (use these citations in your answer):
{web_evidence}

Pre-fetched ACADEMIC EVIDENCE (use these citations in your answer):
{paper_evidence}

Workflow:
1. Planner: outline an approach for synthesizing the evidence above.
2. Researcher: pick the most relevant items from the evidence and summarize key findings with URLs.
3. Writer: produce a structured response with inline citations [Source: Title or URL] and a References section.
4. Critic: evaluate completeness and citation quality. Conclude with the approval token when satisfied."""
        
        # Run the team
        result = await self.team.run(task=task_message)
        
        # Extract conversation history (result.messages is a regular list in
        # autogen-agentchat 0.7).
        messages = []
        for message in result.messages:
            msg_dict = {
                "source": getattr(message, "source", "Unknown"),
                "content": getattr(message, "content", str(message)),
            }
            messages.append(msg_dict)
        
        # Extract final response: prefer the Writer's last message (the
        # synthesized answer). Fall back to the Critic, then to the last
        # message overall.
        import re
        def _clean(text: str) -> str:
            # Strip Qwen3 chain-of-thought blocks for user-facing output.
            text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
            return text.strip()

        final_response = ""
        for preferred in ("Writer", "Critic"):
            for msg in reversed(messages):
                if msg.get("source") == preferred:
                    final_response = _clean(msg.get("content", ""))
                    if final_response:
                        break
            if final_response:
                break
        if not final_response and messages:
            final_response = _clean(messages[-1].get("content", ""))
        
        return self._extract_results(query, messages, final_response)

    def _extract_results(self, query: str, messages: List[Dict[str, Any]], final_response: str = "") -> Dict[str, Any]:
        """
        Extract structured results from the conversation history.

        Args:
            query: Original query
            messages: List of conversation messages
            final_response: Final response from the team

        Returns:
            Structured result dictionary
        """
        # Extract components from conversation
        research_findings = []
        plan = ""
        critique = ""
        
        for msg in messages:
            source = msg.get("source", "")
            content = msg.get("content", "")
            
            if source == "Planner" and not plan:
                plan = content
            
            elif source == "Researcher":
                research_findings.append(content)
            
            elif source == "Critic":
                critique = content
        
        # Count sources mentioned in research
        num_sources = 0
        for finding in research_findings:
            # Rough count of sources based on numbered results
            num_sources += finding.count("\n1.") + finding.count("\n2.") + finding.count("\n3.")
        
        # Clean up final response
        if final_response:
            for token in ("APPROVED-RESEARCH-COMPLETE", "TERMINATE"):
                final_response = final_response.replace(token, "")
            final_response = final_response.strip()
        
        return {
            "query": query,
            "response": final_response,
            "conversation_history": messages,
            "metadata": {
                "num_messages": len(messages),
                "num_sources": max(num_sources, 1),  # At least 1
                "plan": plan,
                "research_findings": research_findings,
                "critique": critique,
                "agents_involved": list(set([msg.get("source", "") for msg in messages])),
            }
        }

    def get_agent_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all agents.

        Returns:
            Dictionary mapping agent names to their descriptions
        """
        return {
            "Planner": "Breaks down research queries into actionable steps",
            "Researcher": "Gathers evidence from web and academic sources",
            "Writer": "Synthesizes findings into coherent responses",
            "Critic": "Evaluates quality and provides feedback",
        }

    def visualize_workflow(self) -> str:
        """
        Generate a text visualization of the workflow.

        Returns:
            String representation of the workflow
        """
        workflow = """
AutoGen Research Workflow:

1. User Query
   ↓
2. Planner
   - Analyzes query
   - Creates research plan
   - Identifies key topics
   ↓
3. Researcher (with tools)
   - Uses web_search() tool
   - Uses paper_search() tool
   - Gathers evidence
   - Collects citations
   ↓
4. Writer
   - Synthesizes findings
   - Creates structured response
   - Adds citations
   ↓
5. Critic
   - Evaluates quality
   - Checks completeness
   - Provides feedback
   ↓
6. Decision Point
   - If APPROVED → Final Response
   - If NEEDS REVISION → Back to Writer
        """
        return workflow


def demonstrate_usage():
    """
    Demonstrate how to use the AutoGen orchestrator.
    
    This function shows a simple example of using the orchestrator.
    """
    import yaml
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create orchestrator
    orchestrator = AutoGenOrchestrator(config)
    
    # Print workflow visualization
    print(orchestrator.visualize_workflow())
    
    # Example query
    query = "What are the latest trends in human-computer interaction research?"
    
    print(f"\nProcessing query: {query}\n")
    print("=" * 70)
    
    # Process query
    result = orchestrator.process_query(query)
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nQuery: {result['query']}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nMetadata:")
    print(f"  - Messages exchanged: {result['metadata']['num_messages']}")
    print(f"  - Sources gathered: {result['metadata']['num_sources']}")
    print(f"  - Agents involved: {', '.join(result['metadata']['agents_involved'])}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    demonstrate_usage()

