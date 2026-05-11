"""
Main Entry Point
Can be used to run the system or evaluation.

Usage:
  python main.py --mode cli           # Run CLI interface
  python main.py --mode web           # Run web interface
  python main.py --mode evaluate      # Run evaluation
"""

import argparse
import asyncio
import sys
from pathlib import Path


def run_cli():
    """Run CLI interface."""
    from src.ui.cli import main as cli_main
    cli_main()


def run_web():
    """Run web interface."""
    import subprocess
    print("Starting Streamlit web interface...")
    subprocess.run(["streamlit", "run", "src/ui/streamlit_app.py"])


async def run_evaluation(queries_path: str = "data/example_queries.json"):
    """Run full batch evaluation through SystemEvaluator + LLM-as-Judge."""
    import yaml
    from dotenv import load_dotenv
    from src.autogen_orchestrator import AutoGenOrchestrator
    from src.evaluation.evaluator import SystemEvaluator

    load_dotenv()
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Initializing AutoGen orchestrator...")
    orchestrator = AutoGenOrchestrator(config)

    print("Initializing evaluator...")
    evaluator = SystemEvaluator(config, orchestrator=orchestrator)

    print("\n" + "=" * 70)
    print(f"RUNNING BATCH EVALUATION ON {queries_path}")
    print("=" * 70)

    report = await evaluator.evaluate_system(queries_path)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    summary = report.get("summary", {})
    scores = report.get("scores", {})
    print(f"Total queries:   {summary.get('total_queries', 0)}")
    print(f"Successful:      {summary.get('successful', 0)}")
    print(f"Failed:          {summary.get('failed', 0)}")
    print(f"Overall avg:     {scores.get('overall_average', 0.0):.3f}\n")
    print("Per-criterion average scores:")
    for crit, val in scores.get("by_criterion", {}).items():
        print(f"  {crit:<22} {val:.3f}")
    print("\nFull report written to outputs/")


def run_autogen():
    """Run AutoGen example."""
    import subprocess
    print("Running AutoGen example...")
    subprocess.run([sys.executable, "example_autogen.py"])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant"
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "web", "evaluate", "autogen"],
        default="autogen",
        help="Mode to run: cli, web, evaluate, or autogen (default)"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli()
    elif args.mode == "web":
        run_web()
    elif args.mode == "evaluate":
        asyncio.run(run_evaluation())
    elif args.mode == "autogen":
        run_autogen()


if __name__ == "__main__":
    main()
