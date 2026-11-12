"""Command-line front end for the VibeMatch Agent.

    python -m src.agent_cli "rainy sunday study, no lyrics, 40 minutes"
    python -m src.agent_cli "something sad" --no-rag      # baseline without retrieval
    python -m src.agent_cli "gym, warm up then go hard" --trace
"""
import argparse
import json
import sys

from src.agent import VibeMatchAgent, configure_logging
from src.recommender import MODES


def render(result) -> str:
    lines = [f"Status: {result.status}   LLM: {result.llm}"]
    if result.message:
        lines.append(result.message)
    if result.items:
        lines.append(f"\n{result.intro}\n")
        for it in result.items:
            s = it.song
            lines.append(f"{it.slot:>2}. {s['title']} - {s['artist']}  [{s['genre']}/{s['mood']}]  "
                         f"energy {s['energy']:.2f}  {s['tempo_bpm']} bpm  ({it.segment})  score {it.score:.2f}")
            lines.append(f"      why: {'; '.join(it.reasons)}")
            if it.citations:
                lines.append(f"      sources: {', '.join(it.citations)}")
        lines.append(f"\nTotal: {result.total_minutes} min   Confidence: {result.confidence:.2f}"
                     f"{'   NEEDS REVIEW' if result.needs_review else ''}   Repair rounds: {result.rounds}")
        lines.append("Checks: " + ", ".join(f"{c.name}={'pass' if c.passed else 'FAIL'}" for c in result.checks))
    for w in result.warnings:
        lines.append(f"note: {w}")
    if result.trace_path:
        lines.append(f"trace: {result.trace_path}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="VibeMatch Agent: natural-language playlist builder")
    p.add_argument("request", help="what you want to listen to")
    p.add_argument("--mode", choices=list(MODES), default="balanced")
    p.add_argument("--no-rag", action="store_true", help="disable retrieval (baseline)")
    p.add_argument("--zero-shot", action="store_true", help="LLM prompt without few-shot examples")
    p.add_argument("--trace", action="store_true", help="print the step-by-step trace as JSON")
    p.add_argument("--save-trace", action="store_true", help="write the trace to logs/traces/")
    args = p.parse_args(argv)

    configure_logging()
    agent = VibeMatchAgent(mode=args.mode, use_rag=not args.no_rag, few_shot=not args.zero_shot,
                           save_trace=args.save_trace)
    result = agent.run(args.request)
    print(render(result))
    if args.trace:
        print(json.dumps(result.trace, indent=2, default=str))
    return 0 if result.status in ("ok", "needs_clarification") else 1


if __name__ == "__main__":
    sys.exit(main())
