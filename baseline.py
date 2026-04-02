import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import TaskType, TriageAction, Severity, Team
from server.environment import BugTriageEnvironment


def run_baseline():
    """
    Run a baseline agent against all 3 tasks.
    Prints scores for each task and the average.
    
    Set OPENAI_API_KEY environment variable for LLM agent.
    Falls back to rule-based agent if no key is set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    results = {}

    print("\n=== Bug Triage Environment — Baseline Scores ===\n")

    for task in TaskType:
        print(f"Running task: {task.value}...")

        env = BugTriageEnvironment(task_type=task, max_steps=1)
        obs = env.reset()

        print(f"  Bug: [{obs.current_bug.id}] {obs.current_bug.title[:55]}...")

        # Get action from LLM or fallback
        if api_key:
            action = _llm_action(obs, api_key)
            agent  = "gpt-4o-mini"
        else:
            action = _rule_based_action(obs)
            agent  = "rule-based-fallback"

        result = env.step(action)

        print(f"  Agent: {agent}")
        print(f"  Reward: {result.reward:.3f}")
        print(f"  Feedback: {result.reward_breakdown.explanation}")
        print()

        results[task.value] = {
            "reward":  result.reward,
            "agent":   agent,
            "bug_id":  obs.current_bug.id,
        }

    # Summary
    average = round(
        sum(r["reward"] for r in results.values()) / len(results), 3
    )

    print(f"Average score across all 3 tasks: {average}\n")
    print("Full results:")
    print(json.dumps(results, indent=2))
    return results


def _rule_based_action(obs):
    """Simple rule-based agent — no API needed."""
    return TriageAction(
        severity="P1",
        duplicate_of="none",
        similarity_reasoning="No obvious match found in existing bugs.",
        assigned_team="backend",
        root_cause_hypothesis=(
            "Likely a backend validation or state management issue."
        ),
        reproduction_verified=True,
        priority_justification=(
            "Classified as P1 due to core feature impact with no workaround."
        ),
    )


def _llm_action(obs, api_key):
    """GPT-4o-mini agent."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    existing = ""
    if obs.existing_bugs:
        existing = "\n\nEXISTING BUGS:\n" + "\n".join(
            f"[{b.id}] {b.title}: {b.description[:80]}"
            for b in obs.existing_bugs
        )

    prompt = f"""You are a software engineer doing bug triage.

TASK: {obs.task_description}

BUG REPORT:
Title: {obs.current_bug.title}
Description: {obs.current_bug.description}
Steps: {'; '.join(obs.current_bug.steps_to_reproduce)}
Environment: {obs.current_bug.environment}
{existing}

Respond ONLY with valid JSON:
{{
  "severity": "P0/P1/P2/P3 or null",
  "duplicate_of": "bug-id or none or null",
  "similarity_reasoning": "string or null",
  "assigned_team": "backend/frontend/infra/security/data/mobile or null",
  "root_cause_hypothesis": "string or null",
  "reproduction_verified": true/false/null,
  "priority_justification": "string or null"
}}
Only fill fields relevant to the task. Set others to null."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=300,
    )

    data = json.loads(response.choices[0].message.content)
    clean = {k: v for k, v in data.items() if v is not None}
    return TriageAction(**clean)


if __name__ == "__main__":
    run_baseline()