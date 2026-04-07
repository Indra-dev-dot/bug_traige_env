import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from models import TriageAction, TriageObservation, TaskType, Severity, Team
from server.environment import BugTriageEnvironment

app = FastAPI(
    title="Bug Triage Environment",
    description="A real-world RL environment for training agents to triage software bug reports.",
    version="0.1.0",
)

# One global environment instance
env = BugTriageEnvironment()


# ── Core OpenEnv endpoints ────────────────────────────────────────────────────

@app.post("/reset")
def reset():
    """Start a new episode. Returns initial observation with a bug report."""
    obs = env.reset()
    return obs.model_dump()


@app.post("/step")
def step(action: TriageAction):
    """Submit a triage action. Returns observation + reward + done flag."""
    result = env.step(action)
    return result.model_dump()


@app.get("/state")
def state():
    """Return current episode metadata."""
    return env.state()


# ── Required hackathon endpoints ──────────────────────────────────────────────

@app.get("/tasks")
def get_tasks():
    """List all 3 tasks with descriptions and action schemas."""
    return JSONResponse({
        "tasks": [
            {
                "id": "severity_classification",
                "name": "Severity Classification",
                "difficulty": "easy",
                "description": (
                    "Given a bug report, classify its severity as "
                    "P0 (critical), P1 (major), P2 (minor), or P3 (trivial)."
                ),
                "required_action_fields": ["severity"],
                "action_schema": {
                    "severity": {
                        "type": "string",
                        "enum": ["P0", "P1", "P2", "P3"],
                    }
                },
                "scoring": "1.0 exact, 0.5 off-by-one, 0.2 off-by-two, 0.0 otherwise",
            },
            {
                "id": "duplicate_detection",
                "name": "Duplicate Detection",
                "difficulty": "medium",
                "description": (
                    "Given a new bug and a list of existing bugs, "
                    "determine if the new bug is a duplicate."
                ),
                "required_action_fields": ["duplicate_of", "similarity_reasoning"],
                "action_schema": {
                    "duplicate_of": {
                        "type": "string",
                        "description": "ID of duplicate bug or none",
                    },
                    "similarity_reasoning": {
                        "type": "string",
                        "description": "One sentence explaining your decision",
                    },
                },
                "scoring": "0.7 for correct ID + up to 0.3 for reasoning",
            },
            {
                "id": "full_triage",
                "name": "Full Triage Report",
                "difficulty": "hard",
                "description": (
                    "Perform complete triage: severity, team, "
                    "root cause, reproduction check, and justification."
                ),
                "required_action_fields": [
                    "severity",
                    "assigned_team",
                    "root_cause_hypothesis",
                    "reproduction_verified",
                    "priority_justification",
                ],
                "action_schema": {
                    "severity": {
                        "type": "string",
                        "enum": ["P0", "P1", "P2", "P3"],
                    },
                    "assigned_team": {
                        "type": "string",
                        "enum": ["backend", "frontend", "infra",
                                 "security", "data", "mobile"],
                    },
                    "root_cause_hypothesis": {"type": "string"},
                    "reproduction_verified": {"type": "boolean"},
                    "priority_justification": {"type": "string"},
                },
                "scoring": "Severity 30%, team 25%, root cause 25%, justification 20%",
            },
        ]
    })


@app.get("/grader")
def get_grader_score():
    """Return grader score for the most recently completed episode."""
    try:
        s = env.state()
        return JSONResponse({
            "episode_id":        s.get("episode_id", ""),
            "step_count":        s.get("step_count", 0),
            "cumulative_reward": s.get("cumulative_reward", 0.0),
            "task_type":         str(s.get("task_type", "")),
            "last_reward":       s.get("last_reward", 0.0),
            "last_feedback":     s.get("last_feedback", ""),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/baseline")
def run_baseline():
    """
    Run a baseline agent against all 3 tasks and return scores.
    Uses GPT-4o-mini if OPENAI_API_KEY is set, otherwise rule-based fallback.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    try:
        results = {}

        for task in TaskType:
            task_env = BugTriageEnvironment(task_type=task, max_steps=1)
            obs      = task_env.reset()
            action   = _get_agent_action(obs, api_key)
            result   = task_env.step(action)

            results[task.value] = {
                "reward":    result.reward,
                "feedback":  result.reward_breakdown.explanation,
                "bug_id":    obs.current_bug.id,
                "agent":     "gpt-4o-mini" if api_key else "rule-based",
            }

        average = round(
            sum(r["reward"] for r in results.values()) / len(results), 3
        )

        return JSONResponse({
            "status":           "success",
            "baseline_scores":  results,
            "average_score":    average,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "environment": "bug-triage", "version": "0.1.0"}


# ── Agent helpers ─────────────────────────────────────────────────────────────

def _get_agent_action(obs, api_key):
    """Use LLM agent if API key available, otherwise rule-based."""
    if api_key:
        return _llm_agent_action(obs, api_key)
    return _rule_based_action(obs)


def _llm_agent_action(obs, api_key):
    """Call GPT-4o-mini to produce a triage action."""
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


def _rule_based_action(obs):
    """Simple fallback agent — no API needed."""
    return TriageAction(
        severity="P1",
        duplicate_of="none",
        similarity_reasoning="No obvious match found in existing bugs.",
        assigned_team="backend",
        root_cause_hypothesis="Likely a backend validation or state management issue.",
        reproduction_verified=True,
        priority_justification=(
            "Classified as P1 due to core feature impact with no workaround."
        ),
    )

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()