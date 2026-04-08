import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from models import TriageAction, TriageObservation, TaskType, Severity, Team
from server.environment import BugTriageEnvironment

app = FastAPI(
    title="Bug Triage Environment",
    description="A real-world RL environment for training agents to triage software bug reports.",
    version="0.1.0",
)

env = BugTriageEnvironment()


# ── Core OpenEnv endpoints ────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Bug Triage Environment",
        "version": "0.1.0",
        "status": "running",
        "endpoints": ["/reset", "/step", "/state", "/health", "/metadata", "/schema", "/tasks", "/grader", "/baseline"],
    }


@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.model_dump()


@app.post("/step")
def step(action: TriageAction):
    result = env.step(action)
    return result.model_dump()


@app.get("/state")
def state():
    return env.state()


# ── OpenEnv validate required endpoints ──────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy", "environment": "bug-triage", "version": "0.1.0"}


@app.get("/metadata")
def metadata():
    return {
        "name": "bug-triage-env",
        "description": "A real-world RL environment where agents learn to triage software bug reports.",
        "version": "0.1.0",
        "author": "HadesnApollo",
        "tags": ["bug-triage", "software-engineering", "real-world", "nlp"],
    }


@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "severity": {"type": "string", "enum": ["P0", "P1", "P2", "P3"]},
                "duplicate_of": {"type": "string"},
                "similarity_reasoning": {"type": "string"},
                "assigned_team": {"type": "string"},
                "root_cause_hypothesis": {"type": "string"},
                "reproduction_verified": {"type": "boolean"},
                "priority_justification": {"type": "string"},
            }
        },
        "observation": {
            "type": "object",
            "properties": {
                "task_type": {"type": "string"},
                "current_bug": {"type": "object"},
                "existing_bugs": {"type": "array"},
                "last_reward": {"type": "number"},
                "last_feedback": {"type": "string"},
                "step_number": {"type": "integer"},
                "max_steps": {"type": "integer"},
                "task_description": {"type": "string"},
            }
        },
        "state": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "step_count": {"type": "integer"},
                "task_type": {"type": "string"},
                "bug_id": {"type": "string"},
                "cumulative_reward": {"type": "number"},
            }
        }
    }


@app.post("/mcp")
async def mcp(request: Request):
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "name": "bug-triage-env",
            "version": "0.1.0",
            "description": "Bug triage RL environment",
        }
    }


# ── Hackathon required endpoints ──────────────────────────────────────────────

@app.get("/tasks")
def get_tasks():
    return JSONResponse({
        "tasks": [
            {
                "id": "severity_classification",
                "name": "Severity Classification",
                "difficulty": "easy",
                "description": "Given a bug report, classify its severity as P0 (critical), P1 (major), P2 (minor), or P3 (trivial).",
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
                "description": "Given a new bug and a list of existing bugs, determine if the new bug is a duplicate.",
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
                "description": "Perform complete triage: severity, team, root cause, reproduction check, and justification.",
                "required_action_fields": [
                    "severity",
                    "assigned_team",
                    "root_cause_hypothesis",
                    "reproduction_verified",
                    "priority_justification",
                ],
                "action_schema": {
                    "severity": {"type": "string", "enum": ["P0", "P1", "P2", "P3"]},
                    "assigned_team": {
                        "type": "string",
                        "enum": ["backend", "frontend", "infra", "security", "data", "mobile"],
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
    api_key = os.getenv("OPENAI_API_KEY")
    try:
        results = {}
        for task in TaskType:
            task_env = BugTriageEnvironment(task_type=task, max_steps=1)
            obs      = task_env.reset()
            action   = _get_agent_action(obs, api_key)
            result   = task_env.step(action)
            results[task.value] = {
                "reward":   result.reward,
                "feedback": result.reward_breakdown.explanation,
                "bug_id":   obs.current_bug.id,
                "agent":    "gpt-4o-mini" if api_key else "rule-based",
            }
        average = round(
            sum(r["reward"] for r in results.values()) / len(results), 3
        )
        return JSONResponse({
            "status":          "success",
            "baseline_scores": results,
            "average_score":   average,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Agent helpers ─────────────────────────────────────────────────────────────

def _get_agent_action(obs, api_key):
    if api_key:
        return _llm_agent_action(obs, api_key)
    return _rule_based_action(obs)


def _llm_agent_action(obs, api_key):
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
}}"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=300,
    )
    data = json.loads(response.choices[0].message.content)
    return TriageAction(**{k: v for k, v in data.items() if v is not None})


def _rule_based_action(obs):
    return TriageAction(
        severity="P1",
        duplicate_of="none",
        similarity_reasoning="No obvious match found in existing bugs.",
        assigned_team="backend",
        root_cause_hypothesis="Likely a backend validation or state management issue.",
        reproduction_verified=True,
        priority_justification="Classified as P1 due to core feature impact with no workaround.",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()