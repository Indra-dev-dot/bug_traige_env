"""
inference.py — Bug Triage Environment Inference Script
Follows the mandatory stdout format: [START], [STEP], [END]
"""

import os
import json
from typing import List, Optional
from openai import OpenAI

# ── Environment variables ─────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "https://HadesnApollo-bug-triage-env.hf.space")
BENCHMARK    = "bug-triage"
MAX_STEPS    = 3
SUCCESS_SCORE_THRESHOLD = 0.5

# ── Mandatory log functions ───────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── HTTP helpers (no openenv-core needed) ─────────────────────────────────────

import urllib.request

def http_get(url: str) -> dict:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def http_post(url: str, data: dict = None) -> dict:
    body = json.dumps(data or {}).encode()
    req  = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


# ── LLM agent ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert software engineer performing bug triage.
You will receive a bug report and must classify it.
Always respond with valid JSON only — no explanation, no markdown."""


def get_llm_action(client: OpenAI, obs: dict) -> dict:
    """Ask the LLM to triage the bug report."""
    bug    = obs["current_bug"]
    task   = obs["task_type"]
    desc   = obs["task_description"]

    existing = ""
    if obs.get("existing_bugs"):
        existing = "\n\nEXISTING BUGS:\n" + "\n".join(
            f"[{b['id']}] {b['title']}: {b['description'][:80]}"
            for b in obs["existing_bugs"]
        )

    user_prompt = f"""TASK: {desc}

BUG REPORT:
ID: {bug['id']}
Title: {bug['title']}
Description: {bug['description']}
Steps: {'; '.join(bug['steps_to_reproduce'])}
Environment: {bug['environment']}
{existing}

Last feedback: {obs.get('last_feedback', 'none')}

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

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=300,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Clean markdown if present
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        # Remove null values
        return {k: v for k, v in data.items() if v is not None}
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        # Fallback rule-based action
        return {
            "severity": "P1",
            "duplicate_of": "none",
            "similarity_reasoning": "No obvious match found.",
            "assigned_team": "backend",
            "root_cause_hypothesis": "Likely a backend validation issue.",
            "reproduction_verified": True,
            "priority_justification": "Classified as P1 due to core feature impact.",
        }


# ── Run one episode ───────────────────────────────────────────────────────────

def run_episode(client: OpenAI, task_name: str) -> dict:
    """Run one full episode for a given task."""
    rewards     = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        obs = http_post(f"{ENV_URL}/reset")

        for step in range(1, MAX_STEPS + 1):
            # Get LLM action
            action = get_llm_action(client, obs)
            action_str = json.dumps(action, separators=(',', ':'))

            # Step environment
            try:
                result = http_post(f"{ENV_URL}/step", action)
                reward = float(result.get("reward", 0.0))
                done   = bool(result.get("done", False))
                error  = None
                obs    = result.get("observation", obs)
            except Exception as e:
                reward = 0.0
                done   = True
                error  = str(e)

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        # Calculate score
        score   = sum(rewards) / len(rewards) if rewards else 0.0
        score   = round(min(max(score, 0.0), 1.0), 3)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return {
        "task":    task_name,
        "score":   score,
        "success": success,
        "steps":   steps_taken,
        "rewards": rewards,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Initialize OpenAI client
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY or "dummy-key",
    )

    # Get all tasks from environment
    try:
        tasks_resp = http_get(f"{ENV_URL}/tasks")
        tasks      = [t["id"] for t in tasks_resp["tasks"]]
    except Exception as e:
        print(f"[DEBUG] Failed to get tasks: {e}", flush=True)
        tasks = [
            "severity_classification",
            "duplicate_detection",
            "full_triage",
        ]

    print(f"[DEBUG] Running {len(tasks)} tasks: {tasks}", flush=True)

    # Run one episode per task
    all_results = []
    for task_name in tasks:
        result = run_episode(client, task_name)
        all_results.append(result)

    # Summary
    avg_score = sum(r["score"] for r in all_results) / len(all_results)
    print(f"\n[DEBUG] Average score: {avg_score:.3f}", flush=True)
    print(f"[DEBUG] All done.", flush=True)


if __name__ == "__main__":
    main()