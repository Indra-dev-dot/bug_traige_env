"""
test_episode.py — Watch your environment work in real time.

This simulates exactly what a judge's LLM agent does:
1. Gets a bug report
2. Tries to triage it
3. Gets scored
4. Tries again with a better answer
5. Gets a higher score

Run with: python3 test_episode.py
"""

import requests
import json

BASE_URL = "https://HadesnApollo-bug-triage-env.hf.space"
# Or use local: BASE_URL = "http://localhost:7860"


def print_bug(bug):
    print(f"\n  ID: {bug['id']}")
    print(f"  Title: {bug['title']}")
    print(f"  Description: {bug['description'][:100]}...")
    print(f"  Environment: {bug['environment']}")


def run_episode(task_type=None):
    print("\n" + "="*60)
    print("STARTING NEW EPISODE")
    print("="*60)

    # Step 1 — Reset environment, get a bug report
    print("\n[1] Calling reset() — getting a bug report...")
    reset_resp = requests.post(f"{BASE_URL}/reset")
    obs = reset_resp.json()

    print(f"\n  Task: {obs['task_type']}")
    print(f"  Instructions: {obs['task_description'][:80]}...")
    print(f"\n  Bug Report:")
    print_bug(obs['current_bug'])

    print(f"\n  Max attempts: {obs['max_steps']}")

    # Step 2 — Bad agent (wrong answer intentionally)
    print("\n[2] Bad agent submits wrong answer...")
    bad_action = {
        "severity": "P3",  # probably wrong
        "assigned_team": "data",
        "root_cause_hypothesis": "unknown",
        "reproduction_verified": False,
        "priority_justification": "not sure",
        "duplicate_of": "none",
        "similarity_reasoning": "looks different",
    }

    step_resp = requests.post(
        f"{BASE_URL}/step",
        json=bad_action,
        headers={"Content-Type": "application/json"}
    )
    result = step_resp.json()

    print(f"\n  Reward: {result['reward']} / 1.0")
    print(f"  Feedback: {result['observation']['last_feedback']}")
    print(f"  Done: {result['done']}")

    # Step 3 — Better agent (more thoughtful answer)
    print("\n[3] Better agent submits improved answer...")
    good_action = {
        "severity": "P1",
        "assigned_team": "backend",
        "root_cause_hypothesis": (
            "The bug is likely caused by a missing validation check "
            "or an unhandled state in the backend service"
        ),
        "reproduction_verified": True,
        "priority_justification": (
            "This affects core functionality and blocks users from "
            "completing their workflow. Needs urgent attention."
        ),
        "duplicate_of": "none",
        "similarity_reasoning": "This appears to be a unique issue not seen before",
    }

    step_resp2 = requests.post(
        f"{BASE_URL}/step",
        json=good_action,
        headers={"Content-Type": "application/json"}
    )
    result2 = step_resp2.json()

    print(f"\n  Reward: {result2['reward']} / 1.0")
    print(f"  Feedback: {result2['observation']['last_feedback']}")
    print(f"  Done: {result2['done']}")

    # Step 4 — Check grader
    print("\n[4] Checking grader scores...")
    grader_resp = requests.get(f"{BASE_URL}/grader")
    grader = grader_resp.json()

    print(f"\n  Episode ID: {grader['episode_id'][:8]}...")
    print(f"  Total steps taken: {grader['step_count']}")
    print(f"  Cumulative reward: {grader['cumulative_reward']}")

    print("\n" + "="*60)
    print(f"EPISODE COMPLETE")
    print(f"Bad answer score:  {result['reward']}")
    print(f"Good answer score: {result2['reward']}")
    improvement = result2['reward'] - result['reward']
    print(f"Improvement:       +{improvement:.3f}")
    print("="*60)


def run_all_tasks():
    """Run one episode for each task type."""
    print("\n" + "="*60)
    print("RUNNING ALL 3 TASKS")
    print("="*60)

    # First check tasks endpoint
    tasks_resp = requests.get(f"{BASE_URL}/tasks")
    tasks = tasks_resp.json()["tasks"]

    print(f"\nFound {len(tasks)} tasks:")
    for t in tasks:
        print(f"  - {t['name']} ({t['difficulty']})")

    # Run baseline
    print("\n[BASELINE] Running official baseline agent...")
    baseline_resp = requests.post(f"{BASE_URL}/baseline")
    baseline = baseline_resp.json()

    print(f"\nBaseline Results:")
    for task_id, scores in baseline["baseline_scores"].items():
        print(f"  {task_id}:")
        print(f"    Reward: {scores['reward']}")
        print(f"    Feedback: {scores['feedback'][:60]}...")

    print(f"\nAverage score: {baseline['average_score']}")


if __name__ == "__main__":
    print("Bug Triage Environment — Live Test")
    print("Connecting to:", BASE_URL)

    # Test health first
    health = requests.get(f"{BASE_URL}/health")
    print(f"Health check: {health.json()}")

    # Run a full episode
    run_episode()

    # Run all tasks with baseline
    run_all_tasks()