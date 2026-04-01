"""
server/environment.py — The core Bug Triage environment.

This is the brain. OpenEnv calls these 3 methods:
  reset()  → give the agent a fresh bug to triage
  step()   → receive the agent's triage → score it → return reward
  state()  → return episode metadata

How one episode works:
  1. reset() picks a task type + a bug report from the dataset
  2. Agent calls step() with its triage decision
  3. Grader scores it → reward returned
  4. Agent gets up to max_steps attempts (can refine its triage)
  5. done=True after max_steps or perfect score
"""

import random
import uuid
from typing import Optional

from models import (
    TriageAction, TriageObservation, TriageStepResult,
    TriageReward, TaskType, BugReport, Severity
)
from graders import grade_severity, grade_duplicate, grade_full_triage
from data.bug_dataset import BUGS_WITH_LABELS, DUPLICATE_BUGS

# OpenEnv base class
try:
    from openenv import Environment, State
except ImportError:
    # Fallback for local dev without openenv installed
    class Environment:
        pass
    class State:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


TASK_DESCRIPTIONS = {
    TaskType.SEVERITY_CLASSIFICATION: (
        "Read the bug report carefully. Classify its severity as one of: "
        "P0 (critical — system down or security breach), "
        "P1 (major — core feature broken, no workaround), "
        "P2 (minor — feature broken but workaround exists), "
        "P3 (trivial — cosmetic or typo). "
        "Set the 'severity' field in your action."
    ),
    TaskType.DUPLICATE_DETECTION: (
        "Read the new bug report. Compare it against the list of existing bugs. "
        "If it's a duplicate, set 'duplicate_of' to the ID of the existing bug. "
        "If it's NOT a duplicate, set 'duplicate_of' to 'none'. "
        "Always provide 'similarity_reasoning' explaining your decision in one sentence."
    ),
    TaskType.FULL_TRIAGE: (
        "Perform a complete triage of this bug report. You must provide: "
        "severity (P0–P3), assigned_team (backend/frontend/infra/security/data/mobile), "
        "root_cause_hypothesis (your best technical guess at what's causing this), "
        "reproduction_verified (true/false — do the steps make sense?), "
        "priority_justification (1–2 sentences explaining your severity + team choice)."
    ),
}


class BugTriageEnvironment(Environment):
    """
    Bug Triage OpenEnv environment.

    Supports 3 tasks of increasing difficulty:
      - severity_classification (easy)
      - duplicate_detection (medium)
      - full_triage (hard)

    task_type can be set explicitly or left as None for random selection.
    """

    def __init__(self, task_type: Optional[TaskType] = None, max_steps: int = 3):
        self.task_type = task_type           # None = random each episode
        self.max_steps = max_steps

        # Episode state — reset on each reset()
        self._episode_id: Optional[str] = None
        self._current_task: Optional[TaskType] = None
        self._current_bug: Optional[BugReport] = None
        self._current_bug_label: Optional[dict] = None
        self._existing_bugs: list[BugReport] = []
        self._step_count: int = 0
        self._last_reward: float = 0.0
        self._last_feedback: str = ""
        self._cumulative_reward: float = 0.0

    # ── OpenEnv required methods ──────────────────────────────────────────────

    def reset(self) -> TriageObservation:
        """Start a fresh episode. Pick a task + bug report."""
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._last_reward = 0.0
        self._last_feedback = ""
        self._cumulative_reward = 0.0

        # Pick task type
        self._current_task = self.task_type or random.choice(list(TaskType))

        # Pick a bug based on task
        if self._current_task == TaskType.DUPLICATE_DETECTION:
            entry = random.choice(DUPLICATE_BUGS)
            self._current_bug = entry["new_bug"]
            self._current_bug_label = {
                "duplicate_id": entry["is_duplicate_of"]
            }
            self._existing_bugs = entry["ground_truth_pool"]
        else:
            entry = random.choice(BUGS_WITH_LABELS)
            self._current_bug = entry["bug"]
            self._current_bug_label = entry["ground_truth"]
            self._existing_bugs = []

        return TriageObservation(
            task_type=self._current_task,
            current_bug=self._current_bug,
            existing_bugs=self._existing_bugs,
            last_reward=0.0,
            last_feedback="",
            step_number=0,
            max_steps=self.max_steps,
            task_description=TASK_DESCRIPTIONS[self._current_task],
        )

    def step(self, action: TriageAction) -> TriageStepResult:
        """Process the agent's triage action and return a scored result."""
        self._step_count += 1

        # Grade based on task type
        reward_obj = self._grade(action)
        self._last_reward = reward_obj.total
        self._last_feedback = reward_obj.explanation
        self._cumulative_reward += reward_obj.total

        # Done if max steps reached or perfect score
        done = (self._step_count >= self.max_steps) or (reward_obj.total >= 1.0)

        observation = TriageObservation(
            task_type=self._current_task,
            current_bug=self._current_bug,
            existing_bugs=self._existing_bugs,
            last_reward=reward_obj.total,
            last_feedback=reward_obj.explanation,
            step_number=self._step_count,
            max_steps=self.max_steps,
            task_description=TASK_DESCRIPTIONS[self._current_task],
        )

        return TriageStepResult(
            observation=observation,
            reward=reward_obj.total,
            reward_breakdown=reward_obj,
            done=done,
            info={
                "episode_id": self._episode_id,
                "step": self._step_count,
                "task": self._current_task,
                "cumulative_reward": round(self._cumulative_reward, 3),
            },
        )

    def state(self) -> State:
        """Return current episode metadata."""
        return State(
            episode_id=self._episode_id or "",
            step_count=self._step_count,
            metadata={
                "task_type": self._current_task,
                "bug_id": self._current_bug.id if self._current_bug else None,
                "cumulative_reward": self._cumulative_reward,
            },
        )

    # ── Internal grading dispatch ─────────────────────────────────────────────

    def _grade(self, action: TriageAction) -> TriageReward:
        if self._current_task == TaskType.SEVERITY_CLASSIFICATION:
            return grade_severity(
                action=action,
                ground_truth_severity=self._current_bug_label["severity"],
            )
        elif self._current_task == TaskType.DUPLICATE_DETECTION:
            return grade_duplicate(
                action=action,
                ground_truth_duplicate_id=self._current_bug_label["duplicate_id"],
                existing_bugs=self._existing_bugs,
            )
        elif self._current_task == TaskType.FULL_TRIAGE:
            return grade_full_triage(
                action=action,
                ground_truth=self._current_bug_label,
                bug=self._current_bug,
            )
        else:
            return TriageReward(total=0.0, explanation="Unknown task type.")