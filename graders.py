import os
from models import TriageAction, TriageReward, Severity, Team, BugReport

SEVERITY_ORDER = {
    Severity.P0: 0,
    Severity.P1: 1,
    Severity.P2: 2,
    Severity.P3: 3,
}

def severity_distance(predicted: Severity, ground_truth: Severity) -> int:
    return abs(SEVERITY_ORDER[predicted] - SEVERITY_ORDER[ground_truth])


# ── Task 1 — Severity Classification (Easy) ──────────────────────────────────

def grade_severity(action: TriageAction, ground_truth_severity: Severity) -> TriageReward:
    if action.severity is None:
        return TriageReward(
            total=0.0,
            severity_score=0.0,
            explanation="No severity provided.",
        )

    dist = severity_distance(action.severity, ground_truth_severity)

    if dist == 0:
        score = 1.0
        explanation = f"Correct. {action.severity} matches {ground_truth_severity}."
    elif dist == 1:
        score = 0.5
        explanation = f"Off by one level. Got {action.severity}, expected {ground_truth_severity}."
    elif dist == 2:
        score = 0.2
        explanation = f"Off by two levels. Got {action.severity}, expected {ground_truth_severity}."
    else:
        score = 0.0
        explanation = f"Completely wrong. Got {action.severity}, expected {ground_truth_severity}."

    return TriageReward(
        total=round(score, 3),
        severity_score=round(score, 3),
        explanation=explanation,
    )


# ── Task 2 — Duplicate Detection (Medium) ────────────────────────────────────

def grade_duplicate(
    action: TriageAction,
    ground_truth_duplicate_id: str,
    existing_bugs: list[BugReport],
) -> TriageReward:
    predicted = (action.duplicate_of or "none").strip().lower()
    expected  = ground_truth_duplicate_id.strip().lower()

    duplicate_correct = 1.0 if predicted == expected else 0.0

    reasoning = (action.similarity_reasoning or "").lower()
    reasoning_keywords = [
        "similar", "same", "duplicate", "identical", "matches",
        "token", "logout", "session", "auth", "different", "unrelated", "new"
    ]
    keyword_hits = sum(1 for kw in reasoning_keywords if kw in reasoning)
    reasoning_quality = min(0.3, keyword_hits * 0.1)

    if duplicate_correct:
        total = 0.7 + reasoning_quality
        explanation = (
            f"Correct. Identified {'no duplicate' if predicted == 'none' else predicted}. "
            f"Reasoning score: {reasoning_quality:.1f}/0.3."
        )
    else:
        total = reasoning_quality * 0.3
        explanation = (
            f"Wrong. Expected {expected}, got {predicted}. "
            f"Small reasoning credit: {total:.2f}."
        )

    return TriageReward(
        total=round(min(1.0, total), 3),
        duplicate_correct=duplicate_correct,
        reasoning_quality=round(reasoning_quality, 3),
        explanation=explanation,
    )


# ── Task 3 — Full Triage (Hard) ───────────────────────────────────────────────

def grade_full_triage(
    action: TriageAction,
    ground_truth: dict,
    bug: BugReport,
) -> TriageReward:

    # 1. Severity (30%)
    severity_gt = ground_truth.get("severity", Severity.P2)
    if action.severity:
        dist = severity_distance(action.severity, severity_gt)
        sev_score = max(0.0, 1.0 - (dist * 0.4))
    else:
        sev_score = 0.0

    # 2. Team assignment (25%)
    team_gt = ground_truth.get("team", Team.UNKNOWN)
    if action.assigned_team is None:
        team_score = 0.0
    elif action.assigned_team == team_gt:
        team_score = 1.0
    elif _teams_are_adjacent(action.assigned_team, team_gt):
        team_score = 0.5
    else:
        team_score = 0.0

    # 3. Root cause (25%)
    root_cause_hint = ground_truth.get("root_cause_hint", "")
    root_score = _score_root_cause(
        hypothesis=action.root_cause_hypothesis or "",
        hint=root_cause_hint,
        bug=bug,
    )

    # 4. Justification (20%)
    just_score = _score_justification(action.priority_justification or "")

    total = (
        0.30 * sev_score +
        0.25 * team_score +
        0.25 * root_score +
        0.20 * just_score
    )

    explanation = (
        f"Severity: {sev_score:.2f} | "
        f"Team: {team_score:.2f} | "
        f"Root cause: {root_score:.2f} | "
        f"Justification: {just_score:.2f} | "
        f"Total: {total:.3f}"
    )

    return TriageReward(
        total=round(total, 3),
        severity_score_t3=round(sev_score, 3),
        team_assignment_score=round(team_score, 3),
        root_cause_score=round(root_score, 3),
        justification_score=round(just_score, 3),
        explanation=explanation,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

ADJACENT_TEAMS: dict[Team, list[Team]] = {
    Team.BACKEND:  [Team.INFRA, Team.DATA],
    Team.FRONTEND: [Team.MOBILE],
    Team.INFRA:    [Team.BACKEND, Team.DATA],
    Team.SECURITY: [Team.BACKEND, Team.INFRA],
    Team.DATA:     [Team.BACKEND, Team.INFRA],
    Team.MOBILE:   [Team.FRONTEND],
}

def _teams_are_adjacent(a: Team, b: Team) -> bool:
    return b in ADJACENT_TEAMS.get(a, [])


def _score_root_cause(hypothesis: str, hint: str, bug: BugReport) -> float:
    if not hypothesis:
        return 0.0

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return _llm_score_root_cause(hypothesis, hint, bug, api_key)

    hint_words  = set(hint.lower().split()) - {"a","an","the","is","in","on","at","to","of","for","not"}
    hyp_words   = set(hypothesis.lower().split())
    overlap     = hint_words & hyp_words

    if not hint_words:
        return 0.5

    coverage     = len(overlap) / len(hint_words)
    length_bonus = 0.1 if len(hypothesis.split()) > 8 else 0.0
    return round(min(1.0, coverage * 0.9 + length_bonus), 3)


def _llm_score_root_cause(hypothesis: str, hint: str, bug: BugReport, api_key: str) -> float:
    try:
        from openai import OpenAI
        import json

        client = OpenAI(api_key=api_key)
        prompt = f"""Rate this root cause hypothesis for a bug triage (0.0-1.0).

BUG: {bug.title}
EXPECTED ROOT CAUSE: {hint}
AGENT HYPOTHESIS: {hypothesis}

Respond ONLY with JSON: {{"score": 0.0, "reason": "..."}}"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100,
        )
        data = json.loads(resp.choices[0].message.content)
        return round(float(data["score"]), 3)
    except Exception:
        return _score_root_cause(hypothesis, hint, bug)


def _score_justification(text: str) -> float:
    if not text:
        return 0.0
    words = text.split()
    length_score = min(0.5, len(words) / 30)
    quality_keywords = [
        "because", "since", "affects", "users", "critical",
        "blocking", "workaround", "impact", "production", "security"
    ]
    kw_score = min(0.5, sum(1 for kw in quality_keywords if kw in text.lower()) * 0.1)
    return round(length_score + kw_score, 3)