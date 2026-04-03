---
title: Bug Triage Env
emoji: 🐛
colorFrom: purple
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
# Bug Triage Environment

A real-world OpenEnv environment where AI agents learn to triage
software bug reports  exactly the task engineering teams do every day.

## What it does

The agent receives bug reports and must:
- Classify severity (P0 critical → P3 trivial)
- Detect duplicate bugs
- Produce full triage reports with team assignment and root cause

## 3 Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| severity_classification | Easy | Classify bug severity P0-P3 |
| duplicate_detection | Medium | Find if bug is a duplicate |
| full_triage | Hard | Complete triage with team + root cause |

## Action Space
```json
{
  "severity": "P0 | P1 | P2 | P3",
  "duplicate_of": "bug-id or none",
  "similarity_reasoning": "string",
  "assigned_team": "backend | frontend | infra | security | data | mobile",
  "root_cause_hypothesis": "string",
  "reproduction_verified": true,
  "priority_justification": "string"
}
```

## Observation Space

Each step the agent sees:
- The full bug report (title, description, steps, environment)
- Task description and instructions
- Last reward and feedback
- Episode progress (step number / max steps)

## Reward Function

- **Easy:** 1.0 exact match, 0.5 off-by-one, 0.2 off-by-two
- **Medium:** 0.7 correct duplicate + 0.3 reasoning quality
- **Hard:** Weighted — severity 30%, team 25%, root cause 25%, justification 20%

## Setup
```bash
pip install -r requirements.txt
uvicorn server.app:app --port 8000
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /reset | POST | Start new episode |
| /step | POST | Submit triage action |
| /state | GET | Current episode state |
| /tasks | GET | List all tasks |
| /grader | GET | Last episode score |
| /baseline | POST | Run baseline agent |

## Run Baseline
```bash
# Without API key (rule-based agent)
python3 baseline.py

# With OpenAI key (GPT-4o-mini agent)
OPENAI_API_KEY=your-key python3 baseline.py
```

## Baseline Scores

| Task | Rule-based | GPT-4o-mini |
|------|-----------|-------------|
| severity_classification | ~0.5 | ~0.8 |
| duplicate_detection | ~0.3 | ~0.7 |
| full_triage | ~0.3 | ~0.6 |

## Docker
```bash
docker build -t bug-triage-env .
docker run -p 8000:8000 bug-triage-env
```