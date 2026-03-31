from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Severity(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class Team(str, Enum):
    BACKEND  = "backend"
    FRONTEND = "frontend"
    INFRA    = "infra"
    SECURITY = "security"
    DATA     = "data"
    MOBILE   = "mobile"
    UNKNOWN  = "unknown"


class TaskType(str, Enum):
    SEVERITY_CLASSIFICATION = "severity_classification"
    DUPLICATE_DETECTION     = "duplicate_detection"
    FULL_TRIAGE             = "full_triage"


class BugReport(BaseModel):
    id: str
    title: str
    description: str
    steps_to_reproduce: list[str]
    environment: str
    reporter: str
    created_at: str


class TriageAction(BaseModel):
    severity: Optional[Severity] = None
    duplicate_of: Optional[str] = None
    similarity_reasoning: Optional[str] = None
    assigned_team: Optional[Team] = None
    root_cause_hypothesis: Optional[str] = None
    reproduction_verified: Optional[bool] = None
    priority_justification: Optional[str] = None


class TriageObservation(BaseModel):
    task_type: TaskType
    current_bug: BugReport
    existing_bugs: list[BugReport] = Field(default_factory=list)
    last_reward: float = 0.0
    last_feedback: str = ""
    step_number: int = 0
    max_steps: int = 3
    task_description: str = ""


class TriageReward(BaseModel):
    total: float
    severity_score: float = 0.0
    duplicate_correct: float = 0.0
    reasoning_quality: float = 0.0
    severity_score_t3: float = 0.0
    team_assignment_score: float = 0.0
    root_cause_score: float = 0.0
    justification_score: float = 0.0
    explanation: str = ""


class TriageStepResult(BaseModel):
    observation: TriageObservation
    reward: float
    reward_breakdown: TriageReward
    done: bool
    info: dict = {}