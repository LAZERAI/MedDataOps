from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ActionType(str, Enum):
    CLEAN_DATA = "clean_data"
    FIX_SQL = "fix_sql"
    SUBMIT = "submit"
    NOOP = "noop"


class OpenEnvActionType(str, Enum):
    CLEAN_DATA = "clean_data"
    RUN_QUERY = "run_query"
    FIX_QUERY = "fix_query"
    SUBMIT = "submit"


class ObservationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_dataset_state: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Current tabular dataset snapshot visible to the agent at this step.",
    )
    current_sql_query: str = Field(
        default="",
        description="Current SQL query string the agent is evaluating or editing.",
    )
    error_messages: list[str] = Field(
        default_factory=list,
        description="Validation, execution, or runtime errors observed in the current step.",
    )
    task_description: str = Field(
        ...,
        description="Natural-language description of the objective the agent must solve.",
    )
    step_number: int = Field(
        default=0,
        ge=0,
        description="Zero-based step index within the current episode.",
    )


class ActionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: OpenEnvActionType = Field(
        ...,
        description="Action category selected by the agent.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific arguments (for example, rows to clean or SQL text to run/fix).",
    )


class RewardModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_clean_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Reward contribution for correctness of data cleaning output.",
    )
    query_correct_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Reward contribution for SQL correctness against expected logic/results.",
    )
    efficiency_bonus: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Positive reward bonus for solving with fewer steps or lower cost.",
    )
    step_penalty: float = Field(
        ...,
        ge=-1.0,
        le=0.0,
        description="Negative per-step penalty to encourage concise solutions.",
    )
    total: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Total reward for the step after combining all reward components.",
    )


class HistoryEntryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_number: int = Field(
        ...,
        ge=0,
        description="Step number when this history entry was recorded.",
    )
    action: ActionModel = Field(
        ...,
        description="Action executed by the agent at this step.",
    )
    observation: ObservationModel = Field(
        ...,
        description="Observation received immediately after action execution.",
    )
    reward: RewardModel = Field(
        ...,
        description="Structured reward assigned for this action/observation transition.",
    )


class StateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(
        ...,
        description="Unique identifier for the active environment episode.",
    )
    step_number: int = Field(
        ...,
        ge=0,
        description="Current zero-based step index in the episode.",
    )
    max_steps: int = Field(
        ...,
        ge=1,
        description="Maximum number of steps allowed before forced termination.",
    )
    done: bool = Field(
        ...,
        description="Whether the current episode is terminated.",
    )
    current_task: TaskPublic = Field(
        ...,
        description="Public metadata for the active task.",
    )
    observation: ObservationModel = Field(
        ...,
        description="Current observation payload exposed to the agent.",
    )
    latest_reward: RewardModel = Field(
        ...,
        description="Most recent structured reward breakdown.",
    )
    cumulative_reward: float = Field(
        ...,
        description="Accumulated reward across all executed steps in the episode.",
    )
    solved_cleaning: bool = Field(
        ...,
        description="Whether the data cleaning objective is currently satisfied.",
    )
    solved_query: bool = Field(
        ...,
        description="Whether the SQL/query objective is currently satisfied.",
    )
    last_sql_error: str | None = Field(
        default=None,
        description="Most recent SQL execution/validation error, if any.",
    )
    history: list[HistoryEntryModel] = Field(
        default_factory=list,
        description="Chronological action-observation-reward history for the episode.",
    )


class TaskPublic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    difficulty: Difficulty
    description: str
    hints: list[str] = Field(default_factory=list)


class TaskSpec(TaskPublic):
    dirty_rows: list[dict[str, Any]]
    broken_sql: str
    expected_clean_rows: list[dict[str, Any]]
    expected_sql: str


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str | None = Field(default=None, description="Optional id: easy|medium|hard")
    seed: int | None = Field(default=None, description="Optional deterministic seed")


class AgentAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    payload: dict[str, Any] = Field(default_factory=dict)


class QueryCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    error: str | None = None
    sample_row_count: int = 0
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)


class EnvironmentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    step_index: int
    max_steps: int
    done: bool
    reward: float
    task: TaskPublic
    dirty_rows: list[dict[str, Any]]
    broken_sql: str
    last_sql_error: str | None = None
    solved_cleaning: bool = False
    solved_sql: bool = False


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: EnvironmentState
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


OBSERVATION_MODEL_EXAMPLE = ObservationModel(
    current_dataset_state=[
        {"patient_id": "P001", "age": 45, "glucose": 102.5, "diagnosis": "pre-diabetes"},
        {"patient_id": "P002", "age": 50, "glucose": None, "diagnosis": "unknown"},
    ],
    current_sql_query="SELECT patient_id, AVG(glucose) AS avg_glucose FROM labs GROUP BY patient_id;",
    error_messages=["NULL glucose values detected in row 2."],
    task_description="Clean missing glucose values and compute per-patient average glucose.",
    step_number=3,
)

ACTION_MODEL_EXAMPLE = ActionModel(
    action_type=OpenEnvActionType.FIX_QUERY,
    parameters={
        "query": "SELECT patient_id, AVG(COALESCE(glucose, 0)) AS avg_glucose FROM labs GROUP BY patient_id;",
        "reason": "Handle NULL glucose values before aggregation.",
    },
)

REWARD_MODEL_EXAMPLE = RewardModel(
    data_clean_score=0.7,
    query_correct_score=0.8,
    efficiency_bonus=0.1,
    step_penalty=-0.05,
    total=0.75,
)

STATE_MODEL_EXAMPLE = StateModel(
    episode_id="episode-demo-001",
    step_number=3,
    max_steps=25,
    done=False,
    current_task=TaskPublic(
        id="easy",
        name="Clean Missing Labs",
        difficulty=Difficulty.EASY,
        description="Fix missing values and validate summary query.",
        hints=["Use COALESCE for null-safe aggregation."],
    ),
    observation=OBSERVATION_MODEL_EXAMPLE,
    latest_reward=REWARD_MODEL_EXAMPLE,
    cumulative_reward=2.35,
    solved_cleaning=True,
    solved_query=False,
    last_sql_error="column glucose_value does not exist",
    history=[
        HistoryEntryModel(
            step_number=3,
            action=ACTION_MODEL_EXAMPLE,
            observation=OBSERVATION_MODEL_EXAMPLE,
            reward=REWARD_MODEL_EXAMPLE,
        )
    ],
)
