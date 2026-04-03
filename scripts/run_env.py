from __future__ import annotations

import json

from meddataops.env import MedDataOpsEnv
from meddataops.models import ActionType, AgentAction


def main() -> None:
    env = MedDataOpsEnv(seed=7)
    initial_state = env.reset()
    print("Initial state:")
    print(json.dumps(initial_state.model_dump(), indent=2))

    noop_action = AgentAction(action_type=ActionType.NOOP, payload={})
    step_result = env.step(noop_action)
    print("\nAfter one noop step:")
    print(json.dumps(step_result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
