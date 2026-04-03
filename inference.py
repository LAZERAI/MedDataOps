from __future__ import annotations

import json

from openai import OpenAI

from meddataops.config import settings
from meddataops.env import MedDataOpsEnv
from meddataops.models import AgentAction


SYSTEM_PROMPT = (
    "You are a clinical data engineering policy. "
    "Given the environment state, return JSON with keys action_type and payload. "
    "Allowed action_type values: clean_data, fix_sql, submit, noop."
)


def choose_action(client: OpenAI, state_payload: dict[str, object]) -> AgentAction:
    response = client.chat.completions.create(
        model=settings.openai_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(state_payload, ensure_ascii=True),
            },
        ],
    )
    content = response.choices[0].message.content or "{}"
    return AgentAction.model_validate_json(content)


def main() -> None:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY must be set to run inference.py")

    env = MedDataOpsEnv(seed=42)
    state = env.reset()
    client = OpenAI(api_key=settings.openai_api_key)

    while not state.done:
        action = choose_action(client, state.model_dump())
        result = env.step(action)
        state = result.observation
        print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
