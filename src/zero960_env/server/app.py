from __future__ import annotations

from openenv.core.env_server import create_app

from zero960_env.models import Zero960Action, Zero960Observation
from zero960_env.server.environment import Zero960Environment

app = create_app(
    env_class=Zero960Environment,
    action_type=Zero960Action,
    observation_type=Zero960Observation,
    env_name="zero960",
)

