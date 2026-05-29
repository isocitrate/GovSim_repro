import pytest

pytest.importorskip("omegaconf")
pytest.importorskip("pettingzoo")

from omegaconf import OmegaConf

from simulation.persona.common import PersonaActionPunishment
from simulation.scenarios.fishing.environment.env import FishingConcurrentEnv


def test_punishment_accounting_clamps_and_logs(tmp_path):
    cfg = OmegaConf.create(
        {
            "num_agents": 2,
            "initial_resource_in_pool": 100,
            "assign_resource_strategy": "stochastic",
            "harvesting_order": "concurrent",
            "inject_universalization": False,
            "enable_punishment": True,
            "punishment": {
                "cost_per_point": 1,
                "penalty_multiplier": 3,
                "max_points_per_target": 10,
            },
        }
    )
    env = FishingConcurrentEnv(
        cfg,
        str(tmp_path),
        {"persona_0": "John", "persona_1": "Kate"},
    )
    env.reset(seed=0)
    env.internal_global_state["action"] = {
        "persona_0": PersonaActionPunishment(
            "persona_0", "punishment", {"persona_1": 12}
        ),
        "persona_1": PersonaActionPunishment(
            "persona_1", "punishment", {"persona_0": 2}
        ),
    }

    env._apply_punishments()

    assert env.rewards["persona_0"] == -16
    assert env.rewards["persona_1"] == -32
    assert env.internal_global_state["punishment_cost"]["persona_0"] == 10
    assert env.internal_global_state["punishment_penalty"]["persona_1"] == 30
    assert len(env.df_acc) == 2
