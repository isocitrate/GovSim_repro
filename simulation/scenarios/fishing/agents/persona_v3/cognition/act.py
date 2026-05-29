from datetime import datetime

from pathfinder import assistant, system, user
from simulation.persona.cognition.act import ActComponent
from simulation.utils import ModelWandbWrapper

from .act_prompts import (
    prompt_action_choose_amount_of_fish_to_catch,
    prompt_action_choose_punishment_points,
)
from .utils import get_universalization_prompt


class FishingActComponent(ActComponent):
    """

    We have to options here:
    - choose at one time-step how many fish to chat
    - choose at one time-strep whether to fish one more time
    """

    def __init__(
        self, model: ModelWandbWrapper, model_framework: ModelWandbWrapper, cfg
    ):
        super().__init__(model, model_framework, cfg)

    def choose_how_many_fish_to_chat(
        self,
        retrieved_memories: list[str],
        current_location: str,
        current_time: datetime,
        context: str,
        interval: list[int],
        overusage_threshold: int,
    ):
        if self.cfg.universalization_prompt:
            context += get_universalization_prompt(overusage_threshold)
        res, html = prompt_action_choose_amount_of_fish_to_catch(
            self.model,
            self.persona.identity,
            retrieved_memories,
            current_location,
            current_time,
            context,
            interval,
            consider_identity_persona=self.cfg.consider_identity_persona,
        )
        res = int(res)
        return res, [html]

    def choose_punishment_points(
        self,
        retrieved_memories: list[str],
        current_location: str,
        current_time: datetime,
        context: str,
        target_resources: dict[str, int],
        max_points: int,
        cost_per_point: float,
        penalty_multiplier: float,
    ):
        punishment_points = {}
        html_interactions = []
        for target_agent_id, target_resource in target_resources.items():
            target = self.persona.other_personas_from_id[target_agent_id].identity
            points, html = prompt_action_choose_punishment_points(
                self.model,
                self.persona.identity,
                retrieved_memories,
                current_location,
                current_time,
                context,
                target.name,
                target_resource,
                max_points,
                cost_per_point,
                penalty_multiplier,
            )
            punishment_points[target_agent_id] = int(max(0, min(max_points, points)))
            html_interactions.append(html)
        return punishment_points, html_interactions
