# pyright: standard

from .get_config_path import get_config_path
from .score_multiple_choice import score_sample
from .load_dataset import load_dataset
from inspect_ai import Task, task
from agent import mas_agent

@task
def autocrat() -> Task:
    config_name = "autocrat_3"
    return Task(
        dataset=load_dataset(),
        solver= mas_agent(get_config_path(config_name)),
        scorer=[score_sample()],
        metadata=dict(config=config_name)

    )

  