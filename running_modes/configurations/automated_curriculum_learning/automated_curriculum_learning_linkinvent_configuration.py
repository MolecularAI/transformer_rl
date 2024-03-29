from running_modes.configurations.automated_curriculum_learning.base_configuration import BaseConfiguration
from running_modes.configurations.automated_curriculum_learning.linkinvent_curriculum_strategy_configuration import \
    LinkInventCurriculumStrategyConfiguration
from running_modes.configurations.automated_curriculum_learning.linkinvent_production_strategy_configuration import \
    LinkInventProductionStrategyConfiguration


class AutomatedCurriculumLearningLinkInventConfiguration(BaseConfiguration):
    agent: str
    prior: str
    curriculum_strategy: LinkInventCurriculumStrategyConfiguration
    production_strategy: LinkInventProductionStrategyConfiguration