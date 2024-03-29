from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_scoring import ScoringFunctionFactory

from running_modes.automated_curriculum_learning.curriculum_strategy.curriculum_strategy import CurriculumStrategy
from running_modes.automated_curriculum_learning.inception.inception import Inception
from running_modes.automated_curriculum_learning.logging.base_logger import BaseLogger
from running_modes.automated_curriculum_learning.production_strategy.production_strategy import ProductionStrategy
from running_modes.configurations.automated_curriculum_learning.automated_curriculum_learning_input_configuration import \
    AutomatedCurriculumLearningInputConfiguration
from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.enums.curriculum_strategy_enum import CurriculumStrategyEnum
from running_modes.enums.production_strategy_enum import ProductionStrategyEnum


class AutomatedCurriculumRunner(BaseRunningMode):
    def __init__(self, config: AutomatedCurriculumLearningInputConfiguration,
                 logger: BaseLogger, prior: GenerativeModelBase, agent: GenerativeModelBase):

        self._config = config
        self._prior = prior
        self._agent = agent
        self._logger = logger

        self.curriculum_strategy_enum = CurriculumStrategyEnum()
        if self._config.curriculum_strategy.name != self.curriculum_strategy_enum.NO_CURRICULUM:
            self._curriculum_strategy = CurriculumStrategy(self._prior, self._agent, self._config.curriculum_strategy,
                                                           self._logger)

        production_sf = ScoringFunctionFactory(self._config.production_strategy.scoring_function)

        self.production_strategy_enum = ProductionStrategyEnum()
        if self._config.production_strategy.name == self.production_strategy_enum.MOL2MOL:
            production_inception = None
            self._config.production_strategy.input = self._config.curriculum_strategy.input
        elif self._config.curriculum_strategy.name != self.curriculum_strategy_enum.NO_CURRICULUM and \
            self._config.production_strategy.retain_inception:
            production_inception = self._curriculum_strategy.inception
        else:
            production_inception = Inception(self._config.production_strategy.inception, production_sf, self._prior)

        self.production_strategy = ProductionStrategy(prior=self._prior, inception=production_inception,
                                                      configuration=self._config.production_strategy,
                                                      logger=self._logger)

    def run(self):
        if self._config.curriculum_strategy.name != self.curriculum_strategy_enum.NO_CURRICULUM:
            self._logger.log_message("Starting Curriculum Learning")
            outcome = self._curriculum_strategy.run()

            if outcome.successful_curriculum:
                self._logger.log_message("Starting Production phase")
                self.production_strategy.run(outcome.agent, outcome.step_counter)
            else:
                self._logger.log_message("Failing to qualify for Production phase")
        else:
            self._logger.log_message("Starting Production phase")
            self.production_strategy.run(self._agent, 0)