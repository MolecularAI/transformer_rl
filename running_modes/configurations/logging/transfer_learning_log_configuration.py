from running_modes.configurations.logging.base_log_config import BaseLoggerConfiguration


class TransferLearningLoggerConfig(BaseLoggerConfiguration):
    use_weights: bool = False
