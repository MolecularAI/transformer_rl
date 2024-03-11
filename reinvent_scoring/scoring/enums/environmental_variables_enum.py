from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentalVariablesEnum:
    PIP_URL = "PIP_URL"
    PIP_KEY = "PIP_KEY"
    PIP_GET_RESULTS = "PIP_GET_RESULTS"
