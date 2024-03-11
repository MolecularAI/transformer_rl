from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumStrategyEnum:
    STANDARD = "standard"
    LINK_INVENT = "link_invent"
    MOL2MOL = "mol2mol"
    NO_CURRICULUM = "no_curriculum"
