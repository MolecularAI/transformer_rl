from dataclasses import dataclass
from typing import Union

import torch

from reinvent_models.link_invent.dto.linkinvent_batch_dto import LinkInventBatchDTO
from reinvent_models.mol2mol.dto.mol2mol_batch_dto import Mol2MolBatchDTO


@dataclass
class BatchLikelihoodDTO:
    batch: Union[Mol2MolBatchDTO, LinkInventBatchDTO]
    likelihood: torch.Tensor