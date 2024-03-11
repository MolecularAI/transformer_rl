from dataclasses import dataclass

from reinvent_models.mol2mol.dto.mol2mol_model_parameters_dto import Mol2MolNetworkParameters

@dataclass
class Mol2MolCreateModelConfiguration:
    input_smiles_path: str
    output_model_path: str
    network: Mol2MolNetworkParameters
    max_sequence_length: int = 128
    use_cuda: bool = True
