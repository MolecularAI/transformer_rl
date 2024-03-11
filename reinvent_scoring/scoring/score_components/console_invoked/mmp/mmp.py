import os
import shutil
import subprocess
import tempfile

import numpy as np
from typing import List
import pandas as pd

from pandas.errors import EmptyDataError
from rdkit import Chem

from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components.console_invoked.base_console_invoked_component import BaseConsoleInvokedComponent
from reinvent_chemistry.conversions import Conversions

from reinvent_scoring.scoring.score_components.console_invoked.mmp.mmp_parameter_dto import MMPParameterDTO


class MMP(BaseConsoleInvokedComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self._mmp_parameters = MMPParameterDTO.parse_obj(self.parameters.specific_parameters)
        self._mmp_parameters.mmp_reference_molecules = [Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
                             for smi in self._mmp_parameters.mmp_reference_molecules if Chem.MolFromSmiles(smi)]

        self._parent_temporary_directory = self._mmp_parameters.mmp_temporary_directory
        if self._parent_temporary_directory is None:
            self._parent_temporary_directory = os.getcwd()
        os.makedirs(self._parent_temporary_directory, exist_ok=True)
        self._temporary_directory = tempfile.mkdtemp(prefix="mmpdb_temporary_", dir=self._parent_temporary_directory)

        self._chemistry = Conversions()

    def _create_command(self, input_smi_path: str, output_fragment_path: str, output_index_path: str):
        commands = []
        command = f"mmpdb --quiet fragment {input_smi_path} -" \
                  f"-num-cuts {self._mmp_parameters.mmpdb_parameters.num_of_cuts} " \
                  f"--delimiter={self._mmp_parameters.mmpdb_parameters.delimiter} " \
                  f"--output {output_fragment_path}"
        commands.append(command)

        command = f"mmpdb --quiet index {output_fragment_path} --out 'csv' " \
                  f"--symmetric " \
                  f"--max-variable-heavies {self._mmp_parameters.mmpdb_parameters.max_variable_heavies} " \
                  f"--output {output_index_path} " \
                  f"--max-variable-ratio {self._mmp_parameters.mmpdb_parameters.max_variable_ratio}"
        commands.append(command)

        return commands

    def _calculate_score(self, molecules: List, step) -> np.array:
        os.makedirs(self._temporary_directory, exist_ok=True)
        input_smi_path = os.path.join(self._temporary_directory, "mmp_input.smi")
        output_fragment_path = os.path.join(self._temporary_directory, 'mmp_output.fragments')
        output_index_path = os.path.join(self._temporary_directory, 'mmp_indexed.csv')

        smiles = self._chemistry.mols_to_smiles(molecules, isomericSmiles=False)
        self._prepare_mmp_input(smiles, input_smi_path)

        # create the external command
        command = self._create_command(input_smi_path, output_fragment_path, output_index_path)

        # execute mmpdb
        self._execute_command(commands=command)

        # retrieve start-generated pairs
        result_df = self._retrieve_reference_generated_mmp(output_index_path)

        # get mmp score (0.5 for not mmp or 1 for mmp default)
        score = self._get_mmp_result(result_df, self._mmp_parameters.mmp_reference_molecules, smiles)

        # clean up
        if not self._mmp_parameters.mmp_debug:
            self._clean_up_temporary_folder(self._temporary_directory)

        return score, None

    def _get_mmp_result(self, result_df: pd.DataFrame, reference_molecules: List[str], smiles: List[str]):
        mmp_result = []
        for smi in smiles:
            mmp_ref_list = []
            for ref_smi in reference_molecules:
                if result_df is not None:
                    if len(result_df[(result_df['Source_Smi']==ref_smi) & (result_df['Target_Smi']==smi)]) > 0:
                        mmp_ref_list.append(1)
                    else:
                        mmp_ref_list.append(0)
                else:
                    mmp_ref_list.append(0)
            mmp_result.append(self._mmp_parameters.value_mapping['MMP'] if any(mmp_ref_list)
                              else self._mmp_parameters.value_mapping['Not MMP'])

        return np.array(mmp_result)

    def _prepare_mmp_input(self, smiles: List[str], input_mmp_path: str):
        mmp_input_mol, mmp_input_ID = [], []
        for i, smi in enumerate(self._mmp_parameters.mmp_reference_molecules):
            mmp_input_mol.append(smi)
            mmp_input_ID.append(f'Source_ID_{i+1}')

        for j, smi in enumerate(smiles):
            mmp_input_mol.append(smi)
            mmp_input_ID.append(f"Generated_ID_{j+1}")

        # save to file for mmp input
        df_output = pd.DataFrame(list(zip(mmp_input_mol, mmp_input_ID)),
                                 columns=['SMILES', 'ID'])
        df_output.to_csv(input_mmp_path, index=False, header=False)

    def _execute_command(self, commands: List[str]):
        for command in commands:
            subprocess.run(command, shell=True)

    def _retrieve_reference_generated_mmp(self, index_file: str):
        try:
            data = pd.read_csv(index_file, sep='\t', header=None)
        except EmptyDataError:
            data = pd.DataFrame()
        if len(data) > 0:
            data.columns = ['Source_Smi', 'Target_Smi', 'Source_Mol_ID', 'Target_Mol_ID', 'Transformation', 'Core']
            # Remove duplicated and generated-generated, ref-ref pairs
            data = self._get_reference_generated_pairs(data)
            data = self._remove_duplicated_transformations(data)
            return data
        else:
            return None

    def _clean_up_temporary_folder(self, temporary_folder):
        if os.path.isdir(temporary_folder):
            shutil.rmtree(temporary_folder)

    def _get_reference_generated_pairs(self, data: pd.DataFrame) -> pd.DataFrame:
        # Get pairs between reference and generated molecules, ignoring generated-generated and ref-ref pairs
        result = data[(data['Source_Mol_ID'].str.contains('Source_ID')) & (
            data['Target_Mol_ID'].str.contains('Generated_ID'))]
        return result

    def _remove_duplicated_transformations(self, data: pd.DataFrame) -> pd.DataFrame:
        # Keep only smallest transformation
        data['Source_R_len'] = data['Transformation'].apply(len)
        data = data.sort_values('Source_R_len')
        data.drop_duplicates(subset=['Source_Mol_ID', 'Target_Mol_ID'], inplace=True)
        return data