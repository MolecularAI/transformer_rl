Evaluation of Reinforcement Learning in Transformer-based Molecular Design
=================================================================================================================
## Description
This repository holds the code for [Evaluation of Reinforcement Learning in Transformer-based Molecular Design](todo). 
It was implemented based on the [Reinvent3](https://github.com/MolecularAI/Reinvent) framework.

Installation
-------------

1. Install [Conda](https://conda.io/projects/conda/en/latest/index.html)
2. Clone this Git repository
3. Open a shell, and go to the repository and create the Conda environment:
   
        $ conda env create -f reinvent.yml

4. Activate the environment:
   
        $ conda activate reinvent




Usage
-----
In `support` directory, there are the PubChem prior, ChEMBL prior, DRD2 predictive model used in the manuscript. 
Additionally, there is an example input configuration `example.json` for running the code, 

`python input.py support/example.json`
    




