from dataclasses import dataclass


@dataclass(frozen=True)
class ComponentSpecificParametersEnum:
    SCIKIT = "scikit"
    DESCRIPTOR_TYPE = "descriptor_type"
    TRANSFORMATION = "transformation"

    # matching substructure
    USE_CHIRALITY = "use_chirality"

    # structural components
    # ---------
    # AZDOCK
    AZDOCK_CONFPATH = "configuration_path"
    AZDOCK_DOCKERSCRIPTPATH = "docker_script_path"
    AZDOCK_ENVPATH = "environment_path"
    AZDOCK_DEBUG = "debug"

    # DockStream
    DOCKSTREAM_CONFPATH = "configuration_path"
    DOCKSTREAM_DOCKERSCRIPTPATH = "docker_script_path"
    DOCKSTREAM_ENVPATH = "environment_path"
    DOCKSTREAM_DEBUG = "debug"

    # ICOLOS
    ICOLOS_CONFPATH = "configuration_path"
    ICOLOS_EXECUTOR_PATH = "executor_path"
    ICOLOS_VALUES_KEY = "values_key"
    ICOLOS_DEBUG = "debug"
    #######################

    CONTAINER_TYPE = "container_type"

    SMILES = "smiles"
    MODEL_PATH = "model_path"

    #######################
    PREDICTION_URL = "prediction_url"

    VALUE_MAPPING = "value_mapping"

    BACKEND = "backend"
