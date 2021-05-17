import os
import datetime
import pathlib
import shutil


def generate_run_directory() -> dict:
    '''Create directory for the run, log files .

    Returns:
        (dict): Dictionary containing names of directories and their paths, paths are pathlib.Path.
    '''

    run_paths = dict()
    project_root_path = pathlib.Path(__file__).parent.parent / 'experiments'
    date_creation = datetime.datetime.now().strftime('%Y_%m_%d-T%H_%M_%S')
    run_paths['run_directory'] = project_root_path / ('run_' + date_creation)

    # Saved models directory for ensemble
    run_paths['trained_models_directory'] = project_root_path / 'trained_models'

    # Logs
    run_paths['logs_directory'] = run_paths['run_directory'] / 'logs'
    run_paths['base_log'] = run_paths['logs_directory'] / 'run.log'
    run_paths['training_log'] = run_paths['logs_directory'] / 'training.log'
    run_paths['evaluation_log'] = run_paths['logs_directory'] / 'evaluation.log'
    run_paths['ensemble_log'] = run_paths['logs_directory'] / 'ensemble.log'

    # Hyperparameter directory for Tensorboard
    run_paths['hyperparameter_tuning_directory'] = run_paths['run_directory'] / 'hyperparameter_tuning'

    # Model specific directories
    run_paths['model_directories'] = dict()

    # gin configuration
    run_paths['path_gin'] = run_paths['run_directory'] / 'config_operative.gin'

    # Directory for fully trained models
    run_paths['saved_models_directory'] = run_paths['run_directory'] / 'saved_models'

    # Create directories
    for name, path in run_paths.items():
        if ('_directory' in name) and not path.exists():
            path.mkdir(parents=True, exist_ok=True)

    # Create files
    for name, path in run_paths.items():
        if any([suffix in name for suffix in ['_log', '_gin']]):
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch(exist_ok=True)

    return run_paths


def generate_model_directories(model_name, run_paths) -> dict:
    '''Generate model specific directories for the given model.
    
    Parameters
    ----------
    `model_name` : string
        Name of a model to be used as the directory name.

    `run_path` : dict
        Dictionary containing the keys 'run_directory' and 'model_directories'.
    '''

    model_directory = run_paths['run_directory'] / model_name

    run_paths['model_directories'][model_name] = dict()

    run_paths['model_directories'][model_name]['summaries'] = model_directory / 'summaries'

    # Checkpoints
    run_paths['model_directories'][model_name]['training_checkpoints'] = model_directory / 'checkpoints'
    run_paths['model_directories'][model_name]['early_stopping_checkpoints'] = model_directory / 'checkpoints' / 'early_stopping'
    
    # Create directories
    for path in run_paths['model_directories'][model_name].values():
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)


def delete_previous_hyperparameter_runs(run_paths) -> None:
    """ Deletes existing runs in hyperparameter directory
        This method should be called once at the beginning of the program.
           Parameters:
               `run_path` (dict): Dictionary containing the paths.
    """
    hyperparameter_directory = run_paths['hyperparameter_tuning_directory']

    # Deleting previous hyperparameter run logs
    for filename in os.listdir(str(hyperparameter_directory)):
        file_path = str(hyperparameter_directory / filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def generate_hyperparameter_model_directories(model_name, run_paths) -> None:
    """Generate Model dictionaries inside hyperparamater main directory

       Parameters:
           `run_path` (dict): Dictionary containing the keys 'run_directory',
                'hyperparameter_tuning', 'model_directories'.
    """

    hyperparameter_directory = run_paths['hyperparameter_tuning_directory']
    hyperparameter_directory_model_directory = hyperparameter_directory / model_name

    # Create the directory
    if not hyperparameter_directory_model_directory.exists():
        hyperparameter_directory_model_directory.mkdir(parents=True, exist_ok=True)


def save_config(path_gin, config) -> None:
    '''Save the gin configuration used in the current run.'''

    with open(path_gin, 'w') as gin_config_file:
        gin_config_file.write(config)
