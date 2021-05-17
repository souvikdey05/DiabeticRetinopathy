import gin
import logging
import absl
import tensorflow as tf
from tensorflow.python.util.deprecation import _PRINT_DEPRECATION_WARNINGS
import pathlib
import shutil

from tune import hyperparameter_tuning
from train import Trainer
from input_pipeline.dataset_loader import DatasetLoader 
from utils import utils_params, utils_misc
from models.architectures import load_model, TransferLearningModel, inception_like, vgg_like
from evaluation.evaluation import Evaluator
from evaluation.ensemble import StackingEnsemble
from evaluation.visualization import GradCAM

FLAGS = absl.flags.FLAGS
# Use --train or --train=true to set this flag, or nothing, true is default.
# Use --notrain or --train=false to set this flag to false.
absl.flags.DEFINE_boolean(name='train', default=False,  help='Specify whether to train a model.')
absl.flags.DEFINE_boolean(name='eval',  default=False, help='Specify whether to evaluate a model.')
absl.flags.DEFINE_boolean(name='ensem', default=False, help='Specify whether to use ensemble learning.')

# Configure the number of threads used by tensorflow
# tf.config.threading.set_intra_op_parallelism_threads(3)
# tf.config.threading.set_inter_op_parallelism_threads(3)


def _deep_visualization(model_architecture, test_dataset, dataset_info, run_paths) -> None:
    # model_architecture_1 = inception_like(input_shape=dataset_info['image']['shape'],
    #                                       number_of_classes=dataset_info['label']['num_classes'])
    # model_architecture_2 = TransferLearningModel(model_type="inception_v3",
    #                                              input_shape=dataset_info['image']['shape'],
    #                                              number_of_classes=dataset_info['label']['num_classes'])
    # model_architecture_2 = model_architecture_2.expand_base_model_and_build_functional_model()
    # y = model_architecture_1(tf.ones(shape=(0, *dataset_info_custom['image']['shape'])))
    # input = tf.ones(shape=(0, *dataset_info['image']['shape']))
    # y = model_architecture_2(input)

    # print(model_architecture_2.input)
    # print(model_architecture_2.get_layer("conv5_block3_out").output)
    # print(model_architecture_2.output)

    grad_cam_obj = GradCAM(model_architecture, run_paths)
    grad_cam_obj.visualize(test_dataset)


def setup_checkpoint_loading(model_name, resume_checkpoint, resume_model_path, run_paths) -> str:
    '''Setup the continuing of a models training.
    If a checkpoint is given then this checkpoint is used for loading a model to continue training.
    If a path to a models directory is given then copy the summaries into the current models directory,
    also get the path of a checkpoint to load.
    This is either the latest checkpoint, if no checkpoint prefix is given in `resume_checkpoint`,
    or it is the checkpoint constructed from the path and the given prefix.

    Returns
    -------
    str
        Path to a checkpoint to load
    '''
    if not resume_model_path:  # resume_checkpoint is the path of a checkpoint file
        return resume_checkpoint

    else:  # resume_checkpoint is a checkpoint prefix
        summaries_directory = pathlib.Path(resume_model_path) / 'summaries'
        # Copy the summaries of the previous attempt
        shutil.copytree(src=str(summaries_directory), 
                        dst=run_paths['model_directories'][model_name]['summaries'],
                        dirs_exist_ok=True)

        if not resume_checkpoint:  # Load latest checkpoint
            return str(pathlib.Path(resume_model_path) / 'checkpoints')
        else:  # Load specified checkpoint
            return str(pathlib.Path(resume_model_path) / 'checkpoints' / resume_checkpoint)


def verify_configs (models) -> None:
    '''Simple tests if the given model configs. are of correct type/structure.
    '''
    # Verify the models type
    if isinstance(models, dict) and ('model' not in models):
       raise ValueError("The model dictionary should contain the key 'model' with a name.")
    
    elif isinstance(models, list):
        for model in models:
            if isinstance(model, dict) and ('model' not in model):
                raise ValueError("The model dictionary should contain the key 'model' with name.")
    
    else:
        raise ValueError("The model should be a dictionary or list of dictonaries.")

    # Verify configs
    if FLAGS.ensem and (not isinstance(models, list) or len(models) < 2):
        raise ValueError("For Ensemble Learning, train more than one model.")


@gin.configurable
def main(argv,
         models, use_hyperparameter_tuning, deep_visualization,         # <- configs
         resume_checkpoint, resume_model_path, evaluation_checkpoint):  # <- configs

    verify_configs(models)

    # Generate folder structures
    run_paths = utils_params.generate_run_directory()
    # Set loggers
    utils_misc.set_loggers(paths=run_paths, logging_level=logging.INFO)
    # Save gin config
    utils_params.save_config(run_paths['path_gin'], gin.config_str() )

    # Create dataset(s), returns the names of the available datasets.
    dataset_handles = DatasetLoader().create_datasets()

    if use_hyperparameter_tuning:
        logging.info("Hyper-parameter tuning set to True. Starting Hyper-parameter tuning...")
        # Delete the previous run directories inside the main 'hyperparameter_tuning' directory
        # utils_params.delete_previous_hyperparameter_runs(run_paths)

        # Now start the runs for all the hyperparameters.
        # All the run specific checkpoints and summaries will be saved under 'run_<datetime>'
        # under 'experiment' folder.
        hyperparameter_tuning(models, run_paths)

    elif FLAGS.train or FLAGS.eval or FLAGS.ensem:

        for index, model_configuration in enumerate(models):
            # Load datasets
            train_dataset, validation_dataset, test_dataset, dataset_info = \
                DatasetLoader().load_dataset(dataset_handle=model_configuration['dataset_handle'] )

            # Load model
            model_architecture = load_model(model_configuration, dataset_info, run_paths)
            
            # Log/display model configuration at start.
            model_config_string = '\n'.join( [f"'{key}': {model_configuration[key] }" 
                if type(model_configuration[key] ) is not str else f"'{key}': '{model_configuration[key] }'" for key in model_configuration.keys() ] )            
            logging.info( ('Current model:\n'
                          f"Model name: '{model_architecture.name}'\n"
                          + model_config_string) )
            
            utils_params.generate_model_directories(model_architecture.name, run_paths)

            resume_checkpoint_path = ''
            # Load checkpoint if this is the first model of the run.
            if (index == 0) and (resume_checkpoint or resume_model_path):
                resume_checkpoint_path = setup_checkpoint_loading(model_architecture.name, resume_checkpoint, resume_model_path, run_paths)

            last_checkpoint = ''
            if FLAGS.train:
                trainer = Trainer(model_architecture, train_dataset, validation_dataset, dataset_info, run_paths,
                                  resume_checkpoint=resume_checkpoint_path, class_weights_scale=model_configuration['class_weights_scale'] )
                last_checkpoint = trainer.train()

            if FLAGS.eval and not FLAGS.ensem:
                # no need to evaluate individual models here for FLAGS.ensem=True,
                # because if eFLAGS.ensem=True then it will be evaluated in the next
                # part of the code below for models individually as well as for
                # ensemble model
                if not FLAGS.train:  # Evaluate a saved model
                    last_checkpoint = evaluation_checkpoint

                _, _, _ = Evaluator(model_architecture, last_checkpoint, test_dataset, dataset_info, run_paths).evaluate()

            if deep_visualization:
                _deep_visualization(model_architecture, test_dataset, dataset_info, run_paths)

        if FLAGS.ensem:
            # Load datasets
            train_dataset, validation_dataset, test_dataset, dataset_info = DatasetLoader().load_dataset(
                dataset_handle=models[0]['dataset_handle'])

            models = load_model(models, dataset_info)
            ensemble = StackingEnsemble(models, None, dataset_info['label']['num_classes'],
                                        dataset_info, run_paths)
            level_0_loaded_models = ensemble.get_level_0_models()  # list of tuple (model_name, loaded_model)

            # for all the loaded models do an evaluation to see how it performs individually
            for model in level_0_loaded_models:
                model_name, model_architecture = model

                # restore_from_checkpoint is False here because this models are already loaded from
                # trained_models folder. So no need to restore it again
                _, _, _ = Evaluator(model_architecture, last_checkpoint, test_dataset, dataset_info,
                                    run_paths, restore_from_checkpoint=False).evaluate()

            ensemble_model = ensemble.get_stacking_ensemble_model()
            utils_params.generate_model_directories(ensemble_model.name, run_paths)

            resume_checkpoint_path = setup_checkpoint_loading(ensemble_model.name,
                                                              resume_checkpoint, resume_model_path, run_paths)

            ensemble_trainer = Trainer(ensemble_model, train_dataset, validation_dataset, dataset_info, run_paths,
                                       resume_checkpoint=resume_checkpoint_path, is_ensemble=True)
            last_checkpoint = ensemble_trainer.train()

            _, _, _ = Evaluator(ensemble_model, last_checkpoint, test_dataset, dataset_info, run_paths,
                                is_ensemble=True).evaluate()





if __name__ == '__main__':
    gin_config_path = pathlib.Path(__file__).parent / 'configs' / 'config.gin'
    gin.parse_config_files_and_bindings([gin_config_path], [])
    absl.app.run(main)
