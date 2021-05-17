import logging
import gin
# from ray import tune
import tensorflow as tf
from models.architectures import TransferLearningModel, inception_like, vgg_like
from train import Trainer
from utils import utils_params, utils_misc

from tensorboard.plugins.hparams import api as hp
from evaluation.evaluation import Evaluator
from input_pipeline.dataset_loader import DatasetLoader


# def train_func(config):
#     # Hyperparameters
#     bindings = []
#     for key, value in config.items():
#         bindings.append(f'{key}={value}')
#
#     # generate folder structures
#     run_paths = utils_params.generate_run_directory(','.join(bindings) )
#
#     # set loggers
#     utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
#
#     # gin-config
#     gin.parse_config_files_and_bindings(['/mnt/home/repos/dl-lab-skeleton/diabetic_retinopathy/configs/config.gin'], bindings)
#     utils_params.save_config(run_paths['path_gin'], gin.config_str())
#
#     # setup pipeline
#     ds_train, ds_val, ds_test, ds_info = load()
#
#     # model
#     model = vgg_like(input_shape=ds_info.features["image"].shape, n_classes=ds_info.features["label"].num_classes)
#
#     trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
#     for val_accuracy in trainer.train():
#         tune.report(val_accuracy=val_accuracy)
#
#
# analysis = tune.run(
#     train_func, num_samples=2, resources_per_trial={'gpu': 1, 'cpu': 4},
#     config={
#         "Trainer.total_steps": tune.grid_search([1e4]),
#         "vgg_like.base_filters": tune.choice([8, 16]),
#         "vgg_like.n_blocks": tune.choice([2, 3, 4, 5]),
#         "vgg_like.dense_units": tune.choice([32, 64]),
#         "vgg_like.dropout_rate": tune.uniform(0, 0.9),
#     })
#
# print("Best config: ", analysis.get_best_config(metric="val_accuracy"))
#
# # Get a dataframe for analyzing trial results.
# df = analysis.dataframe()


@gin.configurable
def hyperparameter_tuning(models, run_paths,
                          transfer_learning_params, inception_like_params, vgg_like_params):  # <- configs
    """Train and evaluate the given models with varying hyperparameter."""

    transfer_learning_models = [*TransferLearningModel.model_types]
    custom_models = ['vgg_like', 'inception_like']

    for model in models:
        model_name = model['model']  # this is the model dict defined in config

        # Make new folder inside 'hyperparameter_tuning' folder based on model names
        utils_params.generate_hyperparameter_model_directories(model_name, run_paths)

        # Setup summary writer
        hyperparameter_logs_directory = run_paths['hyperparameter_tuning_directory'] / model_name
        hyperparameter_summary_writer = tf.summary.create_file_writer(logdir=str(hyperparameter_logs_directory))

        # Metric for hyperparameter comparison that is common among all the models
        HP_METRIC_LOSS = 'loss'
        HP_METRIC_ACCURACY = 'accuracy'

        logging.info(f"Starting hyper-parameter tuning for model {model_name} ->")
        if model_name in transfer_learning_models:
            # Setup Tensorboard hyperparameters
            HP_DATASET = hp.HParam(name='dataset',
                                   domain=hp.Discrete([dataset for dataset in transfer_learning_params['dataset_handle']]))
            HP_DROPOUT_RATE = hp.HParam(name='dropout_rate',
                                        domain=hp.RealInterval(min_value=transfer_learning_params['dropout_rate']['min'],
                                                               max_value=transfer_learning_params['dropout_rate']['max']))

            # Save Tensorboard hyperparameter configuration
            with hyperparameter_summary_writer.as_default():
                hp.hparams_config(
                    hparams=[HP_DATASET, HP_DROPOUT_RATE],
                    metrics=[hp.Metric(tag=HP_METRIC_LOSS, display_name='Loss'),
                             hp.Metric(tag=HP_METRIC_ACCURACY, display_name='Accuracy')]
                )

            # Generate 10 random searches in the HP_DROPOUT_RATE.domain
            dropout_rate_list = []
            dropout_rate_n_search = transfer_learning_params['dropout_rate']['n_search']
            count = 0
            while count < dropout_rate_n_search:
                dropout_rate_uniform = HP_DROPOUT_RATE.domain.sample_uniform()
                dropout_rate_uniform = round(dropout_rate_uniform, 2)
                if dropout_rate_uniform not in dropout_rate_list:
                    dropout_rate_list.append(dropout_rate_uniform)
                    count += 1

            session_num = 0
            # Train with range of hyperparameters
            for dataset in HP_DATASET.domain.values:
                # Load the dataset.
                train_dataset, validation_dataset, test_dataset, dataset_info = DatasetLoader().load_dataset(
                    dataset_handle=dataset)
                for dropout_rate in dropout_rate_list:
                    # Note: dropout rate is selected using random search
                    hparams = {
                        HP_DATASET: dataset,
                        HP_DROPOUT_RATE: dropout_rate,
                    }
                    trail_name = "trail-%d" % session_num
                    logging.info('--- Starting hyperparameter trial: %s' % trail_name)
                    logging.info({h.name: hparams[h] for h in hparams})
                    hyperparameter_summary_writer = tf.summary.create_file_writer(
                        str(hyperparameter_logs_directory / trail_name))

                    # Create the model with different drop_rates
                    first_fine_tune_layer = None
                    if 'first_fine_tune_layer' in model:
                        first_fine_tune_layer = model['first_fine_tune_layer']

                    # print(f"dataset_info = {dataset_info}")
                    model_architecture = TransferLearningModel(model_type=model_name,
                                                               input_shape=dataset_info['image']['shape'],
                                                               number_of_classes=dataset_info['label']['num_classes'],
                                                               first_fine_tune_layer=first_fine_tune_layer,
                                                               dropout_rate=hparams[HP_DROPOUT_RATE])

                    utils_params.generate_model_directories(model_architecture.name, run_paths)

                    trainer = Trainer(model_architecture, train_dataset, validation_dataset, dataset_info, run_paths)
                    last_checkpoint = trainer.train()
                    loss, accuracy, completed_epochs = Evaluator(model_architecture, last_checkpoint, test_dataset, 
                                                                 dataset_info, run_paths).evaluate()

                    # Log the used hyperparameters and results
                    with hyperparameter_summary_writer.as_default():
                        hp.hparams(hparams=hparams)
                        tf.summary.scalar(name=HP_METRIC_LOSS, data=loss, step=completed_epochs)  # step is required.
                        tf.summary.scalar(name=HP_METRIC_ACCURACY, data=accuracy,
                                          step=completed_epochs)  # step is required.

                    session_num += 1

        elif model_name in custom_models:
            if model_name == 'inception_like':
                HP_DATASET = hp.HParam(name='dataset',
                                       domain=hp.Discrete([dataset for dataset in inception_like_params['dataset_handle']]))
                HP_DROPOUT_RATE = hp.HParam(name='dropout_rate',
                                            domain=hp.RealInterval(min_value=inception_like_params['dropout_rate']['min'],
                                                                    max_value=inception_like_params['dropout_rate']['max']))
                HP_MOD_1_FILTER_UNITS = hp.HParam('module_1_filters', hp.Discrete(inception_like_params['module_1_filters']))
                HP_MOD_2_FILTER_UNITS = hp.HParam('module_2_filters', hp.Discrete(inception_like_params['module_2_filters']))
                HP_KERNEL_INITIALIZER_UNITS = hp.HParam('kernel_initializer',
                                                  hp.Discrete(inception_like_params['kernel_initializer']))
                HP_DENSE_NUM_UNITS = hp.HParam('dense_num_units',
                                               domain=hp.IntInterval(min_value=inception_like_params['dense_num_units']['min'],
                                                                     max_value=inception_like_params['dense_num_units']['max']))

                # Generate random searches in the HP_DENSE_NUM_UNITS.domain
                dense_num_units_list = []
                dense_num_units_n_search = inception_like_params['dense_num_units']['n_search']
                count = 0
                while count < dense_num_units_n_search:
                    dense_num_units_uniform = HP_DENSE_NUM_UNITS.domain.sample_uniform()
                    dense_num_units_uniform = int(dense_num_units_uniform)
                    if dense_num_units_uniform not in dense_num_units_list:
                        dense_num_units_list.append(dense_num_units_uniform)
                        count += 1

                # Generate random searches in the HP_DROPOUT_RATE.domain
                dropout_rate_list = []
                dropout_rate_n_search = inception_like_params['dropout_rate']['n_search']
                count = 0
                while count < dropout_rate_n_search:
                    dropout_rate_uniform = HP_DROPOUT_RATE.domain.sample_uniform()
                    dropout_rate_uniform = round(dropout_rate_uniform, 2)
                    if dropout_rate_uniform not in dropout_rate_list:
                        dropout_rate_list.append(dropout_rate_uniform)
                        count += 1

                # Save Tensorboard hyperparameter configuration
                with hyperparameter_summary_writer.as_default():
                    hp.hparams_config(
                        hparams=[HP_DATASET, HP_MOD_1_FILTER_UNITS, HP_MOD_2_FILTER_UNITS, HP_KERNEL_INITIALIZER_UNITS,
                                 HP_DENSE_NUM_UNITS, HP_DROPOUT_RATE],
                        metrics=[hp.Metric(tag=HP_METRIC_LOSS, display_name='Loss'),
                                 hp.Metric(tag=HP_METRIC_ACCURACY, display_name='Accuracy')]
                    )

                session_num = 0
                # Train with range of hyperparameters
                for dataset in HP_DATASET.domain.values:
                    # Load the dataset.
                    train_dataset, validation_dataset, test_dataset, dataset_info = DatasetLoader().load_dataset(
                        dataset_handle=dataset)
                    for module_1_filter in HP_MOD_1_FILTER_UNITS.domain.values:
                        for module_2_filter in HP_MOD_2_FILTER_UNITS.domain.values:
                            for kernel_initializer in HP_KERNEL_INITIALIZER_UNITS.domain.values:
                                for dense_num_units in dense_num_units_list:
                                    # Note: number of dense units is selected using random search
                                    for dropout_rate in dropout_rate_list:
                                        # Note: dropout rate is selected using random search
                                        hparams = {
                                            HP_DATASET: dataset,
                                            HP_MOD_1_FILTER_UNITS: module_1_filter,
                                            HP_MOD_2_FILTER_UNITS: module_2_filter,
                                            HP_KERNEL_INITIALIZER_UNITS: kernel_initializer,
                                            HP_DENSE_NUM_UNITS: dense_num_units,
                                            HP_DROPOUT_RATE: dropout_rate,
                                        }
                                        trail_name = "trail-%d" % session_num
                                        logging.info('--- Starting hyperparameter trial: %s' % trail_name)
                                        logging.info({h.name: hparams[h] for h in hparams})
                                        hyperparameter_summary_writer = tf.summary.create_file_writer(
                                            str(hyperparameter_logs_directory / trail_name))

                                        # Create the model with different hyperparameters
                                        model_architecture = inception_like(input_shape=dataset_info['image']['shape'],
                                                                            number_of_classes=dataset_info['label'][
                                                                                'num_classes'],
                                                                            # 6 is hardcoded here,
                                                                            # 6 is the number of filters in 'inception_module'
                                                                            # Using the same filter size for all the 6 filters.
                                                                            # This can be modified later if we have time for the project

                                                                            module_1_filters=[hparams[
                                                                                                  HP_MOD_1_FILTER_UNITS]] * 6,
                                                                            module_2_filters=[hparams[
                                                                                                  HP_MOD_2_FILTER_UNITS]] * 6,
                                                                            kernel_initializer=hparams[HP_KERNEL_INITIALIZER_UNITS],
                                                                            dropout_rate=hparams[HP_DROPOUT_RATE],
                                                                            number_of_dense_units=hparams[HP_DENSE_NUM_UNITS])

                                        utils_params.generate_model_directories(model_architecture.name, run_paths)

                                        trainer = Trainer(model_architecture, train_dataset, validation_dataset, dataset_info, run_paths)
                                        last_checkpoint = trainer.train()
                                        loss, accuracy, completed_epochs = Evaluator(model_architecture, last_checkpoint, test_dataset,
                                                                                     dataset_info, run_paths).evaluate()

                                        # Log the used hyperparameters and results
                                        with hyperparameter_summary_writer.as_default():
                                            hp.hparams(hparams=hparams)
                                            tf.summary.scalar(name=HP_METRIC_LOSS, data=loss,
                                                              step=completed_epochs)  # step is required.
                                            tf.summary.scalar(name=HP_METRIC_ACCURACY, data=accuracy,
                                                              step=completed_epochs)  # step is required.

                                        session_num += 1

            elif model_name == 'vgg_like':
                HP_DATASET = hp.HParam(name='dataset',
                                       domain=hp.Discrete([dataset for dataset in vgg_like_params['dataset_handle']]))
                HP_DROPOUT_RATE = hp.HParam(name='dropout_rate',
                                            domain=hp.RealInterval(
                                                min_value=vgg_like_params['dropout_rate']['min'],
                                                max_value=vgg_like_params['dropout_rate']['max']))
                HP_BLOCK_NUM_UNITS = hp.HParam('block_num_units', hp.Discrete(vgg_like_params['block_num_units']))  # [2, 4, 8, 16, 32, 64, 128]
                HP_BASE_FILTER_UNITS = hp.HParam('base_filters', hp.Discrete(vgg_like_params['base_filters']))  # [2, 4, 8, 16, 32]
                HP_DENSE_NUM_UNITS = hp.HParam('dense_num_units',
                                               domain=hp.IntInterval(min_value=vgg_like_params['dense_num_units']['min'],
                                                                     max_value=vgg_like_params['dense_num_units']['max']))

                # Generate random searches in the HP_DENSE_NUM_UNITS.domain
                dense_num_units_list = []
                dense_num_units_n_search = vgg_like_params['dense_num_units']['n_search']
                count = 0
                while count < dense_num_units_n_search:
                    dense_num_units_uniform = HP_DENSE_NUM_UNITS.domain.sample_uniform()
                    dense_num_units_uniform = int(dense_num_units_uniform)
                    if dense_num_units_uniform not in dense_num_units_list:
                        dense_num_units_list.append(dense_num_units_uniform)
                        count += 1

                # Generate random searches in the HP_DROPOUT_RATE.domain
                dropout_rate_list = []
                dropout_rate_n_search = vgg_like_params['dropout_rate']['n_search']
                count = 0
                while count < dropout_rate_n_search:
                    dropout_rate_uniform = HP_DROPOUT_RATE.domain.sample_uniform()
                    dropout_rate_uniform = round(dropout_rate_uniform, 2)
                    if dropout_rate_uniform not in dropout_rate_list:
                        dropout_rate_list.append(dropout_rate_uniform)
                        count += 1

                # Save Tensorboard hyperparameter configuration
                with hyperparameter_summary_writer.as_default():
                    hp.hparams_config(
                        hparams=[HP_DATASET, HP_BLOCK_NUM_UNITS, HP_BASE_FILTER_UNITS,
                                 HP_DENSE_NUM_UNITS, HP_DROPOUT_RATE],
                        metrics=[hp.Metric(tag=HP_METRIC_LOSS, display_name='Loss'),
                                 hp.Metric(tag=HP_METRIC_ACCURACY, display_name='Accuracy')]
                    )

                session_num = 0
                # Train with range of hyperparameters
                for dataset in HP_DATASET.domain.values:
                    # Load the dataset.
                    train_dataset, validation_dataset, test_dataset, dataset_info = DatasetLoader().load_dataset(
                        dataset_handle=dataset)

                    for block_num_unit in HP_BLOCK_NUM_UNITS.domain.values:
                        for base_filter in HP_BASE_FILTER_UNITS.domain.values:
                            for dense_num_units in dense_num_units_list:
                                # Note: number of dense units is selected using random search
                                for dropout_rate in dropout_rate_list:
                                    # Note: dropout rate is selected using random search
                                    hparams = {
                                        HP_DATASET: dataset,
                                        HP_BLOCK_NUM_UNITS: block_num_unit,
                                        HP_BASE_FILTER_UNITS: base_filter,
                                        HP_DENSE_NUM_UNITS: dense_num_units,
                                        HP_DROPOUT_RATE: dropout_rate,
                                    }
                                    trail_name = "trail-%d" % session_num
                                    logging.info('--- Starting hyperparameter trial: %s' % trail_name)
                                    logging.info({h.name: hparams[h] for h in hparams})
                                    hyperparameter_summary_writer = tf.summary.create_file_writer(
                                        str(hyperparameter_logs_directory / trail_name))

                                    # Create the model with different hyperparameters
                                    model_architecture = vgg_like(input_shape=dataset_info['image']['shape'],
                                                                  number_of_classes=dataset_info['label']['num_classes'],
                                                                  n_blocks=hparams[HP_BLOCK_NUM_UNITS],
                                                                  base_filters=hparams[HP_BASE_FILTER_UNITS],
                                                                  dropout_rate=hparams[HP_DROPOUT_RATE],
                                                                  number_of_dense_units=hparams[HP_DENSE_NUM_UNITS])

                                    utils_params.generate_model_directories(model_architecture.name, run_paths)

                                    trainer = Trainer(model_architecture, train_dataset, validation_dataset, dataset_info, run_paths)
                                    last_checkpoint = trainer.train()
                                    loss, accuracy, completed_epochs = Evaluator(model_architecture, last_checkpoint, test_dataset,
                                                                                 dataset_info, run_paths).evaluate()

                                    # Log the used hyperparameters and results
                                    with hyperparameter_summary_writer.as_default():
                                        hp.hparams(hparams=hparams)
                                        tf.summary.scalar(name=HP_METRIC_LOSS, data=loss,
                                                          step=completed_epochs)  # step is required.
                                        tf.summary.scalar(name=HP_METRIC_ACCURACY, data=accuracy,
                                                          step=completed_epochs)  # step is required.

                                    session_num += 1