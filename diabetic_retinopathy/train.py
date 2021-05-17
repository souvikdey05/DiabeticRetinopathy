import io
import os
import shutil
import gin
import tensorflow as tf
import enum
import logging
import pathlib
import matplotlib.pyplot as plt

from models.architectures import TransferLearningModel
from evaluation.metrics import ConfusionMatrix, ClassWiseMetric, AverageMetric, Metric, \
                               convert_AverageMetric_result, convert_ClassWiseMetric_result, \
                               format_metric_output, format_confusion_matrix_metric_output
from evaluation.visualization import GradCAM


training_logger = logging.getLogger('training')

Early_stopping_type = enum.Enum('Early_stopping_type', 'Relative Average')

EPOCH_COUNTER_PRINT_INTERVAL = 50


@gin.configurable
class Trainer(object):
    '''Train the given model on the given dataset.'''

    def __init__(self, model, training_dataset, validation_dataset, dataset_info, run_paths,
                 epochs, log_interval, checkpoint_interval, learning_rate, finetune_learning_rate, finetune_epochs,          # <- configs
                 patience, minimum_delta, early_stopping_test_interval, early_stopping_trigger_type, early_stopping_metric,  # <- configs
                 resume_checkpoint='', is_ensemble=False, class_weights_scale=0.0) -> None:
            
        model_directory = run_paths['model_directories'][model.name]

        self.model = model
        self.is_ensemble = is_ensemble

        self.train_dataset = training_dataset
        self.validation_dataset = validation_dataset

        self.epochs = epochs
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.epoch_counter = tf.Variable(initial_value=0, trainable=False) # Of type tf.Variable to allow checkpoint saving.

        # Summary Writer
        self.train_summary_writer = tf.summary.create_file_writer(logdir=str(model_directory['summaries'] / 'train'))
        self.validation_summary_writer = tf.summary.create_file_writer(
            logdir=str(model_directory['summaries'] / 'validation'))

        # Loss objective
        # Sparse means that the target labels are integers, not one-hot encoded arrays.
        # from_logits: If True then the input is expected to be not normalised,
        #   if a softmax activation function is used in the last layer, then the output is normalised,
        #   therefore from_logits is set to False.
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Weighted loss
        self.use_weighted_loss = (class_weights_scale != 0)

        if self.use_weighted_loss:
            class_weights = [class_weights_scale * weight for weight in dataset_info['training_dataset']['training_label_multipliers'] ]
            self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        
       
        # Fine-tuning
        self.finetune_learning_rate = finetune_learning_rate
        self.finetuning_started = False # Flag
        self.finetune_epochs = finetune_epochs

        self.num_classes = dataset_info['label']['num_classes']  # Number of classes in the output

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_confusion_matrix_metrics = ConfusionMatrix(num_classes=self.num_classes, name='train_confusion_matrix_metrics')
        self.train_class_wise_metrics = ClassWiseMetric(num_classes=self.num_classes, name='train_class_wise_metrics')
        self.train_average_metrics = AverageMetric(num_classes=self.num_classes, name='train_average_metrics')

        self.validation_loss = tf.keras.metrics.Mean(name='validation_loss')
        self.validation_confusion_matrix_metrics = ConfusionMatrix(num_classes=self.num_classes, name='validation_confusion_matrix_metrics')
        self.validation_class_wise_metrics = ClassWiseMetric(num_classes=self.num_classes, name='validation_class_wise_metrics')
        self.validation_average_metrics = AverageMetric(num_classes=self.num_classes, name='validation_average_metrics')

        # Checkpoint Manager        
        checkpoint_directory = model_directory['training_checkpoints']
        checkpoint = tf.train.Checkpoint(step_counter=self.epoch_counter, optimizer=self.optimizer, model=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint, max_to_keep=3, directory=checkpoint_directory)   
        if resume_checkpoint:
            if pathlib.Path(resume_checkpoint).is_dir():
                resume_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=resume_checkpoint)
            # Raises an exception if any existing Python objects in the dependency graph are unmatched.
            checkpoint.restore(resume_checkpoint).assert_existing_objects_matched()
            print('Restored from checkpoint: \'{}\''.format(resume_checkpoint) )
        else:
            print('No checkpoint given, initializing from scratch.')

        # Saved Model for the run
        self.saved_model_for_this_run_path = run_paths['saved_models_directory'] / self.model.name

        # Saved Model as a whole
        self.trained_model_path = run_paths['trained_models_directory'] / self.model.name

        # Early-stopping
        self.patience = patience
        self.minimum_delta = minimum_delta
        self.early_stopping_test_interval = early_stopping_test_interval
        self.early_stopping_trigger_type = Early_stopping_type(early_stopping_trigger_type + 1)
        self.early_stopping_metric = Metric.from_string(early_stopping_metric)
        early_stopping_checkpoint = tf.train.Checkpoint(step_counter=self.epoch_counter, model=self.model)
        self.early_stopping_checkpoint_manager = tf.train.CheckpointManager(checkpoint= early_stopping_checkpoint, max_to_keep=1, 
                                                                            directory=str(model_directory['early_stopping_checkpoints'] ) )
        self.last_successful_metric = self.best_metric = self.epochs_without_improvement = 0
        self.last_metrics = list()
                
        if self.early_stopping_test_interval > 0:
            logging.info(f'Early-stopping enabled, patience: {patience}, delta: {minimum_delta}.')
        else:
            logging.info('Early-stopping disabled.')


    @tf.function
    def train_step (self, x_batch_image, x_batch_label) -> None:

        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(x_batch_image, training=True)
            loss = self.loss_object(x_batch_label, predictions)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_confusion_matrix_metrics.update_state(x_batch_label, predictions)
        self.train_class_wise_metrics.update_state(x_batch_label, predictions)
        self.train_average_metrics.update_state(x_batch_label, predictions)


    @tf.function
    def train_step_with_weights (self, x_batch_image, x_batch_label, class_weights: tf.Tensor) -> None:
        '''Train a model with given features and labels; use `class_weights` to modify calculated losses.
        This function is identical to `self.train_step()` except for the weighted losses, it is created to avoid
        problems with retracing of the function.

        Parameters
        ----------
        x_batch_image : [type]
            [description]
        x_batch_label : [type]
            [description]
        class_weights : tf.Tensor
            [description]
        '''

        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(x_batch_image, training=True)
            loss = self.loss_object(x_batch_label, predictions, sample_weight=tf.gather(class_weights, x_batch_label) )

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_confusion_matrix_metrics.update_state(x_batch_label, predictions)
        self.train_class_wise_metrics.update_state(x_batch_label, predictions)
        self.train_average_metrics.update_state(x_batch_label, predictions)


    @tf.function
    def validation_step(self, x_batch_image, x_batch_label, early_stopping=False) -> None:
        '''Test the model by feeding it a batch of features with 'training' set to False.
        Also update the validation metrics.
        
        Parameters
        ----------               
        x_batch_image : tf.tensor
            A batch of images.
        
        x_batch_label : tf.tensor
            A batch of labels.

        early_stopping : boolean
            Whether to update all of the metrics or just the AverageMetric for testing early-stopping.
        '''
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(x_batch_image, training=False)
        validation_loss = self.loss_object(x_batch_label, predictions)

        if not early_stopping:
            self.validation_loss(validation_loss)
            self.validation_confusion_matrix_metrics.update_state(x_batch_label, predictions)
            self.validation_class_wise_metrics.update_state(x_batch_label, predictions)

        self.validation_average_metrics.update_state(x_batch_label, predictions)


    def test_early_stopping (self) -> bool:
        '''Test for early-stopping with the validation dataset.

        Two types of early-stopping tests are implemented:
        - Relative: Improve by a given value (`minimum_delta`) in a given number of epochs (`patience`).
        - Average: The average improvement has to reach a target value (`minimum_delta`) in a given number of epochs (`patience`).

        Returns
        -------
        bool
            True if early-stopping is triggered, False if not.
        '''
        for validation_image, _, validation_retina_grade, _ in self.validation_dataset:
            self.validation_step(validation_image, validation_retina_grade, early_stopping=True)
    
        current_metric = self.validation_average_metrics.calculate_macro_metric(self.early_stopping_metric)        
        self.validation_average_metrics.reset_states()

        early_stopping_criteria_satisfied = False
        if self.early_stopping_trigger_type == Early_stopping_type.Relative:
            if (current_metric - self.last_successful_metric) > self.minimum_delta:
                early_stopping_criteria_satisfied = True
        
        elif self.early_stopping_trigger_type == Early_stopping_type.Average:   
            self.last_metrics.append(current_metric - self.last_successful_metric)
            
            if (sum(self.last_metrics) / len(self.last_metrics) ) > self.minimum_delta:
                early_stopping_criteria_satisfied = True
                self.last_metrics = list()

        if current_metric > self.best_metric: # Always save the currently best model.
            self.best_metric = current_metric
            self.early_stopping_checkpoint_manager.save()

        if early_stopping_criteria_satisfied:
            self.last_successful_metric = current_metric
            self.epochs_without_improvement = 0                    
        else:        
            self.epochs_without_improvement += 1

            if self.epochs_without_improvement > self.patience:
                return True
        
        return False


    def logging_and_summaries (self, epoch_counter) -> None:
        train_loss_result = self.train_loss.result()
        train_confusion_matrix_results = self.train_confusion_matrix_metrics.result()

        train_class_wise_metric_results = self.train_class_wise_metrics.result()
        train_class_wise_metric_results = convert_ClassWiseMetric_result(train_class_wise_metric_results, self.num_classes)
        train_class_wise_metric_results_string = format_metric_output(train_class_wise_metric_results)
        
        train_average_metric_results = self.train_average_metrics.result()
        train_average_metric_results = convert_AverageMetric_result(train_average_metric_results)
        train_average_metric_results_string = format_metric_output(train_average_metric_results)

        validation_loss_result = self.validation_loss.result()
        validation_confusion_matrix_results = self.validation_confusion_matrix_metrics.result()

        validation_class_wise_metric_results = self.validation_class_wise_metrics.result()
        validation_class_wise_metric_results = convert_ClassWiseMetric_result(validation_class_wise_metric_results, self.num_classes)
        validation_class_wise_metric_results_string = format_metric_output(validation_class_wise_metric_results)
        
        validation_average_metric_results = self.validation_average_metrics.result()
        validation_average_metric_results = convert_AverageMetric_result(validation_average_metric_results)
        validation_average_metric_results_string = format_metric_output(validation_average_metric_results)

        template = ('Step {} ->, \n'
                    'Train Loss: {:.3f}, \n'
                    'Train Class Wise Metric: {}, \n'
                    'Train Average Metric: {}, \n'
                    'Validation Loss: {:.3f}, \n'
                    'Validation Class Wise Metric: {}, \n'
                    'Validation Average Metric: {} \n'
                    '*******************************************')
        training_logger.info(template.format(epoch_counter,
                                             train_loss_result,
                                             train_class_wise_metric_results_string,
                                             train_average_metric_results_string,
                                             validation_loss_result,
                                             validation_class_wise_metric_results_string,
                                             validation_average_metric_results_string))
        training_logger.info(train_confusion_matrix_results)
        training_logger.info(validation_confusion_matrix_results)
        training_logger.info("//////////////////////////////////////////////")

        train_confusion_matrix_fig = format_confusion_matrix_metric_output(self.num_classes, train_confusion_matrix_results)

        # Write summary to tensorboard
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss_result, step=epoch_counter)

            for name, value in train_class_wise_metric_results.items():
                tf.summary.scalar(name, value, step=epoch_counter)

            for name, value in train_average_metric_results.items():
                tf.summary.scalar(name, value, step=epoch_counter)

            tf.summary.image('confusion matrix after {} epochs'.format(epoch_counter), self._plot_to_image(train_confusion_matrix_fig), step=epoch_counter)

        validation_confusion_matrix_fig = format_confusion_matrix_metric_output(self.num_classes,
                                                                                validation_confusion_matrix_results)

        with self.validation_summary_writer.as_default():
            tf.summary.scalar('loss', validation_loss_result, step=epoch_counter)

            for name, value in validation_class_wise_metric_results.items():
                tf.summary.scalar(name, value, step=epoch_counter)

            for name, value in validation_average_metric_results.items():
                tf.summary.scalar(name, value, step=epoch_counter)

            tf.summary.image('confusion matrix after {} epochs'.format(epoch_counter), self._plot_to_image(validation_confusion_matrix_fig), step=epoch_counter)


    def train(self) -> str:
        '''
        Returns
        -------
        str
            Absolute path to the checkpoint of the finished model.
        '''
        
        if self.finetuning_started:
            training_logger.info(f'Start fine-tune training for model {self.model.name} for {self.finetune_epochs} epochs, til epoch {self.epochs}.')
        else:
            training_logger.info(f'Start training for model {self.model.name} for {self.epochs} epochs')

        # Reset the Train Metrics
        self.train_loss.reset_states()
        self.train_confusion_matrix_metrics.reset_states()
        self.train_class_wise_metrics.reset_states()
        self.train_average_metrics.reset_states()


        if self.use_weighted_loss:
            class_weights = self.class_weights # Create variable to avoid retracing of self.train_step().



        for x_batch_image, _, x_batch_retina_grade, _ in self.train_dataset:

            epoch_counter = int(self.epoch_counter.assign_add(1) )

            if self.is_ensemble:
                x_batch_image = [x_batch_image for _ in range(len(self.model.input) ) ]

            if epoch_counter % EPOCH_COUNTER_PRINT_INTERVAL == 0:
                tf.print(f"Epoch: {epoch_counter} / {self.epochs} ===>")




            if self.use_weighted_loss: # Two functions to avoid problems with retracing.
                self.train_step_with_weights(x_batch_image, x_batch_retina_grade, self.class_weights)
            else:
                self.train_step(x_batch_image, x_batch_retina_grade)




            if epoch_counter % self.log_interval == 0:
                # Reset the Validation Metrics
                self.validation_loss.reset_states()
                self.validation_confusion_matrix_metrics.reset_states()
                self.validation_class_wise_metrics.reset_states()
                self.validation_average_metrics.reset_states()

                for validation_image, _, validation_retina_grade, _ in self.validation_dataset:
                    if self.is_ensemble:
                        validation_image = [validation_image for _ in range(len(self.model.input))]
                    self.validation_step(validation_image, validation_retina_grade)
                self.logging_and_summaries(epoch_counter)                


            if (self.early_stopping_test_interval > 0) and ( (epoch_counter % self.early_stopping_test_interval) == 0):
                if self.test_early_stopping():
                    logging.info(f'Early-stopping is triggered at epoch {epoch_counter}.')

                    if (not self.finetuning_started) and (self.finetune_epochs > 0) and (isinstance(self.model, TransferLearningModel) ): # Only fine-tune once.
                        return self.finetune()
                    else:
                        return self.early_stopping_checkpoint_manager.latest_checkpoint                    


            if epoch_counter % self.checkpoint_interval == 0:  # Save checkpoint
                try:
                    checkpoint_path = self.checkpoint_manager.save()
                    logging.info(f"Saving checkpoint: '{checkpoint_path}'")
                except:
                    logging.error(f"Failed to save checkpoint at checkpoint interval")


            if epoch_counter % self.epochs == 0:
                training_logger.info(f"Finished training after {epoch_counter} epochs.")
                
                if (not self.finetuning_started) and (self.finetune_epochs > 0) and (isinstance(self.model, TransferLearningModel) ): # Only fine-tune once.
                    return self.finetune()
                else:
                    return self.save()


    def save (self) -> str:
        '''Save the last checkpoint and the trained model.

        Returns
        -------
        str
            Path of the last checkpoint.
        '''
        try:
            # Save the final checkpoint.
            last_checkpoint = self.checkpoint_manager.save()            
            training_logger.info(f"Last checkpoint: '{last_checkpoint}'")
        except:
            last_checkpoint = None
            logging.error(f"Failed to save last checkpoint")

        try:
            # Save final model for this run
            tf.train.Checkpoint(model=self.model).save(file_prefix=str(self.saved_model_for_this_run_path))
            logging.info(f"Final model for this run saved at '{str(self.saved_model_for_this_run_path)}'")
        except:
            logging.error(f"Failed to save final model for this run at '{str(self.saved_model_for_this_run_path)}'")

        if self.trained_model_path.exists():
            # deleting previous trained model files that were saved
            for filename in os.listdir(str(self.trained_model_path)):
                file_path = str(self.trained_model_path / filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
        else:
            self.trained_model_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save final model as a whole
            self.trained_model_path = self.trained_model_path / self.model.name
            tf.train.Checkpoint(model=self.model).save(file_prefix=str(self.trained_model_path))
            logging.info(f"Trained model saved at '{str(self.trained_model_path)}'")
        except:
            logging.error(f"Failed to save trained model at '{str(self.trained_model_path)}'")
        
        return last_checkpoint


    def finetune (self) -> str:
        '''
        Returns
        -------
        str
            Absolute path to the checkpoint of the finished model.
        '''

        if not isinstance(self.model, TransferLearningModel):
            raise TypeError(f"Fine-tuning is only possible with models of type 'TransferLearningModel', received model type: '{type(self.model) }'.")
        
        logging.info('Initialising fine-tuning of the model...')

        # Unfreezing of the top layers allows layers outside of the dense layers at the end of the model to learn. 
        self.model.unfreeze_top_layers()

        # Set flag.
        self.finetuning_started = True
        
        # Change the learning rate of the optimiser.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.finetune_learning_rate)

        # Update the epoch target
        self.epochs = self.epochs + self.finetune_epochs

        return self.train()


    def _plot_to_image(self, figure) -> tf.Tensor:
        '''Converts the matplotlib plot specified by 'figure' to a PNG image and returns it.
        The supplied figure is closed and inaccessible after this call.'''

        buffer = io.BytesIO()

        # Use plt.savefig to save the plot to a PNG in memory.
        plt.savefig(buffer, format='png')

        # Closing the figure prevents it from being displayed directly inside the notebook.
        plt.close(figure)
        buffer.seek(0)

        # Use tf.image.decode_png to convert the PNG buffer
        # to a TF image. Make sure you use 4 channels.
        image = tf.image.decode_png(buffer.getvalue(), channels=4)

        # Use tf.expand_dims to add the batch dimension
        image = tf.expand_dims(image, 0)

        return image
