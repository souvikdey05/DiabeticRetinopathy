import gin
import os
import pathlib
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import typing

from input_pipeline.preprocessing import preprocess_tensorflow_dataset, augment_tensorflow_dataset, \
                                         preprocess_augment, preprocess_resize, \
                                         split_training_dataset, split_training_dataset_for_sampling, \
                                         oversample, undersample, get_dataset_size, join_dataset

# Flag to display the dataset distibution after dataset creation.
PRINT_DATASET_DISTRIBUTIONS = False


@gin.configurable
class DatasetLoader:
    '''A class to load, sample, parse and save datasets.'''

    def __init__(self, dataset_name, dataset_directory, tfrecords_directory, dataset_specifications,  # <- configs.
                 training_dataset_ratio, tfrecords_operation, caching, batch_size                     # <- configs.
                 ) -> None:
        '''Parameters
        ----------
        dataset_name : str
            Name of the dataset. Supported datasets: 'idrid', 'eyepacs' and 'mnist'.
        dataset_directory : str
            Path to where the dataset data is stored.
        tfrecords_directory : str
            Destination for tfrecords files.
        dataset_specifications : list of dict
            A list containing dicts containing information about how a dataset should be sampled;
            if multiple dicts are listed then multiple variations of the dataset are created.            
            dict has to contain the keys: 'name', 'undersample_ratio' and 'oversample_ratio'.
        training_dataset_ratio : float
            Ratio discribing how to split the training dataset into the traning and validation datasets.
        tfrecords_operation : str
            Choices are 'create' and 'read':
            - 'read': Read the already created tfrecords files from `tfrecords_directory`.
            - 'create': Create tfrecords files from the dataset files in `dataset_directory`.
        caching : bool
            Weather to use caching during dataset creation.
        batch_size : int
            Batch size of the created dataset.
        '''

        self.dataset_name = dataset_name
        accepted_dataset_names = ('idrid', 'eyepacs', 'mnist')
        if self.dataset_name not in accepted_dataset_names:
            raise ValueError(
                f"Received invalid dataset name: '{self.dataset_name}', accepted dataset names: {accepted_dataset_names}")

        self.dataset_directory = pathlib.Path(dataset_directory)
        if (not self.dataset_directory.exists() ) or (not self.dataset_directory.is_dir() ):
            raise ValueError(f"Received invalid dataset directory: '{self.dataset_directory}'.")
        
        self.tfrecords_directory = pathlib.Path(tfrecords_directory)
        if not self.tfrecords_directory.exists():
            self.tfrecords_directory.mkdir(parents=True, exist_ok=True)

        if isinstance(dataset_specifications, list) and all( [isinstance(spec, dict) for spec in dataset_specifications] ):
            self.dataset_specifications = dataset_specifications
        elif isinstance(dataset_specifications, dict):
            self.dataset_specifications = [dataset_specifications]
        elif (dataset_specifications is None) or (isinstance(dataset_specifications, list) and not any(dataset_specifications) ):
            self.dataset_specifications = list()
        else:
            raise ValueError(f"Received invalid dataset specifications.")

        if isinstance(training_dataset_ratio, float) and (0.0 < training_dataset_ratio <= 1.0):
            self.training_dataset_ratio = training_dataset_ratio
        else:
            raise ValueError(
                f'The training dataset split ratio has to be: 0.0 < ratio <= 1.0. Received ratio: {training_dataset_ratio}')

        self.tfrecords_operation = tfrecords_operation
        valid_tfrecords_operations = ('create', 'read')
        if self.tfrecords_operation not in valid_tfrecords_operations:
            raise ValueError(
                f"Received invalid tfrecords operation: {self.tfrecords_operation}, accepted operations: {valid_tfrecords_operations}")

        self.caching = caching
        self.batch_size = batch_size


    def create_datasets(self) -> typing.Tuple[str]:
        '''Create datasets.
        - For IDRID: Parse the data and create tfrecord files.
        - For EYEPACS and MNIST: Do nothing, the files are just loaded in `self.load_dataset()`.

        Returns
        -------
        Tuple of string
            List of directories in `self.tfrecords_directory`, these are the names of created datasets
            which can be loaded with `self.load_dataset()`.
        '''
        
        if self.dataset_name == 'idrid':
            if self.tfrecords_operation == 'read':
                logging.info("Selected tfrecords operation is 'read', no datasets are created.")

            else:
                training_dataset, test_dataset = self._create_idrid_dataset()

                for dataset_specification in self.dataset_specifications:
                    logging.info(f"Preparing dataset '{dataset_specification['name']}'...")

                    # Copy the unedited dataset to use it as the base for each variation.
                    training_dataset_copy = training_dataset

                    training_dataset_copy, validation_dataset = self._sample_dataset(dataset_specification['undersample_ratio'],
                                                                                     dataset_specification['oversample_ratio'],
                                                                                     training_dataset_copy,
                                                                                     test_dataset)

                    self._write_tfrecords(training_dataset_copy, validation_dataset, test_dataset, dataset_specification['name'] )

        elif self.dataset_name == 'eyepacs':
            logging.info("Sampling is not implemented for the 'eyepacs' dataset.")

        elif self.dataset_name == 'mnist':
            logging.info("Sampling is not implemented for the 'MNIST' dataset.")

        return self._get_dataset_handles()


    def _create_idrid_dataset(self) -> typing.Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        '''Read image and label files from data directory.

        Returns
        -------
        tuple of two tf.data.Dataset
            The training and test dataset, the training set is not yet split.
        '''
        logging.info(f"Preparing file loading from {self.dataset_directory}...")

        # Parse the CSV files
        labels_base_path = self.dataset_directory / 'labels'
        train_label_data_dir = labels_base_path / "train.csv"
        test_label_data_dir = labels_base_path / "test.csv"

        column_names = ["Image name", "Retinopathy grade", "Risk of macular edema"]
        train_labels_df = pd.read_csv(train_label_data_dir, names=column_names, usecols=column_names)
        train_labels_df = train_labels_df.iloc[1:]  # Removing the first row as it contains header

        test_labels_df = pd.read_csv(test_label_data_dir, names=column_names, usecols=column_names)
        test_labels_df = test_labels_df.iloc[1:]  # Removing the first row as it contains header

        image_base_path = self.dataset_directory / 'images'
        training_dataset = tf.data.Dataset.list_files(str(image_base_path / 'train' / '*.jpg')).cache()
        test_dataset = tf.data.Dataset.list_files(str(image_base_path / 'test' / '*.jpg')).cache()

        def _get_labels(file_path, label_dataframe):
            # Find labels from file_path
            parts = tf.strings.split(file_path, sep=os.sep)
            image_name = tf.strings.split(parts[-1], sep=".")[0]
            image_name_decoded = image_name.numpy().decode("utf-8")

            current_image_label_df = label_dataframe.loc[label_dataframe[column_names[0]] == image_name_decoded]
            retina_grade = int(current_image_label_df.iloc[0][column_names[1]])
            risk_grade = int(current_image_label_df.iloc[0][column_names[2]])

            return image_name, retina_grade, risk_grade

        def _decode_img(file_path):
            # load the raw data from the file as a string
            image = tf.io.read_file(file_path)

            # convert the compressed string to a 3D uint8 tensor
            image = tf.io.decode_jpeg(image, channels=3)

            # do we pre-process
            image = preprocess_resize(image=image)

            # Can't normalize here, because if we store in TFRecords, it
            # is easy to encode the jpeg again to string

            return image

        def _process_train_image_path(file_path: tf.Tensor):
            # find out the label
            image_name, retina_grade, risk_grade = _get_labels(file_path, train_labels_df)

            # decode the image
            img = _decode_img(file_path)

            return img, image_name, retina_grade, risk_grade

        def _process_test_image_path(file_path: tf.Tensor):
            # find out the label
            image_name, retina_grade, risk_grade = _get_labels(file_path, test_labels_df)

            # decode the image
            img = _decode_img(file_path)

            return img, image_name, retina_grade, risk_grade

        # Set num_parallel_calls so multiple images are loaded/processed in parallel.
        training_dataset = training_dataset.map(
            lambda x: tf.py_function(func=_process_train_image_path, inp=[x],
                                     Tout=[tf.float32, tf.string, tf.int8, tf.int8])
        ).cache()
        test_dataset = test_dataset.map(
            lambda x: tf.py_function(func=_process_test_image_path, inp=[x],
                                     Tout=[tf.float32, tf.string, tf.int8, tf.int8])
        ).cache()

        return training_dataset, test_dataset


    def _sample_dataset(self, undersample_ratio, oversample_ratio, training_dataset, test_dataset) -> typing.Tuple[
        tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        '''Split the training dataset and sample the datasets with the given specifications. 

        Parameters
        ----------
        undersample_ratio : float
            [description]
        oversample_ratio : float
            [description]
        training_dataset : tf.data.Dataset
            [description]
        test_dataset : tf.data.Dataset
            [description]

        Returns
        -------
        tuple of three tf.data.Dataset
            The sampled training, validation and test dataset.
        '''
        if undersample_ratio == oversample_ratio == 0:
            # No sampling. Just split the training dataset into training and validation dataset.
            training_dataset, validation_dataset = split_training_dataset(training_dataset, dataset_ratio=self.training_dataset_ratio)

        else:
            training_dataset_by_class, training_dataset_sample_counts, validation_dataset = \
                split_training_dataset_for_sampling(training_dataset, dataset_ratio=self.training_dataset_ratio)

            logging.info(f'Number of samples per class before sampling: {training_dataset_sample_counts}')
            if undersample_ratio > 0:
                training_dataset_by_class, training_dataset_sample_counts = undersample(training_dataset_by_class,
                                                                                        training_dataset_sample_counts,
                                                                                        undersample_ratio)
            if oversample_ratio > 0:
                training_dataset_by_class, training_dataset_sample_counts = oversample(training_dataset_by_class,
                                                                                       training_dataset_sample_counts,
                                                                                       oversample_ratio)

            training_dataset = join_dataset(training_dataset_by_class)

        logging.info(f"Train dataset size: {get_dataset_size(training_dataset)}")
        logging.info(f"Validation dataset size: {get_dataset_size(validation_dataset)}")
        logging.info(f"Test dataset size: {get_dataset_size(test_dataset)}")

        return training_dataset, validation_dataset


    def _get_dataset_handles(self) -> typing.Tuple[str]:
        return (content.name for content in self.tfrecords_directory.iterdir() if content.is_dir() )


    def load_dataset(self, dataset_handle=None) -> typing.Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict]:
        global PRINT_DATASET_DISTRIBUTIONS

        if self.dataset_name == 'idrid':
            logging.info(f"Loading 'IDRID' dataset '{dataset_handle}'...")

            training_dataset, validation_dataset, test_dataset = self._load_tfrecords(dataset_handle)

            if PRINT_DATASET_DISTRIBUTIONS:
                self._display_dataset_class_distributions(training_dataset, validation_dataset)
                exit()

            return self._prepare_idrid_dataset(training_dataset, validation_dataset, test_dataset)


        elif self.dataset_name == 'eyepacs':
            logging.info(f"Loading 'eyepacs' dataset '{dataset_handle}'...")

            (training_dataset, validation_dataset, test_dataset), dataset_info = tfds.load(
                'diabetic_retinopathy_detection/btgraham-300',
                split=['train', 'validation', 'test'],
                shuffle_files=True, with_info=True, data_dir=self.dataset_directory)

            def _preprocess(image_label_dict):
                return image_label_dict['image'], image_label_dict['label']

            training_dataset = training_dataset.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            validation_dataset = validation_dataset.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            test_dataset = test_dataset.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

            return self._prepare_tensorflow_datasets(training_dataset, validation_dataset, test_dataset, dataset_info)


        elif self.dataset_name == 'mnist':
            logging.info(f"Loading 'MNIST' dataset '{dataset_handle}'...")
            (training_dataset, validation_dataset, test_dataset), dataset_info = tfds.load(
                'mnist',
                split=['train[:90%]', 'train[90%:]', 'test'],
                shuffle_files=True,
                as_supervised=True,
                with_info=True,
                data_dir=self.dataset_directory)

            return self._prepare_tensorflow_datasets(training_dataset, validation_dataset, test_dataset, dataset_info)


    def _prepare_tensorflow_datasets(self, training_dataset, validation_dataset, test_dataset, dataset_info
                                    ) -> typing.Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict]:

        # Prepare training dataset
        training_dataset = training_dataset.map(preprocess_tensorflow_dataset,
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.caching:
            training_dataset = training_dataset.cache()
        training_dataset = training_dataset.map(augment_tensorflow_dataset,
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        training_dataset = training_dataset.shuffle(dataset_info.splits['train'].num_examples // 10)
        training_dataset = training_dataset.batch(self.batch_size)
        training_dataset = training_dataset.repeat(-1)
        training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # Prepare validation dataset
        validation_dataset = validation_dataset.map(preprocess_tensorflow_dataset,
                                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_dataset = validation_dataset.batch(self.batch_size)
        if self.caching:
            validation_dataset = validation_dataset.cache()
        validation_dataset = validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # Prepare test dataset
        test_dataset = test_dataset.map(preprocess_tensorflow_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.batch(self.batch_size)
        if self.caching:
            test_dataset = test_dataset.cache()
        test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return training_dataset, validation_dataset, test_dataset, dataset_info


    def _prepare_idrid_dataset(self, training_dataset, validation_dataset, test_dataset) -> typing.Tuple[
        tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict]:

        # Capture the counts so that it can be used later for weighted loss
        training_info, validation_info = self._get_dataset_class_distributions(training_dataset, validation_dataset)

        # Create custom dataset_info because the IDRID dataset doesn't have one.
        dataset_info_custom = dict()
        dataset_info_custom['image'] = {'shape': (256, 256, 3) }
        dataset_info_custom['label'] = {'num_classes': 5}
        dataset_info_custom['training_dataset'] = training_info
        dataset_info_custom['validation_dataset'] = validation_info

        # Prepare training dataset -------------------------
        if self.batch_size is not None:
            training_dataset = training_dataset.batch(self.batch_size)

        # Vectorized mapping - batch the dataset before map function
        training_dataset = training_dataset.map(preprocess_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # It is recommended to cache the dataset after the map transformation
        if self.caching:
            training_dataset = training_dataset.cache()
        training_dataset = training_dataset.shuffle(buffer_size=500, reshuffle_each_iteration=True)
        training_dataset = training_dataset.repeat(-1)
        training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # Prepare validation dataset -------------------------
        if self.batch_size is not None:
            validation_dataset = validation_dataset.batch(self.batch_size)
        if self.caching:
            validation_dataset = validation_dataset.cache()
        validation_dataset = validation_dataset.shuffle(buffer_size=500, reshuffle_each_iteration=True)
        validation_dataset = validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # Prepare test dataset -------------------------
        if self.batch_size is not None:
            test_dataset = test_dataset.batch(self.batch_size)
        if self.caching:
            test_dataset = test_dataset.cache()
        test_dataset = test_dataset.shuffle(buffer_size=500, reshuffle_each_iteration=True)
        test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)       

        return training_dataset, validation_dataset, test_dataset, dataset_info_custom


    def _get_dataset_class_distributions(self, train_dataset, validation_dataset) -> typing.Tuple[dict, dict]:
        '''Helper function for getting the target class distribution.

            Parameters
            ----------
            train_dataset : tf.data.Dataset
                A IDRID dataset with structure: image, image_name, retina_grade, risk_grade.

            validation_dataset : tf.data.Dataset
                A IDRID dataset with structure: image, image_name, retina_grade, risk_grade.

            Returns
            -------
            tuple of dictionaries for training and validation dataset counts
        '''

        training_labels = train_dataset.map(lambda image, image_name, retina_grade, risk_grade: retina_grade)
        training_label_types, training_label_count = np.unique(list(training_labels.as_numpy_iterator()),
                                                               return_counts=True)
        training_label_percent = training_label_count / np.sum(training_label_count)
        max_training_label_count = max(training_label_count)
        training_label_multipliers = [int(round(max_training_label_count/label_count)) for label_count in training_label_count]

        training_info = {
            'training_label_types': training_label_types,
            'training_label_count': training_label_count,
            'training_label_percent': training_label_percent,
            'training_label_multipliers': training_label_multipliers
        }


        validation_labels = validation_dataset.map(lambda image, image_name, retina_grade, risk_grade: retina_grade)
        validation_label_types, validation_label_counts = np.unique(list(validation_labels.as_numpy_iterator()),
                                                                    return_counts=True)
        validation_label_percent = validation_label_counts / np.sum(validation_label_counts)
                
        validation_info = {
            'validation_label_types': validation_label_types,
            'validation_label_counts': validation_label_counts,
            'validation_label_percent': validation_label_percent
        }

        return training_info, validation_info


    def _display_dataset_class_distributions(self, train_dataset, validation_dataset) -> None:
        '''Helper function for debuging.

        Parameters
        ----------
        train_dataset : tf.data.Dataset
            A IDRID dataset with structure: image, image_name, retina_grade, risk_grade.
        
        validation_dataset : tf.data.Dataset
            A IDRID dataset with structure: image, image_name, retina_grade, risk_grade.
        '''

        training_info, validation_info = self._get_dataset_class_distributions(train_dataset, validation_dataset)

        # Format lists of floats to precision two, for readability.
        formatted_training_label_percentages = [f'{percentage:.2f}' for percentage in training_info['training_label_percent'] ]
        formatted_training_label_percentages = '[' + ', '.join(formatted_training_label_percentages) + ']'

        formatted_validation_label_percentages = [f'{percentage:.2f}' for percentage in validation_info['validation_label_percent'] ]
        formatted_validation_label_percentages = '[' + ', '.join(formatted_validation_label_percentages) + ']'

        logging.info( ('Dataset distributions after creation:\n'
                       'Training dataset:\n'
                      f"\tClasses: {training_info['training_label_types'] } -> Number of samples: {training_info['training_label_count'] }\n"
                      f"\tPercentile distribution: {formatted_training_label_percentages}\n"
                      f"\tWeighted Multipliers: {training_info['training_label_multipliers'] }\n"
                       'Validation dataset:\n'
                      f"\tClasses: {validation_info['validation_label_types'] } -> Number of samples: {validation_info['validation_label_counts'] }\n"
                      f"\tPercentile distribution: {formatted_validation_label_percentages}") 
        )


    def _write_tfrecords(self, training_dataset, validation_dataset, test_dataset, dataset_handle) -> None:
        '''Write datasets to Tensorflow TFRecord files.'''

        tfrecords_directory = self.tfrecords_directory / dataset_handle
        if not tfrecords_directory.exists():
            tfrecords_directory.mkdir(parents=True, exist_ok=True)

        logging.info(f"Writing TFRecord files to '{tfrecords_directory}'...")

        if any(tfrecords_directory.iterdir()):
            logging.info(f"Directory '{tfrecords_directory}' is not empty, files will be overridden.")

        # The following functions can be used to convert a value to a type compatible with tf.train.Example.
        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _float_feature(value):
            """Returns a float_list from a float / double."""
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _build_image_feature_dict(data):
            """
            Map function to convert each image to tf.train.Example format

            Returns
            -------
            tf.train.Example
            """
            image, image_name, retina_grade, risk_grade = data
            image_shape = image.shape
            image_string = tf.io.encode_jpeg(image)

            feature = {
                'height': _int64_feature(image_shape[0]),
                'width': _int64_feature(image_shape[1]),
                'depth': _int64_feature(image_shape[2]),
                'retina_grade': _int64_feature(retina_grade),
                'risk_grade': _int64_feature(risk_grade),
                'image_name': _bytes_feature(image_name),
                'image_raw': _bytes_feature(image_string),
            }

            return tf.train.Example(features=tf.train.Features(feature=feature))

        train_image_feature_map = map(_build_image_feature_dict, list(training_dataset.as_numpy_iterator()))
        val_image_feature_map = map(_build_image_feature_dict, list(validation_dataset.as_numpy_iterator()))
        test_image_feature_map = map(_build_image_feature_dict, list(test_dataset.as_numpy_iterator()))

        train_tfrecord_full_path = tfrecords_directory / 'training.tfrecords'
        val_tfrecord_full_path = tfrecords_directory / 'validation.tfrecords'
        test_tfrecord_full_path = tfrecords_directory / 'test.tfrecords'

        # Delete existing TFRecord files
        for path in (train_tfrecord_full_path, val_tfrecord_full_path, test_tfrecord_full_path):
            if path.exists():
                os.remove(str(path))

        # Write the raw train image files
        def _write_tfrecords(file_path, data) -> None:
            with tf.io.TFRecordWriter(str(file_path)) as tfrecord_writer:
                for element in data:
                    tfrecord_writer.write(element.SerializeToString())

        _write_tfrecords(train_tfrecord_full_path, train_image_feature_map)
        _write_tfrecords(val_tfrecord_full_path, val_image_feature_map)
        _write_tfrecords(test_tfrecord_full_path, test_image_feature_map)

        logging.info(f"TFRecord files created.")


    def _load_tfrecords(self, dataset_handle: str) -> typing.Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        '''Reads tfrecords files and returns training, validatino and test datasets.

        Parameters
        ----------
        dataset_handle : str
            Name of the directory/name of the tfrecords of the dataset.
        
        Returns
        -------
        tuple of three tf.data.Dataset
            The parsed training, validation and test dataset.
        '''
        dataset_handles = self._get_dataset_handles()
        if dataset_handle not in dataset_handles:
            raise ValueError(
                f"Received dataset handle '{dataset_handle}' is unknown, known dataset handles: '{','.join(dataset_handles)}'")

        logging.info(f"Reading TFRecord files...")

        tfrecords_directory = self.tfrecords_directory / dataset_handle

        train_dataset = tf.data.TFRecordDataset(str(tfrecords_directory / 'training.tfrecords'))
        validation_dataset = tf.data.TFRecordDataset(str(tfrecords_directory / 'validation.tfrecords'))
        test_dataset = tf.data.TFRecordDataset(str(tfrecords_directory / 'test.tfrecords'))

        def _parse_image_function (image_proto):
            # Parse the input tf.train.Example proto using the dictionary above.
            # Create a dictionary describing the features.
            image_feature_description = {
                'height':       tf.io.FixedLenFeature([], tf.int64),
                'width':        tf.io.FixedLenFeature([], tf.int64),
                'depth':        tf.io.FixedLenFeature([], tf.int64),
                'retina_grade': tf.io.FixedLenFeature([], tf.int64),
                'risk_grade':   tf.io.FixedLenFeature([], tf.int64),
                'image_name':   tf.io.FixedLenFeature([], tf.string),
                'image_raw':    tf.io.FixedLenFeature([], tf.string),
            }

            # Load one example
            parsed_features = tf.io.parse_single_example(image_proto, image_feature_description)

            # Turn your saved image string into an array
            image = tf.image.decode_jpeg(parsed_features['image_raw'])

            image_name = parsed_features['image_name']
            retina_grade = parsed_features['retina_grade']
            risk_grade = parsed_features['risk_grade']

            return image, image_name, retina_grade, risk_grade

        parsed_training_dataset = train_dataset.map(_parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        parsed_validation_dataset = validation_dataset.map(_parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        parsed_test_dataset = test_dataset.map(_parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return parsed_training_dataset, parsed_validation_dataset, parsed_test_dataset
