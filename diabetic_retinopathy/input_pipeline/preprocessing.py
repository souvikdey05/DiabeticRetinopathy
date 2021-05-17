import gin
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import logging
import typing


@gin.configurable
def preprocess_tensorflow_dataset (image, label,
                                   img_height, img_width): # <- configs
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize image: `uint8` -> `float32`.
    tf.cast(image, tf.float32) / 255.

    # Resize image
    image = tf.image.resize(image, size=(img_height, img_width))

    return image, label


@gin.configurable
def preprocess_resize(image,
                      img_height, img_width, preserve_aspect_ratio):  # <- configs
    """Dataset preprocessing: resizing"""
    # Resize image
    image = tf.image.resize(image, size=(img_height, img_width),
                            preserve_aspect_ratio=preserve_aspect_ratio,
                            antialias=True)

    return image


def normalise_image_data (image, image_name, retina_grade, risk_grade):
    '''Convert the image data to type float and normalise the values in the range 0 to 1.'''

    image = tf.image.convert_image_dtype(image, tf.float32)

    return image, image_name, retina_grade, risk_grade


@gin.configurable
def preprocess_resample(dataset,
                        target_dist): # <- configs
    initial_state = {'class_0': 0, 'class_1': 0, 'class_2': 0, 'class_3': 0, 'class_4': 0}

    def _count_classes(counts, data):
        _, _, retina_grade, _ = data

        class_0 = retina_grade == 0
        if class_0.dtype != tf.int32:
            class_0 = tf.cast(class_0, tf.int32)

        class_1 = retina_grade == 1
        if class_1.dtype != tf.int32:
            class_1 = tf.cast(class_1, tf.int32)

        class_2 = retina_grade == 2
        if class_2.dtype != tf.int32:
            class_2 = tf.cast(class_2, tf.int32)

        class_3 = retina_grade == 3
        if class_3.dtype != tf.int32:
            class_3 = tf.cast(class_3, tf.int32)

        class_4 = retina_grade == 4
        if class_4.dtype != tf.int32:
            class_4 = tf.cast(class_4, tf.int32)

        counts['class_0'] += tf.reduce_sum(class_0)
        counts['class_1'] += tf.reduce_sum(class_1)
        counts['class_2'] += tf.reduce_sum(class_2)
        counts['class_3'] += tf.reduce_sum(class_3)
        counts['class_4'] += tf.reduce_sum(class_4)

        return counts

    counts = dataset.reduce(initial_state=initial_state,
                            reduce_func=_count_classes)

    counts = np.array([counts['class_0'].numpy(),
                       counts['class_1'].numpy(),
                       counts['class_2'].numpy(),
                       counts['class_3'].numpy(),
                       counts['class_4'].numpy()]).astype(np.float32)

    logging.info(f"Initial target class counts {counts}...")
    initial_dist_fractions = counts / counts.sum()
    logging.info(f"Initial target class distribution {initial_dist_fractions}...")

    resampler = tf.data.experimental.rejection_resample(
        class_func=(lambda image, image_name, retina_grade, risk_grade: tf.cast(retina_grade, dtype=tf.int32)),
        target_dist=target_dist,
        initial_dist=initial_dist_fractions)
    dataset = dataset.apply(resampler)

    # dropping of extra label created by resampler
    dataset = dataset.map(lambda extra_label, features_and_label: features_and_label)

    sampled_counts = dataset.reduce(initial_state=initial_state,
                                    reduce_func=_count_classes)

    sampled_counts = np.array([sampled_counts['class_0'].numpy(),
                               sampled_counts['class_1'].numpy(),
                               sampled_counts['class_2'].numpy(),
                               sampled_counts['class_3'].numpy(),
                               sampled_counts['class_4'].numpy()]).astype(np.float32)

    logging.info(f"After Sampling target class counts {sampled_counts}...")
    sample_dist_fractions = sampled_counts / sampled_counts.sum()
    logging.info(f"After Sampling target class distribution {sample_dist_fractions}...")

    # # get the count before re-sampling
    # for classes in unique_classes:
    #     dataset_filtered = dataset.filter(
    #         lambda image, image_name, retina_grade, risk_grade: retina_grade == classes)
    #     classes_count[classes] = len(list(dataset_filtered.as_numpy_iterator()))
    #     dataset_separated_on_classes[classes] = dataset_filtered

    return dataset


def augment_tensorflow_dataset (image, label):
    """Data augmentation"""

    return image, label


def preprocess_augment(image, image_name, retina_grade, risk_grade):
    """ Image augmentation"""

    # rotate counter-clockwise before flip
    k = np.random.uniform(low=1.0, high=20.0)
    image = tfa.image.rotate(image, tf.constant(np.pi / k))

    # random flip horizontally
    image = tf.image.random_flip_left_right(image)

    # rotate after flip
    k = np.random.uniform(low=1.0, high=20.0)
    image = tfa.image.rotate(image, tf.constant(np.pi / k))

    # transpose the vector with certain probability
    prob = tf.random.uniform([])
    threshold = tf.constant(0.5)
    if prob > threshold:
        image = tf.image.transpose(image)

    # RGB to YIQ with certain probability
    prob = tf.random.uniform([])
    threshold = tf.constant(0.7)
    if prob > threshold:
        delta = 0.5
        lower_saturation = 0.1
        upper_saturation = 0.9
        lower_value = 0.2
        upper_value = 0.8
        image = tfa.image.random_hsv_in_yiq(image, delta, lower_saturation, upper_saturation, lower_value,
                                            upper_value)

    image = tf.image.random_brightness(image, 0.5)
    image = tf.image.random_contrast(image, 0.2, 0.5)

    return image, image_name, retina_grade, risk_grade


def get_dataset_size (dataset) -> int:
    return len(list(dataset.as_numpy_iterator() ) )


def oversample (dataset_by_class:tf.data.Dataset, sample_counts_by_class:typing.List[int], oversample_ratio:float
                ) -> typing.Tuple[typing.List[tf.data.Dataset], typing.List[int] ]:
    
    logging.info(f'Oversample the dataset with an oversample ratio of: {oversample_ratio}')
    
    # Repeat class samples until close to target ratio
    target_sample_count = int(oversample_ratio * max(sample_counts_by_class) )
    logging.info(f'Oversample target sample count: {target_sample_count}')
    
    numbers_of_full_set_augmentations = [ (target_sample_count // sample_count) - 1 if (sample_count < target_sample_count) else 0
                    for sample_count in sample_counts_by_class]

    unaugmented_dataset_by_class = list(dataset_by_class) # Hard-copy the dataset.

    # Apply image augmentation to a whole dataset multiple times.
    dataset_by_class = [dataset.concatenate(unaugmented_dataset.repeat(number_of_full_set_augmentations).map(preprocess_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE) )
        if number_of_full_set_augmentations > 0 else dataset
        for dataset, unaugmented_dataset, number_of_full_set_augmentations in zip(dataset_by_class, unaugmented_dataset_by_class, numbers_of_full_set_augmentations) ]

    # Calculate the sample counts instead of expensive recounting.
    sample_counts_by_class = [sample_count * (number_of_full_set_augmentations + 1) if (number_of_full_set_augmentations > 0) else sample_count
                                for sample_count, number_of_full_set_augmentations in zip(sample_counts_by_class, numbers_of_full_set_augmentations) ]
    
    # Add missing number of samples to reach target ratio
    numbers_of_missing_samples = [target_sample_count - sample_count if (sample_count < target_sample_count) else 0
                                    for sample_count in sample_counts_by_class]
    dataset_by_class = [dataset.concatenate(unaugmented_dataset.take(missing_samples_count).map(preprocess_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE) )
                            for dataset, unaugmented_dataset, missing_samples_count in zip(dataset_by_class, unaugmented_dataset_by_class, numbers_of_missing_samples) ]
    
    # All classes have either target number of samples or more.
    sample_counts_by_class = [max(sample_count, target_sample_count) for sample_count in sample_counts_by_class]
    logging.info(f'Oversampling finished. Sample counts by class: {sample_counts_by_class}')

    return dataset_by_class, sample_counts_by_class


def undersample (dataset_by_class, sample_counts_by_class, undersample_ratio
                ) -> typing.Tuple[typing.List[tf.data.Dataset], typing.List[int] ]:
    
    logging.info(f'Undersample the dataset with an undersample ratio of: {undersample_ratio}')
    
    # The target value is n times the smallest class sample count.
    target_sample_count = int(min(sample_counts_by_class) * undersample_ratio)
    logging.info(f'Undersample target sample count: {target_sample_count}')

    # Classes with more samples than the target value are reduced to the target value.
    dataset_by_class = [class_dataset.take(target_sample_count) if (sample_count > target_sample_count) else class_dataset
                            for class_dataset, sample_count in zip(dataset_by_class, sample_counts_by_class) ]

    # Avoid to recount the classes, use target count for classes with more samples than target value.
    sample_counts_by_class = [min(sample_count, target_sample_count) for sample_count in sample_counts_by_class]

    logging.info(f'Undersampling finished. Sample counts by class: {sample_counts_by_class}')
    return dataset_by_class, sample_counts_by_class


def split_training_dataset (training_dataset, dataset_ratio) -> typing.Tuple[tf.data.Dataset, tf.data.Dataset]:
    '''Split the training dataset into training and validation dataset.

    Returns
    -------
    tuple of tf.data.Dataset, tf.data.Dataset
        The training and the validation dataset.
    '''
    logging.info(f'Splitting the training dataset with a ratio of: {dataset_ratio}')
    
    dataset_by_class, class_sample_counts = split_dataset_by_class(training_dataset)
    logging.info(f'Sample counts for train dataset before splitting: {class_sample_counts}')
    dataset_by_class = [dataset.shuffle(buffer_size=dataset_size, reshuffle_each_iteration=True)
                            for dataset, dataset_size in zip(dataset_by_class, class_sample_counts) ]

    train_dataset_sample_counts = [int(dataset_ratio * class_sample_count) for class_sample_count in class_sample_counts]

    training_dataset =   dataset_by_class[0].take(train_dataset_sample_counts[0] )
    validation_dataset = dataset_by_class[0].skip(train_dataset_sample_counts[0] )
    for class_dataset, sample_count in zip(dataset_by_class[1:], train_dataset_sample_counts[1:] ):
        training_dataset =   training_dataset.concatenate(class_dataset.take(sample_count) )
        validation_dataset = validation_dataset.concatenate(class_dataset.skip(sample_count) )
    
    return training_dataset, validation_dataset


def split_training_dataset_for_sampling (training_dataset, dataset_ratio) -> typing.Tuple[tf.data.Dataset, tf.data.Dataset]:
    '''Split the training dataset into training and validation dataset.
    '''
    logging.info(f'Splitting the training dataset with a ratio of: {dataset_ratio}')
    
    dataset_by_class, class_sample_counts = split_dataset_by_class(training_dataset)
    logging.info(f'Sample counts for train dataset before splitting: {class_sample_counts}')
    dataset_by_class = [dataset.shuffle(buffer_size=dataset_size, reshuffle_each_iteration=True)
                            for dataset, dataset_size in zip(dataset_by_class, class_sample_counts) ]

    train_dataset_sample_counts = [int(dataset_ratio * class_sample_count) for class_sample_count in class_sample_counts]

    training_dataset_by_class = [class_dataset.take(sample_count) 
                                    for class_dataset, sample_count in zip(dataset_by_class, train_dataset_sample_counts) ]

    validation_dataset = dataset_by_class[0].skip(train_dataset_sample_counts[0] )
    for class_dataset, sample_count in zip(dataset_by_class[1:], train_dataset_sample_counts[1:] ):
        validation_dataset = validation_dataset.concatenate(class_dataset.skip(sample_count) )
        
    return training_dataset_by_class, train_dataset_sample_counts, validation_dataset


def split_dataset_by_class (dataset) -> typing.Tuple[typing.List[tf.data.Dataset], typing.List[int] ]:
    number_of_classes = 5
    dataset_by_class = [dataset.filter(lambda image, image_name, retina_grade, risk_grade: retina_grade == label)
                            for label in range(0, number_of_classes) ]
    sample_counts_by_class = [get_dataset_size(dataset) for dataset in dataset_by_class]

    return dataset_by_class, sample_counts_by_class


def join_dataset (dataset_by_class) -> tf.data.Dataset:
    
    dataset = dataset_by_class[0]
    for class_dataset in dataset_by_class[1:]:
        dataset = dataset.concatenate(class_dataset)

    return dataset
