import logging
import tensorflow as tf


def set_loggers(paths, logging_level=0, b_stream=False, b_debug=False):
    # Base logger
    base_logger = logging.getLogger()
    base_logger.setLevel(logging_level)

    # tf logger
    logger_tf = tf.get_logger()
    logger_tf.setLevel(logging_level)

    file_handler = logging.FileHandler(paths['base_log'])
    base_logger.addHandler(hdlr=file_handler)
    logger_tf.addHandler(file_handler)

    # Training logger
    training_logger = logging.getLogger('training')
    training_logger.setLevel(logging_level)
    training_logger.addHandler(hdlr=logging.FileHandler(paths['training_log']))

    # Evaluation logger
    evaluation_logger = logging.getLogger('evaluation')
    evaluation_logger.setLevel(logging_level)
    evaluation_logger.addHandler(hdlr=logging.FileHandler(paths['evaluation_log']))

    # Ensemble logger
    evaluation_logger = logging.getLogger('ensemble')
    evaluation_logger.setLevel(logging_level)
    evaluation_logger.addHandler(hdlr=logging.FileHandler(paths['training_log']))

    # Plot to console
    if b_stream:
        stream_handler = logging.StreamHandler()
        base_logger.addHandler(stream_handler)

    if b_debug:
        tf.debugging.set_log_device_placement(False)
