import tensorflow as tf
import seaborn as sb
import matplotlib.pyplot as plt
import enum


class Metric(enum.Enum):
    Accuracy =    enum.auto(),
    Precision =   enum.auto(),
    Recall =      enum.auto(),
    F1 =          enum.auto(),
    Specificity = enum.auto(),
    MCC =         enum.auto()

    @classmethod
    def from_string (cls, text):        
        lowered_text = text.lower()

        for name, member in cls.__members__.items():
            if lowered_text == name.lower():
                return member
        else:
            names = [name for name, member in cls.__members__.items() ]
            raise ValueError(f"Invalid metric name: '{text}', valid names: {names}.")

    @classmethod
    def has_member (cls, name):
        return name in cls.__members__

    @classmethod
    def has_member_case_insensitive (cls, name):
        return name.lower() in map(str.lower, cls.__members__.keys() )


class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, num_classes, name="confusion_matrix_metric", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight("confusion_matrix",
                                                shape=(num_classes, num_classes),
                                                initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, 1)

        # The matrix columns represent the prediction labels and the rows represent the real labels
        confusion_matrix_temp = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32,
                                                         num_classes=self.num_classes,
                                                         weights=sample_weight)
        self.confusion_matrix.assign_add(confusion_matrix_temp)

        return self.confusion_matrix

    def result(self):
        """
            returns confusion matrix -
            The matrix rows represent the prediction labels and the columns represent the real labels
        """
        transposed_confusion_matrix = tf.transpose(self.confusion_matrix)

        return transposed_confusion_matrix

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.confusion_matrix.assign(tf.zeros(shape=self.confusion_matrix.shape))


class ClassWiseMetric(tf.keras.metrics.Metric):

    def __init__(self, num_classes, name="class_wise_metric", **kwargs):
        super(ClassWiseMetric, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight("confusion_matrix",
                                                shape=(num_classes, num_classes),
                                                initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, 1)

        # The matrix columns represent the prediction labels and the rows represent the real labels
        confusion_matrix_temp = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32,
                                                         num_classes=self.num_classes,
                                                         weights=sample_weight)
        self.confusion_matrix.assign_add(confusion_matrix_temp)
        return self.confusion_matrix

    def result(self):
        """
            returns class wise accuracy, precision, recall, f1, specificity, mcc
        """
        # The matrix rows represent the prediction labels and the columns represent the real labels
        transposed_confusion_matrix = tf.transpose(self.confusion_matrix)
        # tf.print(f"transposed_confusion_matrix = {transposed_confusion_matrix}")

        diag_part = tf.linalg.diag_part(transposed_confusion_matrix)

        true_positives = tf.reshape(diag_part, [-1, 1])  # (num_classes, 1) column matrix - one row for each class
        row_sums = tf.reduce_sum(transposed_confusion_matrix, 1, keepdims=True)  # (num_classes, 1) column matrix

        false_positives = row_sums - true_positives  # (num_classes, 1) column matrix - one row for each class

        col_sums = tf.reduce_sum(transposed_confusion_matrix, 0, keepdims=True)
        col_sums = tf.reshape(col_sums, [-1, 1])  # (num_classes, 1) column matrix

        false_negatives = col_sums - true_positives  # (num_classes, 1) column matrix - one row for each class

        total = tf.reduce_sum(transposed_confusion_matrix)
        total = tf.cast(total, tf.float32)

        true_negatives = total - true_positives - false_positives - false_negatives  # (num_classes, 1) column matrix - one row for each class

        # cast them to float
        true_positives = tf.dtypes.cast(true_positives, tf.float32)
        false_positives = tf.dtypes.cast(false_positives, tf.float32)
        true_negatives = tf.dtypes.cast(true_negatives, tf.float32)
        false_negatives = tf.dtypes.cast(false_negatives, tf.float32)

        # metric each class
        accuracy_i = (true_positives + true_negatives) / total
        precision_i = true_positives / (true_positives + false_positives + tf.constant(1e-15))
        recall_i = true_positives / (true_positives + false_negatives + tf.constant(1e-15))
        f1_i = 2 * precision_i * recall_i / (precision_i + recall_i + tf.constant(1e-15))
        specificity_i = true_negatives / (true_negatives + false_positives + tf.constant(1e-15))
        mcc_i = (true_positives * true_negatives - false_positives * false_negatives) / (tf.sqrt(
            (true_positives + false_positives) * (true_positives + false_negatives) * (
                    true_negatives + false_positives) * (true_negatives + false_negatives)) + tf.constant(1e-15))

        # reshape precision_i, recall_i, f1_i metrics to (1, num_classes) flatten vector
        accuracy_i = tf.reshape(accuracy_i, [-1])
        precision_i = tf.reshape(precision_i, [-1])
        recall_i = tf.reshape(recall_i, [-1])
        f1_i = tf.reshape(f1_i, [-1])
        specificity_i = tf.reshape(specificity_i, [-1])
        mcc_i = tf.reshape(mcc_i, [-1])

        return accuracy_i, precision_i, recall_i, f1_i, specificity_i, mcc_i

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.confusion_matrix.assign(tf.zeros(shape=self.confusion_matrix.shape))


class AverageMetric(tf.keras.metrics.Metric):

    def __init__(self, num_classes, name="average_metric", **kwargs):
        super(AverageMetric, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight("confusion_matrix",
                                                shape=(num_classes, num_classes),
                                                initializer="zeros")


    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, 1)
        confusion_matrix_temp = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32,
                                                         num_classes=self.num_classes,
                                                         weights=sample_weight)
        self.confusion_matrix.assign_add(confusion_matrix_temp)
        return self.confusion_matrix


    def result(self):
        """
            returns class wise accuracy, precision, recall, f1, specificity, mcc
        """
        # The matrix rows represent the prediction labels and the columns represent the real labels
        transposed_confusion_matrix = tf.transpose(self.confusion_matrix)

        diag_part = tf.linalg.diag_part(transposed_confusion_matrix)

        true_positives = tf.reshape(diag_part, [-1, 1])  # (num_classes, 1) column matrix - one row for each class

        row_sums = tf.reduce_sum(transposed_confusion_matrix, 1, keepdims=True)  # (num_classes, 1) column matrix

        false_positives = row_sums - true_positives  # (num_classes, 1) column matrix - one row for each class
        col_sums = tf.reduce_sum(transposed_confusion_matrix, 0, keepdims=True)
        col_sums = tf.reshape(col_sums, [-1, 1])  # (num_classes, 1) column matrix

        false_negatives = col_sums - true_positives  # (num_classes, 1) column matrix - one row for each class

        total = tf.reduce_sum(transposed_confusion_matrix)
        total = tf.cast(total, tf.float32)

        true_negatives = total - true_positives - false_positives - false_negatives  # (num_classes, 1) column matrix - one row for each class

        # cast them to float
        true_positives = tf.dtypes.cast(true_positives, tf.float32)
        false_positives = tf.dtypes.cast(false_positives, tf.float32)
        true_negatives = tf.dtypes.cast(true_negatives, tf.float32)
        false_negatives = tf.dtypes.cast(false_negatives, tf.float32)

        # metric each class
        accuracy_i = (true_positives + true_negatives) / total
        precision_i = true_positives / (true_positives + false_positives + tf.constant(1e-15))
        recall_i = true_positives / (true_positives + false_negatives + tf.constant(1e-15))
        f1_i = 2 * precision_i * recall_i / (precision_i + recall_i + tf.constant(1e-15))
        specificity_i = true_negatives / (true_negatives + false_positives + tf.constant(1e-15))
        mcc_i = (true_positives * true_negatives - false_positives * false_negatives) / (tf.sqrt(
            (true_positives + false_positives) * (true_positives + false_negatives) * (
                    true_negatives + false_positives) * (true_negatives + false_negatives)) + tf.constant(1e-15))

        # reshape precision_i, recall_i, f1_i metrics to (1, num_classes) flatten vector
        accuracy_i = tf.reshape(accuracy_i, [-1])
        precision_i = tf.reshape(precision_i, [-1])
        recall_i = tf.reshape(recall_i, [-1])
        f1_i = tf.reshape(f1_i, [-1])
        specificity_i = tf.reshape(specificity_i, [-1])
        mcc_i = tf.reshape(mcc_i, [-1])

        # macro averaging
        accuracy_macro = tf.reduce_sum(accuracy_i) / self.num_classes
        precision_macro = tf.reduce_sum(precision_i) / self.num_classes
        recall_macro = tf.reduce_sum(recall_i) / self.num_classes
        f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro + tf.constant(1e-15))
        specificity_macro = tf.reduce_sum(specificity_i) / self.num_classes

        total_true_positive = tf.reduce_sum(true_positives)
        total_true_negative = tf.reduce_sum(true_negatives)
        total_false_positive = tf.reduce_sum(false_positives)
        total_false_negative = tf.reduce_sum(false_negatives)

        true_positives_macro = total_true_positive / self.num_classes
        true_negative_macro = total_true_negative / self.num_classes
        false_positives_macro = total_false_positive / self.num_classes
        false_negative_macro = total_false_negative / self.num_classes

        mcc_macro = (true_positives_macro * true_negative_macro - false_positives_macro * false_negative_macro) / (
                tf.sqrt(
                    (true_positives_macro + false_positives_macro) * (
                            true_positives_macro + false_negative_macro) * (
                            true_negative_macro + false_positives_macro) * (
                            true_negative_macro + false_negative_macro)) + tf.constant(1e-15))

        # micro averaging
        accuracy_micro = (total_true_positive + total_true_negative) / (self.num_classes * total + tf.constant(1e-15))
        precision_micro = total_true_positive / (total_true_positive + total_false_positive + tf.constant(1e-15))
        recall_micro = total_true_positive / (total_true_positive + total_false_negative + tf.constant(1e-15))
        f1_micro = 2 * precision_micro * recall_macro / (precision_micro + recall_micro + tf.constant(1e-15))
        specificity_micro = total_true_negative / (total_true_negative + total_false_positive + tf.constant(1e-15))
        mcc_micro = (total_true_positive * total_true_negative - total_false_positive * total_false_negative) / (
                tf.sqrt(
                    (total_true_positive + total_false_positive) * (total_true_positive + total_false_negative) * (
                            total_true_negative + total_false_positive) * (
                            total_true_negative + total_false_negative)) + tf.constant(1e-15))

        return accuracy_macro, precision_macro, recall_macro, f1_macro, specificity_macro, mcc_macro, \
               accuracy_micro, precision_micro, recall_micro, f1_micro, specificity_micro, mcc_micro


    def calculate_macro_metric (self, metric: Metric) -> tf.Tensor:
        '''Calculates the macro values for one of the metrics:
        Accuracy, Precision, Recall, F1, Specificity, Matthews-Correlation-Coefficient (MCC).

        Returns
        -------
        tf.Tensor
            The value of the selected as a scalar tf.Tensor.
        '''        
        transposed_confusion_matrix = tf.transpose(self.confusion_matrix)
        diag_part = tf.linalg.diag_part(transposed_confusion_matrix)
        true_positives = tf.reshape(diag_part, [-1, 1] )

        row_sums = tf.reduce_sum(transposed_confusion_matrix, 1, keepdims=True)  # (num_classes, 1) column matrix
        false_positives = row_sums - true_positives
        
        col_sums = tf.reduce_sum(transposed_confusion_matrix, 0, keepdims=True)
        col_sums = tf.reshape(col_sums, [-1, 1] )
        false_negatives = col_sums - true_positives

        total = tf.reduce_sum(transposed_confusion_matrix)
        total = tf.cast(total, tf.float32)
        true_negatives = total - true_positives - false_positives - false_negatives

        true_positives =  tf.dtypes.cast(true_positives,  tf.float32)
        false_positives = tf.dtypes.cast(false_positives, tf.float32)
        true_negatives =  tf.dtypes.cast(true_negatives,  tf.float32)
        false_negatives = tf.dtypes.cast(false_negatives, tf.float32)

        def _calculate_precision ():
            precision_i = true_positives / (true_positives + false_positives + tf.constant(1e-15) )
            precision_i = tf.reshape(precision_i, [-1] )
            precision_macro = tf.reduce_sum(precision_i) / self.num_classes
            return precision_i, precision_macro
        
        def _calculate_precision ():
            recall_i = true_positives / (true_positives + false_negatives + tf.constant(1e-15))
            recall_i = tf.reshape(recall_i, [-1] )
            recall_macro = tf.reduce_sum(recall_i) / self.num_classes
            return recall_i, recall_macro


        if metric == Metric.Accuracy:
            accuracy_i = (true_positives + true_negatives) / total
            accuracy_i = tf.reshape(accuracy_i, [-1] )
            accuracy_macro = tf.reduce_sum(accuracy_i) / self.num_classes
            return accuracy_macro

        elif metric == Metric.Specificity:
            specificity_i = true_negatives / (true_negatives + false_positives + tf.constant(1e-15))
            specificity_i = tf.reshape(specificity_i, [-1] )
            specificity_macro = tf.reduce_sum(specificity_i) / self.num_classes
            return specificity_macro

        elif metric == Metric.Precision:
            precision_i, precision_macro = _calculate_precision()
            return precision_macro

        elif metric == Metric.Recall:
            recall_i, recall_macro = _calculate_precision()
            return recall_macro
        
        elif metric == Metric.F1:
            precision_i, precision_macro = _calculate_precision()
            recall_i, recall_macro = _calculate_precision()

            f1_i = 2 * precision_i * recall_i / (precision_i + recall_i + tf.constant(1e-15) )
            f1_i = tf.reshape(f1_i, [-1] )
            f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro + tf.constant(1e-15) )
            return f1_macro

        elif metric == Metric.MCC:
            total_true_positive =  tf.reduce_sum(true_positives)
            total_true_negative =  tf.reduce_sum(true_negatives)
            total_false_positive = tf.reduce_sum(false_positives)
            total_false_negative = tf.reduce_sum(false_negatives)

            true_positives_macro =  total_true_positive / self.num_classes
            true_negative_macro =   total_true_negative / self.num_classes
            false_positives_macro = total_false_positive / self.num_classes
            false_negative_macro =  total_false_negative / self.num_classes

            mcc_macro = (true_positives_macro * true_negative_macro - false_positives_macro * false_negative_macro) / (
                        tf.sqrt(
                            (true_positives_macro + false_positives_macro) * (true_positives_macro + false_negative_macro) * 
                            (true_negative_macro + false_positives_macro) * (true_negative_macro + false_negative_macro)
                        ) + tf.constant(1e-15) )
            return mcc_macro
        
        else:
            raise ValueError(f"Invalid metric name: '{metric}', valid names: {Metric.__members__.keys() }.")


    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.confusion_matrix.assign(tf.zeros(shape=self.confusion_matrix.shape))


def convert_AverageMetric_result(result):
    names = ['accuracy_macro', 'precision_macro', 'recall_macro', 'f1_macro', 'specificity_macro', 'mcc_macro',
             'accuracy_micro', 'precision_micro', 'recall_micro', 'f1_micro', 'specificity_micro', 'mcc_micro']

    converted_result = dict()
    for position, name in enumerate(names):
        converted_result[name] = result[position]

    return converted_result


def convert_ClassWiseMetric_result(result, num_classes):
    names = ['accuracy_{0}', 'precision_{0}', 'recall_{0}', 'f1_{0}', 'specificity_{0}', 'mcc_{0}']
    converted_result = dict()

    for position, name in enumerate(names):
        for classIdx in range(num_classes):
            converted_result[name.format(classIdx)] = result[position][classIdx]

    return converted_result


def format_metric_output(results):
    """Formats the metric dictionary to appropriate output string
        Parameters:
            `results` (dict): Metric dictionary

        Returns:
            (str): formatted metric output string
    """
    metric_string = ""
    for name, value in results.items():
        metric_string = "{0} {1}: {2:.3f} ".format(metric_string, name, value)

    return metric_string


def format_confusion_matrix_metric_output(num_classes, conf_matrix):
    """Creates the heat map for confusion matrix
        Parameters:
            `num_classes` (int): Number of target classes
            `conf_matrix` (ndarray): confusion matrix

        Returns:
            (plt.figure): heat map of the confusion matrix
    """
    axis_labels = [str(i) for i in range(num_classes)]
    figure = plt.figure(figsize=(num_classes + 2, num_classes))
    sb.heatmap(conf_matrix, xticklabels=axis_labels, yticklabels=axis_labels,
               annot=True, fmt='g')
    plt.xlabel('True')
    plt.ylabel('Prediction')
    return figure
