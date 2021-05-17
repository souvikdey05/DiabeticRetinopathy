import gin
import tensorflow as tf

from models.layers import vgg_block, inception_module


def load_model(model_config, dataset_info, run_paths) -> tf.keras.Model:
    '''Create a keras model from a dict of model configuration and dataset information.
    '''
    model_types = [*TransferLearningModel.model_types, 'vgg_like', 'inception_like']

    # Checking for valid models
    if isinstance(model_config, dict):
        model_name = model_config['model']
        if model_name not in model_types:
            raise ValueError(f"Invalid model name: '{model_name}'.")
    else:
        raise ValueError('Invalid model configuration.')


    model_name = model_config['model']
    
    if model_name in TransferLearningModel.model_types:
        return TransferLearningModel(model_type=model_name,
                                     input_shape=dataset_info['image']['shape'],
                                     number_of_classes=dataset_info['label']['num_classes'],
                                     duplication_test=run_paths['model_directories'].keys(),
                                     **model_config)

    elif model_name == 'vgg_like':
        return vgg_like(input_shape=dataset_info['image']['shape'],
                         number_of_classes=dataset_info['label']['num_classes'],                         
                         duplication_test=run_paths['model_directories'].keys(),
                         **model_config)

    elif model_name == 'inception_like':                
        return inception_like(input_shape=dataset_info['image']['shape'],
                               number_of_classes=dataset_info['label']['num_classes'],
                               duplication_test=run_paths['model_directories'].keys(),
                               **model_config)


def _duplication_test (new_model_name, existing_models):
    '''Test to see if a model of the same name already exists in this run.
    If there already exists a model then an indicator is added to name of the current model, like '--1'.

    Parameters
    ----------
    new_model_name : str
        Name to compare against existing ones.
    existing_models : list of str or dict
        Either a list of the existing model names or a dict where the keys are the models names.

    Returns
    -------
    str
        Either the given model name or an edited version if a model of the same name already exists.
    '''

    if new_model_name in existing_models: # Duplicate model exists
        duplicates = [model_name for model_name in existing_models if model_name.startswith(new_model_name) ]
        return new_model_name + f'--{len(duplicates)}' # The standard ..(1) is not possible because of invalid tf scope names.
    
    return new_model_name


@gin.configurable
def vgg_like(input_shape, number_of_classes, duplication_test=None, **kwargs) -> tf.keras.Model:
    '''Defines a VGG-like architecture.

    Parameters:
        `input_shape` (tuple: 3): input shape of the neural network
        `n_classes` (int): number of classes, corresponding to the number of output neurons

        Additional args -
        `base_filters` (int): number of base filters, which are doubled for every VGG block
        `n_blocks` (int): number of VGG blocks
        `dense_units` (int): number of dense units
        `dropout_rate` (float): dropout rate
    Returns:
        (keras.Model): keras model object
    '''

    assert number_of_classes > 0

    n_blocks = kwargs.get('n_blocks', 1)
    assert n_blocks > 0, 'Number of blocks has to be at least 1.'

    base_filters =          kwargs.get('base_filters', 8)
    dropout_rate =          kwargs.get('dropout_rate', 0.2)
    number_of_dense_units = kwargs.get('number_of_dense_units', 32)
    dataset_handle =        kwargs.get('dataset_handle', '')
    class_weights_scale =   kwargs.get('class_weights_scale', 0.0)


    input = tf.keras.Input(input_shape)

    output = vgg_block(input, base_filters)
    for i in range(2, n_blocks):
        output = vgg_block(output, base_filters * 2 ** i)
    output = tf.keras.layers.GlobalAveragePooling2D()(output)
    output = tf.keras.layers.Dense(number_of_dense_units, activation=tf.nn.relu)(output)
    output = tf.keras.layers.Dropout(dropout_rate)(output)
    output = tf.keras.layers.Dense(number_of_classes)(output)

    strings = [str(x) for x in (base_filters, n_blocks, number_of_dense_units) ]
    dropout_rate_string = str(dropout_rate).replace('.', '_')
    class_weights_scale_string = str(class_weights_scale).replace('.', '_')    
    name = '-'.join( ['vgg_like', *strings, dataset_handle, dropout_rate_string, class_weights_scale_string] )

    if duplication_test:
        name = _duplication_test(name, duplication_test)

    return tf.keras.Model(inputs=input, outputs=output, name=name)


@gin.configurable
def inception_like(input_shape, number_of_classes, duplication_test=None, **kwargs) -> tf.keras.Model:
    '''Creates a model with inception-like architecture.

    Parameters:
        `input_shape` (tuple: 3): Input shape of the model
        `number_of_classes` (int): Number of classes, corresponding to the number of output neurons

        Additional args -
        `module_1_filters` (tuple 6): Dimensions of the filters of the convolution layers of the first inception module
        `module_2_filters` (tuple 6): Dimensions of the filters of the convolution layers of the second inception module
        `kernel_initializer` (str): Kernel weight initializer for Conv and Dense layers
        `dropout_rate` (float): Dropout rate
        `number_of_dense_units` (int): Number of dense units in the last layer before the classification layer
    Returns:
        (keras.Model): Keras model object
    '''

    assert number_of_classes > 0

    n_blocks = kwargs.get('n_blocks', 1)
    assert n_blocks > 0, 'Number of blocks has to be at least 1.'

    default_filters = [25, 25, 25, 25, 25, 25]
    module_1_filters =      kwargs.get('module_1_filters', default_filters)
    module_2_filters =      kwargs.get('module_2_filters', default_filters)
    dropout_rate =          kwargs.get('dropout_rate', 0.2)
    number_of_dense_units = kwargs.get('number_of_dense_units', 250)
    kernel_initializer =    kwargs.get('kernel_initializer', 'glorot_uniform')
    dataset_handle =        kwargs.get('dataset_handle', '')
    class_weights_scale =   kwargs.get('class_weights_scale', 0.0)


    input = tf.keras.Input(shape=input_shape)

    output = inception_module(input, *module_1_filters, kernel_initializer)
    output = inception_module(output, *module_2_filters, kernel_initializer)

    output = tf.keras.layers.GlobalAveragePooling2D()(output)
    output = tf.keras.layers.Dense(units=number_of_dense_units,
                                   kernel_initializer=kernel_initializer,
                                   activation='relu')(output)
    output = tf.keras.layers.Dropout(dropout_rate)(output)
    output = tf.keras.layers.Dense(units=number_of_classes, activation='softmax')(output)

    filter_strings = ['_'.join( [str(filter) for filter in filters] ) for filters in [module_1_filters, module_2_filters] ]
    dropout_rate_string = str(dropout_rate).replace('.', '_')
    class_weights_scale_string = str(class_weights_scale).replace('.', '_')
    name = '-'.join(['inception_like', *filter_strings, kernel_initializer, str(number_of_dense_units), dataset_handle, 
        dropout_rate_string, class_weights_scale_string] )

    if duplication_test:
        name = _duplication_test(name, duplication_test)

    return tf.keras.Model(inputs=input, outputs=output, name=name)


@gin.configurable
class TransferLearningModel(tf.keras.Model):
    '''Create a model with transfer learning.'''

    model_types = ['resnet50', 'inceptionv3']


    def __init__(self, model_type, input_shape, number_of_classes, first_fine_tune_layer, duplication_test=None, **kwargs):

        assert number_of_classes > 0

        dropout_rate =        kwargs.get('dropout_rate', 0.2)
        dataset_handle =      kwargs.get('dataset_handle', '')
        class_weights_scale = kwargs.get('class_weights_scale', 0.0)

        dropout_rate_string = str(dropout_rate).replace('.', '_')
        class_weights_scale_string = str(class_weights_scale).replace('.', '_')

        name = '-'.join( [model_type, dataset_handle, dropout_rate_string, class_weights_scale_string] )
        if duplication_test:
            name = _duplication_test(name, duplication_test)

        super(TransferLearningModel, self).__init__(name=name)

        input_tensor = tf.keras.Input(shape=input_shape)

        if model_type == 'resnet50':
            self.base_model = tf.keras.applications.ResNet50(input_shape=input_shape,
                                                             include_top=False,
                                                             weights='imagenet',
                                                             input_tensor=input_tensor)
            if first_fine_tune_layer is None:
                first_fine_tune_layer = 150

        elif model_type == 'inceptionv3':
            self.base_model = tf.keras.applications.InceptionV3(input_shape=input_shape,
                                                                include_top=False,
                                                                weights='imagenet',
                                                                input_tensor=input_tensor)
            if first_fine_tune_layer is None:
                first_fine_tune_layer = 300

        self.base_model.trainable = False
        self.first_fine_tune_layer = first_fine_tune_layer

        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)
        self.classification_layer = tf.keras.layers.Dense(units=number_of_classes, activation='softmax')


    def unfreeze_top_layers(self):
        '''The RESNET50 model    -> 175 layers, excluding top.
           The Inceptionv3 model -> 311 layers, excluding top (it has BatchNormalizationLayers)'''

        for layer in self.base_model.layers[self.first_fine_tune_layer:]:
            layer.trainable = True


    def call(self, inputs, training=False):
        # Tensorflow tutorial:
        # When you set layer.trainable = False, the BatchNormalization layer will run in inference mode,
        # and will not update its mean and variance statistics.
        # When you unfreeze a model that contains BatchNormalization layers in order to do fine-tuning,
        # you should keep the BatchNormalization layers in inference mode by passing training = False when calling the base model.
        # Otherwise, the updates applied to the non-trainable weights will destroy what the model has learned.
        x = self.base_model(inputs=inputs, training=False)
        x = self.global_average_layer(x)
        if training:
            x = self.dropout_layer(x)
        return self.classification_layer(x)


    def print_base_model_layers(self):
        for i, layer in enumerate(self.base_model.layers):
            tf.print(i, layer.name, layer.output_shape)


    def build_graph(self, raw_shape):
        x = tf.keras.layers.Input(shape=raw_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


    def expand_base_model_and_build_functional_model(self):
        base_model_conv_layer = None

        for layer in reversed(self.base_model.layers):
            if len(layer.output_shape) == 4:
                base_model_conv_layer = layer.name
                break

        x_2 = self.global_average_layer(self.base_model.get_layer(base_model_conv_layer).output)
        out = self.classification_layer(x_2)
        return tf.keras.Model(inputs=self.base_model.input, outputs=out)
