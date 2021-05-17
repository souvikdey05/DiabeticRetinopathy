import io
import logging
import tensorflow as tf
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import datetime
import pathlib

evaluation_logger = logging.getLogger('evaluation')

class GradCAM:
    def __init__(self, model, run_paths, layer_name=None):
        self.model = model  # this is the conv model + output pred model
        self.saved_model_path = run_paths['trained_models_directory']
        self.model = self._load_saved_models() # load the trained model

        self.layer_name = layer_name

        # if the layer name is None, find the target output layer
        if self.layer_name is None:
            self.layer_name = self._find_last_conv_layer()
        else:
            conv_layer = self.model.get_layer(self.layer_name)
            if len(conv_layer.output_shape) != 4:
                raise ValueError("The layer name doesn't correspond to convolution layer")
        # print(f"self.layer_name = {self.layer_name}")

        # Summary Writer
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        summary_directory = pathlib.Path(run_paths['evaluation_log']).parent / current_time
        self.vis_summary_writer = tf.summary.create_file_writer(logdir=str(summary_directory / 'visualization'))

    def _load_saved_models(self):
        # Load the saved models
        loaded_model = self.model  # loaded_model: Model which accepts the data from the saved models.
        model_path = self.saved_model_path / loaded_model.name

        if model_path.exists():
            evaluation_logger.info(f"Loading model {loaded_model.name} from path {str(model_path)}")
            model_path = str(model_path / loaded_model.name) + '-1'
            # The suffix '-1' is hardcoded, because if models are selected from a 'trained_models' dir. then there should only be one checkpoint.
            # If it is necessary to accept any checkpoint, then the suffix has to be read per model from the files in the dir..

            tf.train.Checkpoint(model=loaded_model).restore(save_path=model_path)
        else:
            evaluation_logger.error(
                    f"Could not find the path {model_path}. Try training first")

        return loaded_model

    def _find_last_conv_layer(self):

        """ find the last Conv Layer"""
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name

        raise ValueError("No Convolution layer found")

    def _populate_gradcam_heatmap(self, image, target_class_index, compare_prediction=False, eps=1e-15):

        """
        Returns the heat map of the gradcam wrt to the given class index and
        also wrt to the top predicted class
        """
        # First, we create a model that maps the input image to the activations
        # of the last conv layer and the output of the softmax activations from the model
        grad_cam_model = tf.keras.Model(inputs=[self.model.input],
                                        outputs=[self.model.get_layer(self.layer_name).output,
                                                 self.model.output])

        # Compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape(persistent=True) as tape:
            # Assume image is already casted and normalized
            inputs = image
            (conv_outputs, predictions) = grad_cam_model(inputs)
            tape.watch(conv_outputs)
            loss_wrt_given_class_index = predictions[:, target_class_index]

            if compare_prediction:
                top_pred_index = tf.argmax(predictions[0])
                loss_wrt_top_class_index = predictions[:, top_pred_index]

        # This is the gradient of the given class with regard to
        # the output feature map of the last conv layer
        grads_wrt_given_class_index = tape.gradient(loss_wrt_given_class_index, conv_outputs)

        if compare_prediction:
            # This is the gradient of the top predicted class with regard to
            # the output feature map of the last conv layer
            grads_wrt_top_class_index = tape.gradient(loss_wrt_top_class_index, conv_outputs)

        # compute the guided gradients
        cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
        cast_grads_wrt_given_class_index = tf.cast(grads_wrt_given_class_index > 0, "float32")
        if compare_prediction:
            cast_grads_wrt_top_class_index = tf.cast(grads_wrt_top_class_index > 0, "float32")

        guided_grads_wrt_given_class_index = cast_conv_outputs * cast_grads_wrt_given_class_index * grads_wrt_given_class_index

        if compare_prediction:
            guided_grads_wrt_top_class_index = cast_conv_outputs * cast_grads_wrt_top_class_index * grads_wrt_top_class_index

        # the convolution and guided gradients have a batch dimension (which we don't need)
        conv_outputs = conv_outputs[0]
        guided_grads_wrt_given_class_index = guided_grads_wrt_given_class_index[0]
        if compare_prediction:
            guided_grads_wrt_top_class_index = guided_grads_wrt_top_class_index[0]

        # compute the average of the gradient values, and using them as weights,
        # compute the class activation maps of the filters with respect to the weights
        weights_grads_wrt_given_class_index = tf.reduce_mean(guided_grads_wrt_given_class_index, axis=(0, 1))

        if compare_prediction:
            weights_grads_wrt_top_class_index = tf.reduce_mean(guided_grads_wrt_top_class_index, axis=(0, 1))

        guided_cam_wrt_given_class_index = tf.reduce_sum(tf.multiply(weights_grads_wrt_given_class_index,
                                                                     conv_outputs),
                                                         axis=-1)

        if compare_prediction:
            guided_cam_wrt_top_class_index = tf.reduce_sum(tf.multiply(weights_grads_wrt_top_class_index,
                                                                       conv_outputs),
                                                           axis=-1)

        # convert to numpy
        guided_cam_wrt_given_class_index = guided_cam_wrt_given_class_index.numpy()
        if compare_prediction:
            guided_cam_wrt_top_class_index = guided_cam_wrt_top_class_index.numpy()

        # normalize the heatmap such that all values lie in the range [0, 1],
        # scale the resulting values to the range [0, 255] and then convert to an unsigned 8-bit integer
        numer = guided_cam_wrt_given_class_index - np.min(guided_cam_wrt_given_class_index)
        denom = (np.max(guided_cam_wrt_given_class_index) - np.min(guided_cam_wrt_given_class_index)) + eps
        heatmap_wrt_given_class_index = numer / denom

        if compare_prediction:
            numer = guided_cam_wrt_top_class_index - np.min(guided_cam_wrt_top_class_index)
            denom = (np.max(guided_cam_wrt_top_class_index) - np.min(guided_cam_wrt_top_class_index)) + eps
            heatmap_wrt_top_class_index = numer / denom

        if compare_prediction:
            return heatmap_wrt_given_class_index, heatmap_wrt_top_class_index, top_pred_index
        return heatmap_wrt_given_class_index

    def _superimpose_heatmap_on_image(self, heatmap, image, alpha=0.4):
        # rescale the heatmap and convert to uint8 type
        heatmap = np.uint8(heatmap * 255)

        # We use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

        image = np.uint8(image * 255)

        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
        axs[0].set_title('original image', fontsize=8)
        axs[0].imshow(image)

        axs[1].set_title('heatmap image', fontsize=8)
        axs[1].imshow(jet_heatmap)

        axs[2].set_title('superimposed image', fontsize=8)
        axs[2].imshow(image)
        axs[2].imshow(jet_heatmap, alpha=alpha)

        return fig, axs

    def visualize(self, dataset):
        evaluation_logger.info("Start visualization for model {}".format(self.model.name))

        for image, image_name, retina_grade, _ in dataset.unbatch():
            image_expanded_dim = np.expand_dims(image, axis=0)  # make it batch dimension for the model
            heatmap_wrt_given_class_index, \
            heatmap_wrt_top_class_index, pred_class = self._populate_gradcam_heatmap(image_expanded_dim,
                                                                                     target_class_index=retina_grade,
                                                                                     compare_prediction=True)

            fig_wrt_given_class_index, axs = self._superimpose_heatmap_on_image(
                heatmap_wrt_given_class_index, image)
            fig_wrt_given_class_index.suptitle(f'For class {retina_grade}', fontsize=10)

            fig_wrt_top_class_index, axs = self._superimpose_heatmap_on_image(
                heatmap_wrt_top_class_index, image)
            fig_wrt_top_class_index.suptitle(f'For predicted class {pred_class}', fontsize=10)
            # plt.show()

            with self.vis_summary_writer.as_default():
                tf.summary.image('GradCAM of {0} wrt to true label {1}'.format(image_name, retina_grade),
                                 self._plot_to_image(fig_wrt_given_class_index),
                                 step=0)
                tf.summary.image('GradCAM of {0} wrt to pred label {1}'.format(image_name, pred_class),
                                 self._plot_to_image(fig_wrt_top_class_index),
                                 step=0)

        evaluation_logger.info("Visualization finished for model {}".format(self.model.name))

    def _plot_to_image(self, figure):
        """
        Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call.
        """

        buf = io.BytesIO()

        # Use plt.savefig to save the plot to a PNG in memory.
        plt.savefig(buf, format='png')

        # Closing the figure prevents it from being displayed directly inside the notebook.
        plt.close(figure)
        buf.seek(0)

        # Use tf.image.decode_png to convert the PNG buffer
        # to a TF image. Make sure you use 4 channels.
        image = tf.image.decode_png(buf.getvalue(), channels=4)

        # Use tf.expand_dims to add the batch dimension
        image = tf.expand_dims(image, 0)

        return image