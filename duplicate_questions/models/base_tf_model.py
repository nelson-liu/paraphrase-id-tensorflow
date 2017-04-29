import logging
import math
import numpy as np
import tensorflow as tf
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaseTFModel:
    """
    This class is a base model class for Tensorflow that other Tensorflow
    models should inherit from. It defines a unifying API for training and
    prediction.

    Parameters
    ----------
    mode: str
        One of [train|predict], to indicate what you want the model to do.
    """
    def __init__(self, mode):
        self.mode = mode
        self.global_step = tf.get_variable(name="global_step",
                                           shape=[],
                                           dtype='int32',
                                           initializer=tf.constant_initializer(0),
                                           trainable=False)

        # Outputs from the model
        self.y_pred = None
        self.loss = None
        self.accuracy = None

        self.training_op = None
        self.summary_op = None

    def _create_placeholders(self):
        raise NotImplementedError

    def _build_forward(self):
        raise NotImplementedError

    def build_graph(self, seed=0):
        """
        Build the graph, ostensibly by setting up the placeholders and then
        creating the forward pass.

        Parameters
        ----------
        seed: int, optional (default=0)
             The graph-level seed to use when building the graph.
        """
        logger.info("Building graph...")
        tf.set_random_seed(seed)
        self._create_placeholders()
        self._build_forward()

    def _get_train_feed_dict(self, batch):
        """
        Given a train batch from a batch generator,
        return the appropriate feed_dict to pass to the
        model during training.

        Parameters
        ----------
        batch: tuple of NumPy arrays
            A tuple of NumPy arrays containing the data necessary
            to train.
        """
        raise NotImplementedError

    def _get_validation_feed_dict(self, batch):
        """
        Given a validation batch from a batch generator,
        return the appropriate feed_dict to pass to the
        model during validation.

        Parameters
        ----------
        batch: tuple of NumPy arrays
            A tuple of NumPy arrays containing the data necessary
            to validate.
        """
        raise NotImplementedError

    def _get_test_feed_dict(self, batch):
        """
        Given a test batch from a batch generator,
        return the appropriate feed_dict to pass to the
        model during prediction.

        Parameters
        ----------
        batch: tuple of NumPy arrays
            A tuple of NumPy arrays containing the data necessary
            to predict.
        """
        raise NotImplementedError

    def train(self,
              train_batch_generator, val_batch_generator,
              num_train_steps_per_epoch, num_epochs,
              num_val_steps, save_path, log_path,
              val_period=250, log_period=10, save_period=250,
              max_ckpts_to_keep=10, patience=0):
        """
        Train the model.

        Parameters
        ----------
        train_batch_generator: generator
            This generator should infinitely generate batches of instances
            for use in training.

        val_batch_generator: generator
            This generator should infinitely generate batches of instances
            for use in validation.

        num_train_steps_per_epoch: int
            The number of training steps after which an epoch has passed.

        num_epochs: int
            The number of epochs to train for.

        num_val_steps: int
            The number of batches after which the model has seen all of
            The validation data.

        save_path: str
            The input path to the tensorflow Saver responsible for
            checkpointing.

        log_path: str
            The input path to the tensorflow SummaryWriter responsible for
            logging the progress.

        val_period: int, optional (default=250)
            Number of steps between each evaluation of performance on the
            held-out validation set.

        log_period: int, optional (default=10)
            Number of steps between each summary op evaluation.

        save_period: int, optional (default=250)
            Number of steps between each model checkpoint.

        max_ckpts_to_keep: int, optional (default=10)
            The maximum number of model to checkpoints to keep.

        patience: int, optional (default=0)
            The number of epochs with no improvement in validation loss
            after which training will be stopped.
        """

        global_step = 0
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            # Set up the classes for logging to Tensorboard.
            train_writer = tf.summary.FileWriter(log_path + "/train",
                                                 sess.graph)
            val_writer = tf.summary.FileWriter(log_path + "/val",
                                               sess.graph)
            # Set up a Saver for periodically serializing the model.
            saver = tf.train.Saver(max_to_keep=max_ckpts_to_keep)

            total_num_train_steps = num_train_steps_per_epoch * num_epochs
            epoch_validation_losses = []
            # Iterate over a generator that returns batches.
            for batch in tqdm(train_batch_generator,
                              total=total_num_train_steps):
                global_step = sess.run(self.global_step)

                # is_epoch_end is True if we are at the end of an epoch.
                is_epoch_end = global_step % num_train_steps_per_epoch == 0

                inputs, targets = batch
                feed_dict = self._get_train_feed_dict(batch)

                # Do a gradient update, and log results to Tensorboard
                # if necessary.
                if global_step % log_period == 0:
                    train_loss, _, train_summary = sess.run(
                        [self.loss,
                         self.training_op,
                         self.summary_op],
                        feed_dict=feed_dict)
                    train_writer.add_summary(train_summary, global_step)
                else:
                    train_loss, _ = sess.run(
                        [self.loss,
                         self.training_op],
                        feed_dict=feed_dict)

                if global_step % val_period == 0 or is_epoch_end:
                    # Calculate the mean of the validation metrics
                    # over the validation set.
                    val_accuracies = []
                    val_losses = []
                    for iter_num, val_batch in enumerate(val_batch_generator):
                        feed_dict = self._get_validation_feed_dict(val_batch)
                        val_batch_acc, val_batch_loss = sess.run(
                            [self.accuracy, self.loss],
                            feed_dict=feed_dict)

                        val_accuracies.append(val_batch_acc)
                        val_losses.append(val_batch_loss)
                        if iter_num >= num_val_steps:
                            break

                    # Take the mean of the accuracies and losses.
                    mean_val_accuracy = np.mean(val_accuracies)
                    mean_val_loss = np.mean(val_losses)

                    # Create a new Summary object with mean_val accuracy
                    # and mean_val_loss and add it to Tensorboard.
                    val_summary = tf.Summary(value=[
                        tf.Summary.Value(tag="val_summaries/loss",
                                         simple_value=mean_val_loss),
                        tf.Summary.Value(tag="val_summaries/accuracy",
                                         simple_value=mean_val_accuracy)])
                    val_writer.add_summary(val_summary, global_step)

                # Write a model checkpoint if necessary.
                if global_step % save_period == 0:
                    saver.save(sess,
                               save_path,
                               global_step=global_step)

                if global_step >= total_num_train_steps:
                    logger.info("Finished {} epochs, which is {} train "
                                "steps".format(num_epochs,
                                               total_num_train_steps))
                    break

                if is_epoch_end:
                    # End of the epoch, so check validation loss
                    # and stop if applicable.

                    # Get the lowest validation loss in the last
                    # patience+1 epochs
                    patience_val_losses = epoch_validation_losses[-(patience + 1):]
                    if patience_val_losses:
                        min_patience_val_loss = min(patience_val_losses)
                    else:
                        min_patience_val_loss = math.inf
                    if min_patience_val_loss <= mean_val_loss:
                        # past loss was lower, so stop
                        logger.info("Validation loss of {} in last {} "
                                    "epochs, which is lower than current "
                                    "epoch validation loss of {}; stopping "
                                    "early.".format(min_patience_val_loss,
                                                    patience,
                                                    mean_val_loss))
                        break
                    epoch_validation_losses.append(mean_val_loss)

    def predict(self, test_batch_generator, model_load_dir,
                num_test_steps=None):
        """
        Load a serialized model and use it for prediction on a test
        set (from a finite generator).

        Parameters
        ----------
        test_batch_generator: generator
            This generator should generate batches of instances for use at
            test time. Unlike the train and validation generators, it should be
            finite.

        model_load_dir: str
            Path to a directory with serialized tensorflow checkpoints for the
            model to be run. The most recent checkpoint will be loaded and used
            for prediction.

        num_test_steps: int
            The number of steps (calculated by ceil(total # test examples / batch_size))
            in testing. This does not have any effect on how much of the test data
            is read; inference keeps going until the generator is exhausted. It
            is used to set a total for the progress bar.
        """
        logger.info("num_test_steps is not set, pass in a value "
                    "to show a progress bar.")
        with tf.Session() as sess:
            saver = tf.train.Saver()
            logger.info("Getting latest checkpoint in {}".format(model_load_dir))
            last_checkpoint = tf.train.latest_checkpoint(model_load_dir)
            logger.info("Attempting to load checkpoint at {}".format(last_checkpoint))
            saver.restore(sess, last_checkpoint)
            logger.info("Successfully loaded {}!".format(last_checkpoint))
            y_pred = []
            for batch in tqdm(test_batch_generator,
                              total=num_test_steps):
                feed_dict = self._get_test_feed_dict(batch)
                y_pred_batch = sess.run(self.y_pred, feed_dict=feed_dict)
                y_pred.append(y_pred_batch)
            y_pred_flat = np.concatenate(y_pred, axis=0)
        return y_pred_flat
