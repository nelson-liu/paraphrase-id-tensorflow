import logging
import numpy as np
from itertools import islice

from .data_indexer import DataIndexer
from .dataset import TextDataset

logger = logging.getLogger(__name__)


class DataManager():
    """
    The goal of this class is to act as a centralized place
    to do high-level operations on your data (i.e. loading them from filename
    to NumPy arrays).
    """

    def __init__(self, instance_type):
        self.data_indexer = DataIndexer()
        self.instance_type = instance_type
        # List of lists of filenames that self.data_indexer
        # has been fit on.
        self.data_indexer_fitted = False
        self.training_data_max_lengths = {}

    @staticmethod
    def get_batch_generator(get_instance_generator, batch_size):
        """
        Convenience function that, when called, produces a generator that yields
        individual instances as numpy arrays into a generator
        that yields batches of instances.

        Parameters
        ----------
        instance_generator: numpy array generator
            The instance_generator should be an infinite generator that outputs
            individual training instances (as numpy arrays in this codebase,
            but any iterable works). The expected format is:
            ((input0, input1,...), (target0, target1, ...))

        batch_size: int, optional
            The size of each batch. Depending on how many
            instances there are in the dataset, the last batch
            may have less instances.

        Returns
        -------
        output: returns a tuple of 2 tuples
            The expected return schema is:
            ((input0, input1, ...), (target0, target1, ...),
            where each of "input*" and "target*" are numpy arrays.
            The number of rows in each input and target numpy array
            should be the same as the batch size.
        """

        # batched_instances is a list of batch_size instances, where each
        # instance is a tuple ((inputs), targets)
        instance_generator = get_instance_generator()
        batched_instances = list(islice(instance_generator, batch_size))
        while batched_instances:
            # Take the batched instances and create a batch from it.
            # The batch is a tuple ((inputs), targets), where (inputs)
            # can be (inputs0, inputs1, etc...). each of "inputs*" and
            # "targets" are numpy arrays.
            flattened = ([ins[0] for ins in batched_instances],
                         [ins[1] for ins in batched_instances])
            flattened_inputs, flattened_targets = flattened
            batch_inputs = tuple(map(np.array, tuple(zip(*flattened_inputs))))
            batch_targets = tuple(map(np.array, tuple(zip(*flattened_targets))))
            yield batch_inputs, batch_targets
            batched_instances = list(islice(instance_generator, batch_size))

    def get_train_data_from_file(self, filenames, min_count=1,
                                 max_instances=None,
                                 max_lengths=None, pad=True, mode="word"):
        """
        Given a filename or list of filenames, return a generator for producing
        individual instances of data ready for use in a model read from those
        file(s).

        Given a string path to a file in the format accepted by the instance,
        we fit the data_indexer word dictionary on it. Next, we use this
        DataIndexer to convert the instance into IndexedInstances (replacing
        words with integer indices).

        This function returns a function to construct generators that take
        these IndexedInstances, pads them to the appropriate lengths (either the
        maximum lengths in the dataset, or lengths specified in the constructor),
        and then converts them to NumPy arrays suitable for training with
        instance.as_training_data. The generator yields one instance at a time,
        represented as tuples of (inputs, labels).

        Parameters
        ----------
        filenames: List[str]
            A collection of filenames to read the specific self.instance_type
            from, line by line.

        min_count: int, default=1
            The minimum number of times a word must occur in order
            to be indexed.

        max_instances: int, default=None
            If not None, the maximum number of instances to produce as
            training data. If necessary, we will truncate the dataset.
            Useful for debugging and making sure things work with small
            amounts of data.

        max_lengths: dict from str to int, default=None
            If not None, the max length of a sequence in a given dimension.
            The keys for this dict must be in the same format as
            the instances' get_lengths() function. These are the lengths
            that the instances are padded or truncated to.

        pad: boolean, default=True
            If True, pads or truncates the instances to either the input
            max_lengths or max_lengths across the train filenames. If False,
            no padding or truncation is applied.

        mode: str, optional (default="word")
            String describing whether to return the word-level representations,
            character-level representations, or both. One of "word",
            "character", or "word+character"

        Returns
        -------
        output: returns a function to construct a train data generator
            This returns a function that can be called to produce a tuple of
            (instance generator, train_set_size). The instance generator
            outputs instances as generated by the as_training_data function
            of the underlying instance class. The train_set_size is the number
            of instances in the train set, which can be used to initialize a
            progress bar.
        """
        if self.data_indexer_fitted:
            raise ValueError("You have already called get_train_data from "
                             "this DataManager, so you cannnot do it again. "
                             "If you want to train on multiple datasets, pass "
                             "in a list of files.")
        logger.info("Getting training data from {}".format(filenames))
        training_dataset = TextDataset.read_from_file(filenames,
                                                      self.instance_type)
        if max_instances:
            logger.info("Truncating the training dataset "
                        "to {} instances".format(max_instances))
            training_dataset = training_dataset.truncate(max_instances)

        training_dataset_size = len(training_dataset.instances)

        # Since this is data for training, we fit the data indexer
        logger.info("Fitting data indexer word "
                    "dictionary, min_count is {}.".format(min_count))
        self.data_indexer.fit_word_dictionary(training_dataset,
                                              min_count=min_count)
        self.data_indexer_fitted = True

        # With our fitted data indexer, we convert the dataset
        # from string tokens to numeric int indices.
        logger.info("Indexing dataset")
        indexed_training_dataset = training_dataset.to_indexed_dataset(
            self.data_indexer)

        # We now need to check if the user specified max_lengths for
        # the instance, and accordingly truncate or pad if applicable. If
        # max_lengths is None for a given string key, we assume that no
        # truncation is to be done and the max lengths should be read from the
        # instances.
        if not pad and max_lengths:
            raise ValueError("Passed in max_lengths {}, but set pad to false. "
                             "Did you mean to do this?".format(max_lengths))

        # Get max lengths from the dataset
        dataset_max_lengths = indexed_training_dataset.max_lengths()
        logger.info("Instance max lengths {}".format(dataset_max_lengths))
        max_lengths_to_use = dataset_max_lengths
        if pad:
            # If the user set max lengths, iterate over the
            # dictionary provided and verify that they did not
            # pass any keys to truncate that are not in the instance.
            if max_lengths is not None:
                for input_dimension, length in max_lengths.items():
                    if input_dimension in dataset_max_lengths:
                        max_lengths_to_use[input_dimension] = length
                    else:
                        raise ValueError("Passed a value for the max_lengths "
                                         "that does not exist in the "
                                         "instance. Improper input length "
                                         "dimension (key) we found was {}, "
                                         "lengths dimensions in the instance "
                                         "are {}".format(
                                             input_dimension,
                                             dataset_max_lengths.keys()))
            logger.info("Padding lengths to "
                        "length: {}".format(str(max_lengths_to_use)))
        self.training_data_max_lengths = max_lengths_to_use

        # This is a hack to get the function to run the code above immediately,
        # instead of doing the standard python generator lazy-ish evaluation.
        # This is necessary to set the class variables ASAP.
        def _get_train_data_generator():
            for indexed_instance in indexed_training_dataset.instances:
                # For each instance, we want to pad or truncate if applicable
                if pad:
                    indexed_instance.pad(max_lengths_to_use)
                # Now, we want to take the instance and convert it into
                # NumPy arrays suitable for training.
                inputs, labels = indexed_instance.as_training_data(mode=mode)
                yield inputs, labels
        return _get_train_data_generator, training_dataset_size

    def get_validation_data_from_file(self, filenames, max_instances=None,
                                      max_lengths=None, pad=True, mode="word"):
        """
        Given a filename or list of filenames, return a generator for producing
        individual instances of data ready for use as validation data in a
        model read from those file(s).

        Given a string path to a file in the format accepted by the instance,
        we use a data_indexer previously fitted on train data. Next, we use
        this DataIndexer to convert the instance into IndexedInstances
        (replacing words with integer indices).

        This function returns a function to construct generators that take
        these IndexedInstances, pads them to the appropriate lengths (either the
        maximum lengths in the dataset, or lengths specified in the constructor),
        and then converts them to NumPy arrays suitable for training with
        instance.as_validation_data. The generator yields one instance at a time,
        represented as tuples of (inputs, labels).

        Parameters
        ----------
        filenames: List[str]
            A collection of filenames to read the specific self.instance_type
            from, line by line.

        max_instances: int, default=None
            If not None, the maximum number of instances to produce as
            training data. If necessary, we will truncate the dataset.
            Useful for debugging and making sure things work with small
            amounts of data.

        max_lengths: dict from str to int, default=None
            If not None, the max length of a sequence in a given dimension.
            The keys for this dict must be in the same format as
            the instances' get_lengths() function. These are the lengths
            that the instances are padded or truncated to.

        pad: boolean, default=True
            If True, pads or truncates the instances to either the input
            max_lengths or max_lengths used on the train filenames. If False,
            no padding or truncation is applied.

        mode: str, optional (default="word")
            String describing whether to return the word-level representations,
            character-level representations, or both. One of "word",
            "character", or "word+character"

        Returns
        -------
        output: returns a function to construct a validation data generator
            This returns a function that can be called to produce a tuple of
            (instance generator, validation_set_size). The instance generator
            outputs instances as generated by the as_validation_data function
            of the underlying instance class. The validation_set_size is the number
            of instances in the validation set, which can be used to initialize a
            progress bar.
        """
        logger.info("Getting validation data from {}".format(filenames))
        validation_dataset = TextDataset.read_from_file(filenames,
                                                        self.instance_type)
        if max_instances:
            logger.info("Truncating the validation dataset "
                        "to {} instances".format(max_instances))
            validation_dataset = validation_dataset.truncate(max_instances)

        validation_dataset_size = len(validation_dataset.instances)

        # With our fitted data indexer, we we convert the dataset
        # from string tokens to numeric int indices.
        logger.info("Indexing validation dataset with "
                    "DataIndexer fit on train data.")
        indexed_validation_dataset = validation_dataset.to_indexed_dataset(
            self.data_indexer)

        # We now need to check if the user specified max_lengths for
        # the instance, and accordingly truncate or pad if applicable. If
        # max_lengths is None for a given string key, we assume that no
        # truncation is to be done and the max lengths should be taken from
        # the train dataset.
        if not pad and max_lengths:
            raise ValueError("Passed in max_lengths {}, but set pad to false. "
                             "Did you mean to do this?".format(max_lengths))
        if pad:
            # Get max lengths from the train dataset
            training_data_max_lengths = self.training_data_max_lengths
            logger.info("Max lengths in training "
                        "data: {}".format(training_data_max_lengths))

            max_lengths_to_use = training_data_max_lengths
            # If the user set max lengths, iterate over the
            # dictionary provided and verify that they did not
            # pass any keys to truncate that are not in the instance.
            if max_lengths is not None:
                for input_dimension, length in max_lengths.items():
                    if input_dimension in training_data_max_lengths:
                        max_lengths_to_use[input_dimension] = length
                    else:
                        raise ValueError("Passed a value for the max_lengths "
                                         "that does not exist in the "
                                         "instance. Improper input length "
                                         "dimension (key) we found was {}, "
                                         "lengths dimensions in the instance "
                                         "are {}".format(
                                             input_dimension,
                                             training_data_max_lengths.keys()))
            logger.info("Padding lengths to "
                        "length: {}".format(str(max_lengths_to_use)))

        # This is a hack to get the function to run the code above immediately,
        # instead of doing the standard python generator lazy-ish evaluation.
        # This is necessary to set the class variables ASAP.
        def _get_validation_data_generator():
            for indexed_val_instance in indexed_validation_dataset.instances:
                # For each instance, we want to pad or truncate if applicable
                if pad:
                    indexed_val_instance.pad(max_lengths_to_use)
                # Now, we want to take the instance and convert it into
                # NumPy arrays suitable for validation.
                inputs, labels = indexed_val_instance.as_training_data(mode=mode)

                yield inputs, labels
        return _get_validation_data_generator, validation_dataset_size

    def get_test_data_from_file(self, filenames, max_instances=None,
                                max_lengths=None, pad=True, mode="word"):
        """
        Given a filename or list of filenames, return a generator for producing
        individual instances of data ready for use as model test data.

        Given a string path to a file in the format accepted by the instance,
        we use a data_indexer previously fitted on train data. Next, we use
        this DataIndexer to convert the instance into IndexedInstances
        (replacing words with integer indices).

        This function returns a function to construct generators that take
        these IndexedInstances, pads them to the appropriate lengths (either the
        maximum lengths in the dataset, or lengths specified in the constructor),
        and then converts them to NumPy arrays suitable for training with
        instance.as_testinging_data. The generator yields one instance at a time,
        represented as tuples of (inputs, labels).

        Parameters
        ----------
        filenames: List[str]
            A collection of filenames to read the specific self.instance_type
            from, line by line.

        max_instances: int, default=None
            If not None, the maximum number of instances to produce as
            training data. If necessary, we will truncate the dataset.
            Useful for debugging and making sure things work with small
            amounts of data.

        max_lengths: dict from str to int, default=None
            If not None, the max length of a sequence in a given dimension.
            The keys for this dict must be in the same format as
            the instances' get_lengths() function. These are the lengths
            that the instances are padded or truncated to.

        pad: boolean, default=True
            If True, pads or truncates the instances to either the input
            max_lengths or max_lengths used on the train filenames. If False,
            no padding or truncation is applied.

        mode: str, optional (default="word")
            String describing whether to return the word-level representations,
            character-level representations, or both. One of "word",
            "character", or "word+character"

        Returns
        -------
        output: returns a function to construct a test data generator
            This returns a function that can be called to produce a tuple of
            (instance generator, test_set_size). The instance generator
            outputs instances as generated by the as_testing_data function
            of the underlying instance class. The test_set_size is the number
            of instances in the test set, which can be used to initialize a
            progress bar.
        """
        logger.info("Getting test data from {}".format(filenames))
        test_dataset = TextDataset.read_from_file(filenames,
                                                  self.instance_type)
        if max_instances:
            logger.info("Truncating the test dataset "
                        "to {} instances".format(max_instances))
            test_dataset = test_dataset.truncate(max_instances)

        test_dataset_size = len(test_dataset.instances)

        # With our fitted data indexer, we we convert the dataset
        # from string tokens to numeric int indices.
        logger.info("Indexing test dataset with DataIndexer "
                    "fit on train data.")
        indexed_test_dataset = test_dataset.to_indexed_dataset(
            self.data_indexer)

        # We now need to check if the user specified max_lengths for
        # the instance, and accordingly truncate or pad if applicable. If
        # max_lengths is None for a given string key, we assume that no
        # truncation is to be done and the max lengths should be taken from
        # the train dataset.
        if not pad and max_lengths:
            raise ValueError("Passed in max_lengths {}, but set pad to false. "
                             "Did you mean to do this?".format(max_lengths))
        if pad:
            # Get max lengths from the train dataset
            training_data_max_lengths = self.training_data_max_lengths
            logger.info("Max lengths in training "
                        "data: {}".format(training_data_max_lengths))

            max_lengths_to_use = training_data_max_lengths
            # If the user set max lengths, iterate over the
            # dictionary provided and verify that they did not
            # pass any keys to truncate that are not in the instance.
            if max_lengths is not None:
                for input_dimension, length in max_lengths.items():
                    if input_dimension in training_data_max_lengths:
                        max_lengths_to_use[input_dimension] = length
                    else:
                        raise ValueError("Passed a value for the max_lengths "
                                         "that does not exist in the "
                                         "instance. Improper input length "
                                         "dimension (key) we found was {}, "
                                         "lengths dimensions in the instance "
                                         "are {}".format(
                                             input_dimension,
                                             training_data_max_lengths.keys()))
            logger.info("Padding lengths to "
                        "length: {}".format(str(max_lengths_to_use)))

        # This is a hack to get the function to run the code above immediately,
        # instead of doing the standard python generator lazy-ish evaluation.
        # This is necessary to set the class variables ASAP.
        def _get_test_data_generator():
            for indexed_test_instance in indexed_test_dataset.instances:
                # For each instance, we want to pad or truncate if applicable
                if pad:
                    indexed_test_instance.pad(max_lengths_to_use)
                # Now, we want to take the instance and convert it into
                # NumPy arrays suitable for validation.
                inputs = indexed_test_instance.as_testing_data(mode=mode)

                yield inputs
        return _get_test_data_generator, test_dataset_size
