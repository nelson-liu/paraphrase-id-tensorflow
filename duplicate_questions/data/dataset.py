import codecs
import itertools
import logging

import tqdm

from .instances.instance import Instance

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Dataset:
    """
    A collection of Instances. This base class has general methods that apply
    to all collections of Instances. That basically is just methods that
    operate on sets, like merging and truncating.

    """
    def __init__(self, instances):
        """
        Construct a dataset from a List of Instances.

        Parameters
        ----------
        instances: List of Instance
            The list of instances to build a Dataset from.
        """
        if not isinstance(instances, list):
            raise ValueError("Expected instances to be type "
                             "List, found {} of type "
                             "{}".format(instances, type(instances)))
        if not isinstance(instances[0], Instance):
            raise ValueError("Expected instances to be a List "
                             "of Instances, but the first element "
                             "of the input list was {} of type "
                             "{}".format(instances[0], type(instances[0])))
        self.instances = instances

    def merge(self, other):
        """
        Combine two datasets. If you call try to merge two Datasets of the same
        subtype, you will end up with a Dataset of the same type (i.e., calling
        IndexedDataset.merge() with another IndexedDataset will return an
        IndexedDataset). If the types differ, this method currently raises an
        error, because the underlying Instance objects are not currently type
        compatible.
        """
        if type(self) is type(other):
            return self.__class__(self.instances + other.instances)
        else:
            raise ValueError("Cannot merge datasets with different types")

    def truncate(self, max_instances):
        """
        Truncate the dataset to a fixed size.

        Parameters
        ----------
        max_instances: int
            The maximum amount of instances allowed in this dataset. If there
            are more instances than `max_instances` in this dataset, we
            return a new dataset with a random subset of size `max_instances`.
            If there are fewer than `max_instances` already, we just return
            self.
        """
        if not isinstance(max_instances, int):
            raise ValueError("Expected max_instances to be type "
                             "int, found {} of type "
                             "{}".format(max_instances, type(max_instances)))
        if max_instances < 1:
            raise ValueError("max_instances must be at least 1"
                             ", found {}".format(max_instances))
        if len(self.instances) <= max_instances:
            return self
        new_instances = [i for i in self.instances]
        return self.__class__(new_instances[:max_instances])


class TextDataset(Dataset):
    """
    A Dataset of TextInstances, with a few helper methods. TextInstances aren't
    useful for much until they've been indexed. So this class just has methods
    to read in data from a file and converting it into other kinds of Datasets.
    """
    def __init__(self, instances):
        """
        Construct a new TextDataset

        Parameters
        ----------
        instances: List of TextInstance
            A list of TextInstances to construct
            the TextDataset from.
        """
        super(TextDataset, self).__init__(instances)

    def to_indexed_dataset(self, data_indexer):
        """
        Converts the Dataset into an IndexedDataset, given a DataIndexer.

        Parameters
        ----------
        data_indexer: DataIndexer
            The DataIndexer to use in converting words to indices.
        """
        indexed_instances = [instance.to_indexed_instance(data_indexer) for
                             instance in tqdm.tqdm(self.instances)]
        return IndexedDataset(indexed_instances)

    @staticmethod
    def read_from_file(filename, instance_class):
        """
        Read a dataset (basically a list of Instances) from
        a data file.

        Parameters
        ----------
        filename: str
            The string filename from which to read the instances.

        instance_class: Instance
            The Instance class to create from these lines.

        Returns
        -------
        text_dataset: TextDataset
           A new TextDataset with the instances read from the file.
        """
        if not isinstance(filename, str):
            raise ValueError("Expected filename to be a string, "
                             "but was {} of type "
                             "{}".format(filename, type(filename)))
        lines = [x.strip() for x
                 in tqdm.tqdm(codecs.open(filename, "r",
                                          "utf-8").readlines())]
        return TextDataset.read_from_lines(lines, instance_class)

    @staticmethod
    def read_from_lines(lines, instance_class):
        """
        Read a dataset (basically a list of Instances) from
        a data file.

        Parameters
        ----------
        lines: List of str
            A list containing string representations of each
            line in the file.

        instance_class: Instance
            The Instance class to create from these lines.

        Returns
        -------
        text_dataset: TextDataset
           A new TextDataset with the instances read from the list.
        """
        if not isinstance(lines, list):
            raise ValueError("Expected lines to be a list, "
                             "but was {} of type "
                             "{}".format(lines, type(lines)))
        if not isinstance(lines[0], str):
            raise ValueError("Expected lines to be a list of strings, "
                             "but the first element of the list was {} "
                             "of type {}".format(lines[0], type(lines[0])))
        instances = [instance_class.read_from_line(line) for line in lines]
        labels = [(x.label, x) for x in instances]
        labels.sort(key=lambda x: str(x[0]))
        label_counts = [(label, len([x for x in group]))
                        for label, group
                        in itertools.groupby(labels, lambda x: x[0])]
        label_count_str = str(label_counts)
        if len(label_count_str) > 100:
            label_count_str = label_count_str[:100] + '...'
        logger.info("Finished reading dataset; label counts: %s",
                    label_count_str)
        return TextDataset(instances)


class IndexedDataset(Dataset):
    """
    A Dataset of IndexedInstances, with some helper methods.

    IndexedInstances have text sequences replaced with lists of word indices,
    and are thus able to be padded to consistent lengths and converted to
    training inputs.

    """
    def __init__(self, instances):
        super(IndexedDataset, self).__init__(instances)

    def max_lengths(self):
        max_lengths = {}
        lengths = [instance.get_lengths() for instance in self.instances]
        if not lengths:
            return max_lengths
        for key in lengths[0]:
            max_lengths[key] = max(x[key] if key in x else 0 for x in lengths)
        return max_lengths

    def pad_instances(self, max_lengths=None):
        """
        Make all of the IndexedInstances in the dataset have the same length
        by padding them (in the front) with zeros.

        If max_length is given for a particular dimension, we will pad all
        instances to that length (including left-truncating instances if
        necessary). If not, we will find the longest instance and pad all
        instances to that length. Note that max_lengths is a _List_, not an int
        - there could be several dimensions on which we need to pad, depending
        on what kind of instance we are dealing with.

        This method _modifies_ the current object, it does not return a new
        IndexedDataset.
        """
        # First we need to decide _how much_ to pad. To do that, we find the
        # max length for all relevant padding decisions from the instances
        # themselves. Then we check whether we were given a max length for a
        # particular dimension. If we were, we use that instead of the
        # instance-based one.
        logger.info("Getting max lengths from instances")
        instance_max_lengths = self.max_lengths()
        logger.info("Instance max lengths: %s", str(instance_max_lengths))
        lengths_to_use = {}
        for key in instance_max_lengths:
            if max_lengths and max_lengths[key] is not None:
                lengths_to_use[key] = max_lengths[key]
            else:
                lengths_to_use[key] = instance_max_lengths[key]

        logger.info("Now actually padding instances to length: %s",
                    str(lengths_to_use))
        for instance in tqdm.tqdm(self.instances):
            instance.pad(lengths_to_use)

    def as_training_data(self):
        """
        Takes each IndexedInstance and converts it into (inputs, labels),
        according to the Instance's as_training_data() method. Note that
        you might need to call numpy.asarray() on the results of this; we
        don't do that for you, because the inputs might be complicated.
        """
        inputs = []
        labels = []
        instances = self.instances
        for instance in instances:
            instance_inputs, label = instance.as_training_data()
            inputs.append(instance_inputs)
            labels.append(label)
        return inputs, labels

    def as_testing_data(self):
        """
        Takes each IndexedInstance and converts it into inputs,
        according to the Instance's as_testing_data() method. Note that
        you might need to call numpy.asarray() on the results of this; we
        don't do that for you, because the inputs might be complicated.
        """
        inputs = []
        instances = self.instances
        for instance in instances:
            instance_inputs = instance.as_testing_data()
            inputs.append(instance_inputs)
        return inputs
