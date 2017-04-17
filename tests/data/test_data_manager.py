from numpy.testing import assert_allclose
import numpy as np
from overrides import overrides

from duplicate_questions.data.data_manager import DataManager
from duplicate_questions.data.instances.sts_instance import STSInstance

from ..common.test_case import DuplicateTestCase


class TestDataManagerTrain(DuplicateTestCase):
    @overrides
    def setUp(self):
        super(TestDataManagerTrain, self).setUp()
        self.write_duplicate_questions_train_file()
        self.data_manager = DataManager(STSInstance)

    def test_get_train_data_default(self):
        train_gen = self.data_manager.get_train_data_from_file(
            [self.TRAIN_FILE])
        train_gen = list(train_gen)
        assert len(train_gen) == 3
        inputs, labels = train_gen[0]
        assert_allclose(inputs[0], np.array([2, 0]))
        assert_allclose(inputs[1], np.array([3, 4]))
        assert_allclose(labels[0], np.array([1, 0]))

        inputs, labels = train_gen[1]
        assert_allclose(inputs[0], np.array([5, 0]))
        assert_allclose(inputs[1], np.array([6, 0]))
        assert_allclose(labels[0], np.array([0, 1]))

        inputs, labels = train_gen[2]
        assert_allclose(inputs[0], np.array([7, 0]))
        assert_allclose(inputs[1], np.array([8, 0]))
        assert_allclose(labels[0], np.array([1, 0]))

    def test_get_train_data_pad_with_max_lens(self):
        train_gen = self.data_manager.get_train_data_from_file(
            [self.TRAIN_FILE],
            max_lengths={"num_sentence_words": 1})
        train_gen = list(train_gen)
        inputs, labels = train_gen[0]
        assert_allclose(inputs[0], np.array([2]))
        assert_allclose(inputs[1], np.array([3]))
        assert_allclose(labels[0], np.array([1, 0]))

        inputs, labels = train_gen[1]
        assert_allclose(inputs[0], np.array([5]))
        assert_allclose(inputs[1], np.array([6]))
        assert_allclose(labels[0], np.array([0, 1]))

        inputs, labels = train_gen[2]
        assert_allclose(inputs[0], np.array([7]))
        assert_allclose(inputs[1], np.array([8]))
        assert_allclose(labels[0], np.array([1, 0]))

    def test_get_train_data_with_max_instances(self):
        train_gen = self.data_manager.get_train_data_from_file(
            [self.TRAIN_FILE],
            max_instances=2)
        train_gen = list(train_gen)
        assert len(train_gen) == 2
        inputs, labels = train_gen[0]
        assert_allclose(inputs[0], np.array([2, 0]))
        assert_allclose(inputs[1], np.array([3, 4]))
        assert_allclose(labels[0], np.array([1, 0]))

        inputs, labels = train_gen[1]
        assert_allclose(inputs[0], np.array([5, 0]))
        assert_allclose(inputs[1], np.array([6, 0]))
        assert_allclose(labels[0], np.array([0, 1]))

    def test_get_train_data_errors(self):
        with self.assertRaises(ValueError):
            self.data_manager.get_train_data_from_file(
                [self.TRAIN_FILE],
                max_lengths={"num_sentence_words": 1},
                pad=False)
        with self.assertRaises(ValueError):
            self.data_manager.get_train_data_from_file(
                [self.TRAIN_FILE],
                max_lengths={"some wrong key": 1})

    def test_get_train_data_no_pad(self):
        train_gen = self.data_manager.get_train_data_from_file(
            [self.TRAIN_FILE],
            pad=False)
        train_gen = list(train_gen)
        assert len(train_gen) == 3
        inputs, labels = train_gen[0]
        assert_allclose(inputs[0], np.array([2]))
        assert_allclose(inputs[1], np.array([3, 4]))
        assert_allclose(labels[0], np.array([1, 0]))

        inputs, labels = train_gen[1]
        assert_allclose(inputs[0], np.array([5]))
        assert_allclose(inputs[1], np.array([6]))
        assert_allclose(labels[0], np.array([0, 1]))

        inputs, labels = train_gen[2]
        assert_allclose(inputs[0], np.array([7]))
        assert_allclose(inputs[1], np.array([8]))
        assert_allclose(labels[0], np.array([1, 0]))

    def test_generate_train_batches(self):
        train_gen = self.data_manager.get_train_data_from_file(
            [self.TRAIN_FILE])
        batch_gen = self.data_manager.batch_generator(train_gen, 2)
        first_batch = batch_gen.__next__()
        inputs, labels = first_batch
        assert len(inputs) == 2
        assert len(labels) == 1
        assert_allclose(inputs[0], np.array([[2, 0], [5, 0]]))
        assert_allclose(inputs[1], np.array([[3, 4], [6, 0]]))
        assert_allclose(labels[0], np.array([[1, 0], [0, 1]]))

        second_batch = batch_gen.__next__()
        inputs, labels = second_batch
        assert len(inputs) == 2
        assert len(labels) == 1
        assert_allclose(inputs[0], np.array([[7, 0], [2, 0]]))
        assert_allclose(inputs[1], np.array([[8, 0], [3, 4]]))
        assert_allclose(labels[0], np.array([[1, 0], [1, 0]]))

        third_batch = batch_gen.__next__()
        inputs, labels = third_batch
        assert len(inputs) == 2
        assert len(labels) == 1
        assert_allclose(inputs[0], np.array([[5, 0], [7, 0]]))
        assert_allclose(inputs[1], np.array([[6, 0], [8, 0]]))
        assert_allclose(labels[0], np.array([[0, 1], [1, 0]]))


class TestDataManagerValidation(DuplicateTestCase):
    @overrides
    def setUp(self):
        super(TestDataManagerValidation, self).setUp()
        self.write_duplicate_questions_train_file()
        self.write_duplicate_questions_validation_file()
        self.data_manager = DataManager(STSInstance)
        self.data_manager.get_train_data_from_file([self.TRAIN_FILE])

    def test_get_validation_data_default(self):
        val_gen = self.data_manager.get_validation_data_from_file(
            [self.VALIDATION_FILE])
        val_gen = list(val_gen)
        assert len(val_gen) == 3
        inputs, labels = val_gen[0]
        assert_allclose(inputs[0], np.array([2, 0]))
        assert_allclose(inputs[1], np.array([3, 1]))
        assert_allclose(labels[0], np.array([1, 0]))

        inputs, labels = val_gen[1]
        assert_allclose(inputs[0], np.array([1, 0]))
        assert_allclose(inputs[1], np.array([1, 0]))
        assert_allclose(labels[0], np.array([0, 1]))

        inputs, labels = val_gen[2]
        assert_allclose(inputs[0], np.array([7, 0]))
        assert_allclose(inputs[1], np.array([8, 1]))
        assert_allclose(labels[0], np.array([1, 0]))

    def test_get_validation_data_pad_with_max_lens(self):
        val_gen = self.data_manager.get_validation_data_from_file(
            [self.VALIDATION_FILE],
            max_lengths={"num_sentence_words": 1})

        val_gen = list(val_gen)
        assert len(val_gen) == 3
        inputs, labels = val_gen[0]
        assert_allclose(inputs[0], np.array([2]))
        assert_allclose(inputs[1], np.array([3]))
        assert_allclose(labels[0], np.array([1, 0]))

        inputs, labels = val_gen[1]
        assert_allclose(inputs[0], np.array([1]))
        assert_allclose(inputs[1], np.array([1]))
        assert_allclose(labels[0], np.array([0, 1]))

        inputs, labels = val_gen[2]
        assert_allclose(inputs[0], np.array([7]))
        assert_allclose(inputs[1], np.array([8]))
        assert_allclose(labels[0], np.array([1, 0]))

    def test_get_validation_data_with_max_instances(self):
        val_gen = self.data_manager.get_validation_data_from_file(
            [self.VALIDATION_FILE],
            max_instances=2)
        val_gen = list(val_gen)
        assert len(val_gen) == 2
        inputs, labels = val_gen[0]
        assert_allclose(inputs[0], np.array([2, 0]))
        assert_allclose(inputs[1], np.array([3, 1]))
        assert_allclose(labels[0], np.array([1, 0]))

        inputs, labels = val_gen[1]
        assert_allclose(inputs[0], np.array([1, 0]))
        assert_allclose(inputs[1], np.array([1, 0]))
        assert_allclose(labels[0], np.array([0, 1]))

    def test_get_validation_data_errors(self):
        with self.assertRaises(ValueError):
            self.data_manager.get_validation_data_from_file(
                [self.VALIDATION_FILE],
                max_lengths={"num_sentence_words": 1},
                pad=False)
        with self.assertRaises(ValueError):
            self.data_manager.get_validation_data_from_file(
                [self.VALIDATION_FILE],
                max_lengths={"some wrong key": 1})

    def test_get_validation_data_no_pad(self):
        val_gen = self.data_manager.get_validation_data_from_file(
            [self.VALIDATION_FILE],
            pad=False)
        val_gen = list(val_gen)
        assert len(val_gen) == 3
        inputs, labels = val_gen[0]
        assert_allclose(inputs[0], np.array([2]))
        assert_allclose(inputs[1], np.array([3, 1]))
        assert_allclose(labels[0], np.array([1, 0]))

        inputs, labels = val_gen[1]
        assert_allclose(inputs[0], np.array([1]))
        assert_allclose(inputs[1], np.array([1]))
        assert_allclose(labels[0], np.array([0, 1]))

        inputs, labels = val_gen[2]
        assert_allclose(inputs[0], np.array([7]))
        assert_allclose(inputs[1], np.array([8, 1, 1]))
        assert_allclose(labels[0], np.array([1, 0]))

    def test_generate_validation_batches(self):
        val_gen = self.data_manager.get_validation_data_from_file(
            [self.VALIDATION_FILE])
        batch_gen = self.data_manager.batch_generator(val_gen, 2)
        first_batch = batch_gen.__next__()
        inputs, labels = first_batch
        assert len(inputs) == 2
        assert len(labels) == 1
        assert_allclose(inputs[0], np.array([[2, 0], [1, 0]]))
        assert_allclose(inputs[1], np.array([[3, 1], [1, 0]]))
        assert_allclose(labels[0], np.array([[1, 0], [0, 1]]))

        second_batch = batch_gen.__next__()
        inputs, labels = second_batch
        assert len(inputs) == 2
        assert len(labels) == 1
        assert_allclose(inputs[0], np.array([[7, 0], [2, 0]]))
        assert_allclose(inputs[1], np.array([[8, 1], [3, 1]]))
        assert_allclose(labels[0], np.array([[1, 0], [1, 0]]))

        third_batch = batch_gen.__next__()
        inputs, labels = third_batch
        assert len(inputs) == 2
        assert len(labels) == 1
        assert_allclose(inputs[0], np.array([[1, 0], [7, 0]]))
        assert_allclose(inputs[1], np.array([[1, 0], [8, 1]]))
        assert_allclose(labels[0], np.array([[0, 1], [1, 0]]))


class TestDataManagerTest(DuplicateTestCase):
    @overrides
    def setUp(self):
        super(TestDataManagerTest, self).setUp()
        self.write_duplicate_questions_train_file()
        self.write_duplicate_questions_test_file()
        self.data_manager = DataManager(STSInstance)
        self.data_manager.get_train_data_from_file([self.TRAIN_FILE])

    def test_get_test_data_default(self):
        test_gen = self.data_manager.get_test_data_from_file(
            [self.TEST_FILE])
        test_gen = list(test_gen)
        assert len(test_gen) == 3

        inputs, labels = test_gen[0]
        assert_allclose(inputs[0], np.array([2, 1]))
        assert_allclose(inputs[1], np.array([1, 0]))
        assert len(labels) == 0

        inputs, labels = test_gen[1]
        assert_allclose(inputs[0], np.array([4, 0]))
        assert_allclose(inputs[1], np.array([5, 1]))
        assert len(labels) == 0

        inputs, labels = test_gen[2]
        assert_allclose(inputs[0], np.array([6, 0]))
        assert_allclose(inputs[1], np.array([7, 0]))
        assert len(labels) == 0

    def test_get_test_data_pad_with_max_lens(self):
        test_gen = self.data_manager.get_test_data_from_file(
            [self.TEST_FILE],
            max_lengths={"num_sentence_words": 1})
        test_gen = list(test_gen)
        assert len(test_gen) == 3

        inputs, labels = test_gen[0]
        assert_allclose(inputs[0], np.array([2]))
        assert_allclose(inputs[1], np.array([1]))
        assert len(labels) == 0

        inputs, labels = test_gen[1]
        assert_allclose(inputs[0], np.array([4]))
        assert_allclose(inputs[1], np.array([5]))
        assert len(labels) == 0

        inputs, labels = test_gen[2]
        assert_allclose(inputs[0], np.array([6]))
        assert_allclose(inputs[1], np.array([7]))
        assert len(labels) == 0

    def test_get_test_data_with_max_instances(self):
        test_gen = self.data_manager.get_test_data_from_file(
            [self.TEST_FILE],
            max_instances=2)
        test_gen = list(test_gen)
        assert len(test_gen) == 2

        inputs, labels = test_gen[0]
        assert_allclose(inputs[0], np.array([2, 1]))
        assert_allclose(inputs[1], np.array([1, 0]))
        assert len(labels) == 0

        inputs, labels = test_gen[1]
        assert_allclose(inputs[0], np.array([4, 0]))
        assert_allclose(inputs[1], np.array([5, 1]))
        assert len(labels) == 0

    def test_get_test_data_errors(self):
        with self.assertRaises(ValueError):
            self.data_manager.get_test_data_from_file(
                [self.TEST_FILE],
                max_lengths={"num_sentence_words": 1},
                pad=False)
        with self.assertRaises(ValueError):
            self.data_manager.get_test_data_from_file(
                [self.TEST_FILE],
                max_lengths={"some wrong key": 1})

    def test_get_test_data_no_pad(self):
        test_gen = self.data_manager.get_test_data_from_file(
            [self.TEST_FILE],
            pad=False)
        test_gen = list(test_gen)
        assert len(test_gen) == 3

        inputs, labels = test_gen[0]
        assert_allclose(inputs[0], np.array([2, 1, 2]))
        assert_allclose(inputs[1], np.array([1]))
        assert len(labels) == 0

        inputs, labels = test_gen[1]
        assert_allclose(inputs[0], np.array([4]))
        assert_allclose(inputs[1], np.array([5, 1]))
        assert len(labels) == 0

        inputs, labels = test_gen[2]
        assert_allclose(inputs[0], np.array([6]))
        assert_allclose(inputs[1], np.array([7]))
        assert len(labels) == 0

    def test_generate_test_batches(self):
        test_gen = self.data_manager.get_test_data_from_file(
            [self.TEST_FILE])
        batch_gen = self.data_manager.batch_generator(test_gen, 2)
        first_batch = batch_gen.__next__()
        inputs, labels = first_batch
        assert len(inputs) == 2
        assert len(labels) == 0
        assert_allclose(inputs[0], np.array([[2, 1], [4, 0]]))
        assert_allclose(inputs[1], np.array([[1, 0], [5, 1]]))

        second_batch = batch_gen.__next__()
        inputs, labels = second_batch
        assert len(inputs) == 2
        assert len(labels) == 0
        assert_allclose(inputs[0], np.array([[6, 0], [2, 1]]))
        assert_allclose(inputs[1], np.array([[7, 0], [1, 0]]))

        third_batch = batch_gen.__next__()
        inputs, labels = third_batch
        assert len(inputs) == 2
        assert len(labels) == 0
        assert_allclose(inputs[0], np.array([[4, 0], [6, 0]]))
        assert_allclose(inputs[1], np.array([[5, 1], [7, 0]]))
