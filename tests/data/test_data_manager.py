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
        get_train_gen, train_size = self.data_manager.get_train_data_from_file(
            [self.TRAIN_FILE])
        assert train_size == 3
        train_gen = get_train_gen()
        inputs1, labels1 = train_gen.__next__()
        assert_allclose(inputs1[0], np.array([2, 0]))
        assert_allclose(inputs1[1], np.array([3, 4]))
        assert_allclose(labels1[0], np.array([1, 0]))

        inputs2, labels2 = train_gen.__next__()
        assert_allclose(inputs2[0], np.array([5, 0]))
        assert_allclose(inputs2[1], np.array([6, 0]))
        assert_allclose(labels2[0], np.array([0, 1]))

        inputs3, labels3 = train_gen.__next__()
        assert_allclose(inputs3[0], np.array([7, 0]))
        assert_allclose(inputs3[1], np.array([8, 0]))
        assert_allclose(labels3[0], np.array([1, 0]))

        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            train_gen.__next__()

        # Test that we can make a new train generator
        new_train_gen = get_train_gen()
        # Verify that the new and old generator are not the same object
        assert new_train_gen != train_gen
        new_inputs1, new_labels1 = new_train_gen.__next__()
        assert_allclose(new_inputs1, inputs1)
        assert_allclose(new_labels1, labels1)
        new_inputs2, new_labels2 = new_train_gen.__next__()
        assert_allclose(new_inputs2, inputs2)
        assert_allclose(new_labels2, labels2)
        new_inputs3, new_labels3 = new_train_gen.__next__()
        assert_allclose(new_inputs3, inputs3)
        assert_allclose(new_labels3, labels3)
        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            new_train_gen.__next__()

    def test_get_train_data_default_character(self):
        get_train_gen, train_size = self.data_manager.get_train_data_from_file(
            [self.TRAIN_FILE], mode="character")
        train_gen = get_train_gen()
        assert train_size == 3
        inputs1, labels1 = train_gen.__next__()
        assert_allclose(inputs1[0], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 10, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert_allclose(inputs1[1], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 11, 0, 0, 0],
                                              [6, 9, 2, 7, 8, 3, 5, 4, 12, 19, 17, 18]]))
        assert_allclose(labels1[0], np.array([1, 0]))

        inputs2, labels2 = train_gen.__next__()
        assert_allclose(inputs2[0], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 13, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert_allclose(inputs2[1], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 14, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert_allclose(labels2[0], np.array([0, 1]))

        inputs3, labels3 = train_gen.__next__()
        assert_allclose(inputs3[0], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 15, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert_allclose(inputs3[1], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 16, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert_allclose(labels3[0], np.array([1, 0]))

        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            train_gen.__next__()

    def test_get_train_data_default_word_and_character(self):
        get_train_gen, train_size = self.data_manager.get_train_data_from_file(
            [self.TRAIN_FILE], mode="word+character")
        train_gen = get_train_gen()
        assert train_size == 3
        inputs1, labels1 = train_gen.__next__()
        assert_allclose(inputs1[0], np.array([2, 0]))
        assert_allclose(inputs1[1], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 10, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert_allclose(inputs1[2], np.array([3, 4]))
        assert_allclose(inputs1[3], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 11, 0, 0, 0],
                                              [6, 9, 2, 7, 8, 3, 5, 4, 12, 19, 17, 18]]))
        assert_allclose(labels1[0], np.array([1, 0]))

        inputs2, labels2 = train_gen.__next__()
        assert_allclose(inputs2[0], np.array([5, 0]))
        assert_allclose(inputs2[1], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 13, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert_allclose(inputs2[2], np.array([6, 0]))
        assert_allclose(inputs2[3], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 14, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert_allclose(labels2[0], np.array([0, 1]))

        inputs3, labels3 = train_gen.__next__()
        assert_allclose(inputs3[0], np.array([7, 0]))
        assert_allclose(inputs3[1], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 15, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert_allclose(inputs3[2], np.array([8, 0]))
        assert_allclose(inputs3[3], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 16, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert_allclose(labels3[0], np.array([1, 0]))

        # Should cycle back to the start
        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            train_gen.__next__()

    def test_get_train_data_pad_with_max_lens(self):
        get_train_gen, train_size = self.data_manager.get_train_data_from_file(
            [self.TRAIN_FILE],
            max_lengths={"num_sentence_words": 1})
        train_gen = get_train_gen()
        assert train_size == 3
        inputs1, labels1 = train_gen.__next__()
        assert_allclose(inputs1[0], np.array([2]))
        assert_allclose(inputs1[1], np.array([3]))
        assert_allclose(labels1[0], np.array([1, 0]))

        inputs2, labels2 = train_gen.__next__()
        assert_allclose(inputs2[0], np.array([5]))
        assert_allclose(inputs2[1], np.array([6]))
        assert_allclose(labels2[0], np.array([0, 1]))

        inputs3, labels3 = train_gen.__next__()
        assert_allclose(inputs3[0], np.array([7]))
        assert_allclose(inputs3[1], np.array([8]))
        assert_allclose(labels3[0], np.array([1, 0]))

        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            train_gen.__next__()

    def test_get_train_data_with_max_instances(self):
        get_train_gen, train_size = self.data_manager.get_train_data_from_file(
            [self.TRAIN_FILE],
            max_instances=2)
        train_gen = get_train_gen()
        assert train_size == 2
        inputs1, labels1 = train_gen.__next__()
        assert_allclose(inputs1[0], np.array([2, 0]))
        assert_allclose(inputs1[1], np.array([3, 4]))
        assert_allclose(labels1[0], np.array([1, 0]))

        inputs2, labels2 = train_gen.__next__()
        assert_allclose(inputs2[0], np.array([5, 0]))
        assert_allclose(inputs2[1], np.array([6, 0]))
        assert_allclose(labels2[0], np.array([0, 1]))

        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            train_gen.__next__()

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
        get_train_gen, train_size = self.data_manager.get_train_data_from_file(
            [self.TRAIN_FILE],
            pad=False)
        train_gen = get_train_gen()
        assert train_size == 3
        inputs1, labels1 = train_gen.__next__()
        assert_allclose(inputs1[0], np.array([2]))
        assert_allclose(inputs1[1], np.array([3, 4]))
        assert_allclose(labels1[0], np.array([1, 0]))

        inputs2, labels2 = train_gen.__next__()
        assert_allclose(inputs2[0], np.array([5]))
        assert_allclose(inputs2[1], np.array([6]))
        assert_allclose(labels2[0], np.array([0, 1]))

        inputs3, labels3 = train_gen.__next__()
        assert_allclose(inputs3[0], np.array([7]))
        assert_allclose(inputs3[1], np.array([8]))
        assert_allclose(labels3[0], np.array([1, 0]))

        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            train_gen.__next__()

    def test_generate_train_batches(self):
        get_train_gen, train_size = self.data_manager.get_train_data_from_file(
            [self.TRAIN_FILE])
        batch_gen = DataManager.get_batch_generator(get_train_gen, 2)
        new_batch_gen = DataManager.get_batch_generator(get_train_gen, 2)

        # Assert that the new generator is a different object
        # than the old generator.
        assert new_batch_gen != batch_gen
        assert train_size == 3

        first_batch = batch_gen.__next__()
        new_first_batch = new_batch_gen.__next__()
        inputs, labels = first_batch
        new_inputs, new_labels = new_first_batch
        assert len(inputs) == len(new_inputs) == 2
        assert len(labels) == len(new_labels) == 1

        # Ensure output matches ground truth
        assert_allclose(inputs[0], np.array([[2, 0], [5, 0]]))
        assert_allclose(inputs[1], np.array([[3, 4], [6, 0]]))
        assert_allclose(labels[0], np.array([[1, 0], [0, 1]]))
        # Ensure both generators produce same results.
        assert_allclose(inputs[0], new_inputs[0])
        assert_allclose(inputs[1], new_inputs[1])
        assert_allclose(labels[0], labels[0])

        second_batch = batch_gen.__next__()
        new_second_batch = new_batch_gen.__next__()
        inputs, labels = second_batch
        new_inputs, new_labels = new_second_batch
        assert len(inputs) == len(new_inputs) == 2
        assert len(labels) == len(new_labels) == 1

        # Ensure output matches ground truth
        assert_allclose(inputs[0], np.array([[7, 0]]))
        assert_allclose(inputs[1], np.array([[8, 0]]))
        assert_allclose(labels[0], np.array([[1, 0]]))
        # Ensure both generators produce same results.
        assert_allclose(inputs[0], new_inputs[0])
        assert_allclose(inputs[1], new_inputs[1])
        assert_allclose(labels[0], labels[0])

        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            batch_gen.__next__()
            new_batch_gen.__next__()


class TestDataManagerValidation(DuplicateTestCase):
    @overrides
    def setUp(self):
        super(TestDataManagerValidation, self).setUp()
        self.write_duplicate_questions_train_file()
        self.write_duplicate_questions_validation_file()
        self.data_manager = DataManager(STSInstance)
        self.data_manager.get_train_data_from_file([self.TRAIN_FILE])

    def test_get_validation_data_default(self):
        get_val_gen, val_size = self.data_manager.get_validation_data_from_file(
            [self.VALIDATION_FILE])
        assert val_size == 3
        val_gen = get_val_gen()
        inputs1, labels1 = val_gen.__next__()
        assert_allclose(inputs1[0], np.array([2, 0]))
        assert_allclose(inputs1[1], np.array([3, 1]))
        assert_allclose(labels1[0], np.array([1, 0]))

        inputs2, labels2 = val_gen.__next__()
        assert_allclose(inputs2[0], np.array([1, 0]))
        assert_allclose(inputs2[1], np.array([1, 0]))
        assert_allclose(labels2[0], np.array([0, 1]))

        inputs3, labels3 = val_gen.__next__()
        assert_allclose(inputs3[0], np.array([7, 0]))
        assert_allclose(inputs3[1], np.array([8, 1]))
        assert_allclose(labels3[0], np.array([1, 0]))

        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            val_gen.__next__()

        # Test that we can make a new val generator
        new_val_gen = get_val_gen()
        # Verify that the new and old generator are not the same object
        assert new_val_gen != val_gen
        new_inputs1, new_labels1 = new_val_gen.__next__()
        assert_allclose(new_inputs1, inputs1)
        assert_allclose(new_labels1, labels1)
        new_inputs2, new_labels2 = new_val_gen.__next__()
        assert_allclose(new_inputs2, inputs2)
        assert_allclose(new_labels2, labels2)
        new_inputs3, new_labels3 = new_val_gen.__next__()
        assert_allclose(new_inputs3, inputs3)
        assert_allclose(new_labels3, labels3)

        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            new_val_gen.__next__()

    def test_get_validation_data_default_character(self):
        get_val_gen, val_size = self.data_manager.get_validation_data_from_file(
            [self.VALIDATION_FILE], mode="character")
        assert val_size == 3
        val_gen = get_val_gen()
        inputs1, labels1 = val_gen.__next__()
        assert_allclose(inputs1[0], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 10, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert_allclose(inputs1[1], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 11, 0, 0, 0],
                                              [6, 9, 2, 7, 8, 3, 5, 4, 1, 0, 0, 0]]))
        assert_allclose(labels1[0], np.array([1, 0]))

        inputs2, labels2 = val_gen.__next__()
        assert_allclose(inputs2[0], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 1, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert_allclose(inputs2[1], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 10, 1, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert_allclose(labels2[0], np.array([0, 1]))

        inputs3, labels3 = val_gen.__next__()
        assert_allclose(inputs3[0], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 15, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert_allclose(inputs3[1], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 16, 0, 0, 0],
                                              [6, 9, 2, 7, 8, 3, 5, 4, 10, 10, 0, 0]]))
        assert_allclose(labels3[0], np.array([1, 0]))

        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            val_gen.__next__()

    def test_get_validation_data_default_word_and_character(self):
        get_val_gen, val_size = self.data_manager.get_validation_data_from_file(
            [self.VALIDATION_FILE], mode="word+character")
        val_gen = get_val_gen()
        assert val_size == 3
        inputs1, labels1 = val_gen.__next__()
        assert_allclose(inputs1[0], np.array([2, 0]))
        assert_allclose(inputs1[1], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 10, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert_allclose(inputs1[2], np.array([3, 1]))
        assert_allclose(inputs1[3], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 11, 0, 0, 0],
                                              [6, 9, 2, 7, 8, 3, 5, 4, 1, 0, 0, 0]]))
        assert_allclose(labels1[0], np.array([1, 0]))

        inputs2, labels2 = val_gen.__next__()
        assert_allclose(inputs2[0], np.array([1, 0]))
        assert_allclose(inputs2[1], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 1, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert_allclose(inputs2[2], np.array([1, 0]))
        assert_allclose(inputs2[3], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 10, 1, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert_allclose(labels2[0], np.array([0, 1]))

        inputs3, labels3 = val_gen.__next__()
        assert_allclose(inputs3[0], np.array([7, 0]))
        assert_allclose(inputs3[1], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 15, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert_allclose(inputs3[2], np.array([8, 1]))
        assert_allclose(inputs3[3], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 16, 0, 0, 0],
                                              [6, 9, 2, 7, 8, 3, 5, 4, 10, 10, 0, 0]]))
        assert_allclose(labels3[0], np.array([1, 0]))

        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            val_gen.__next__()

    def test_get_validation_data_pad_with_max_lens(self):
        get_val_gen, val_size = self.data_manager.get_validation_data_from_file(
            [self.VALIDATION_FILE],
            max_lengths={"num_sentence_words": 1})
        val_gen = get_val_gen()
        assert val_size == 3
        inputs1, labels1 = val_gen.__next__()
        assert_allclose(inputs1[0], np.array([2]))
        assert_allclose(inputs1[1], np.array([3]))
        assert_allclose(labels1[0], np.array([1, 0]))

        inputs2, labels2 = val_gen.__next__()
        assert_allclose(inputs2[0], np.array([1]))
        assert_allclose(inputs2[1], np.array([1]))
        assert_allclose(labels2[0], np.array([0, 1]))

        inputs3, labels3 = val_gen.__next__()
        assert_allclose(inputs3[0], np.array([7]))
        assert_allclose(inputs3[1], np.array([8]))
        assert_allclose(labels3[0], np.array([1, 0]))

        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            val_gen.__next__()

    def test_get_validation_data_with_max_instances(self):
        get_val_gen, val_size = self.data_manager.get_validation_data_from_file(
            [self.VALIDATION_FILE],
            max_instances=2)
        val_size == 2
        val_gen = get_val_gen()
        inputs1, labels1 = val_gen.__next__()
        assert_allclose(inputs1[0], np.array([2, 0]))
        assert_allclose(inputs1[1], np.array([3, 1]))
        assert_allclose(labels1[0], np.array([1, 0]))

        inputs2, labels2 = val_gen.__next__()
        assert_allclose(inputs2[0], np.array([1, 0]))
        assert_allclose(inputs2[1], np.array([1, 0]))
        assert_allclose(labels2[0], np.array([0, 1]))

        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            val_gen.__next__()

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
        get_val_gen, val_size = self.data_manager.get_validation_data_from_file(
            [self.VALIDATION_FILE],
            pad=False)
        assert val_size == 3
        val_gen = get_val_gen()
        inputs1, labels1 = val_gen.__next__()
        assert_allclose(inputs1[0], np.array([2]))
        assert_allclose(inputs1[1], np.array([3, 1]))
        assert_allclose(labels1[0], np.array([1, 0]))

        inputs2, labels2 = val_gen.__next__()
        assert_allclose(inputs2[0], np.array([1]))
        assert_allclose(inputs2[1], np.array([1]))
        assert_allclose(labels2[0], np.array([0, 1]))

        inputs3, labels3 = val_gen.__next__()
        assert_allclose(inputs3[0], np.array([7]))
        assert_allclose(inputs3[1], np.array([8, 1, 1]))
        assert_allclose(labels3[0], np.array([1, 0]))

        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            val_gen.__next__()

    def test_generate_validation_batches(self):
        get_val_gen, val_size = self.data_manager.get_validation_data_from_file(
            [self.VALIDATION_FILE])
        batch_gen = self.data_manager.get_batch_generator(get_val_gen, 2)
        new_batch_gen = DataManager.get_batch_generator(get_val_gen, 2)
        assert val_size == 3

        # Assert that the new generator is a different object
        # than the old generator.
        assert new_batch_gen != batch_gen

        first_batch = batch_gen.__next__()
        new_first_batch = new_batch_gen.__next__()
        inputs, labels = first_batch
        new_inputs, new_labels = new_first_batch
        assert len(inputs) == len(new_inputs) == 2
        assert len(labels) == len(new_labels) == 1

        # Ensure output matches ground truth.
        assert_allclose(inputs[0], np.array([[2, 0], [1, 0]]))
        assert_allclose(inputs[1], np.array([[3, 1], [1, 0]]))
        assert_allclose(labels[0], np.array([[1, 0], [0, 1]]))
        # Ensure both generators produce same results.
        assert_allclose(inputs[0], new_inputs[0])
        assert_allclose(inputs[1], new_inputs[1])
        assert_allclose(labels[0], labels[0])

        second_batch = batch_gen.__next__()
        new_second_batch = new_batch_gen.__next__()
        inputs, labels = second_batch
        new_inputs, new_labels = new_second_batch
        assert len(inputs) == len(new_inputs) == 2
        assert len(labels) == len(new_labels) == 1

        # Ensure output matches ground truth.
        assert_allclose(inputs[0], np.array([[7, 0]]))
        assert_allclose(inputs[1], np.array([[8, 1]]))
        assert_allclose(labels[0], np.array([[1, 0]]))
        # Ensure both generators produce same results.
        assert_allclose(inputs[0], new_inputs[0])
        assert_allclose(inputs[1], new_inputs[1])
        assert_allclose(labels[0], labels[0])

        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            batch_gen.__next__()
            new_batch_gen.__next__()


class TestDataManagerTest(DuplicateTestCase):
    @overrides
    def setUp(self):
        super(TestDataManagerTest, self).setUp()
        self.write_duplicate_questions_train_file()
        self.write_duplicate_questions_test_file()
        self.data_manager = DataManager(STSInstance)
        self.data_manager.get_train_data_from_file([self.TRAIN_FILE])

    def test_get_test_data_default(self):
        get_test_gen, test_size = self.data_manager.get_test_data_from_file(
            [self.TEST_FILE])
        assert test_size == 3
        test_gen = get_test_gen()
        inputs1, labels1 = test_gen.__next__()
        assert_allclose(inputs1[0], np.array([2, 1]))
        assert_allclose(inputs1[1], np.array([1, 0]))

        inputs2, labels2 = test_gen.__next__()
        assert_allclose(inputs2[0], np.array([4, 0]))
        assert_allclose(inputs2[1], np.array([5, 1]))

        inputs3, labels3 = test_gen.__next__()
        assert_allclose(inputs3[0], np.array([6, 0]))
        assert_allclose(inputs3[1], np.array([7, 0]))

        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            test_gen.__next__()

        # Test that we can make a new test generator
        new_test_gen = get_test_gen()
        # Verify that the new and old generator are not the same object
        assert new_test_gen != test_gen
        new_inputs1, new_labels1 = new_test_gen.__next__()
        assert_allclose(new_inputs1, inputs1)
        assert_allclose(new_labels1, labels1)
        new_inputs2, new_labels2 = new_test_gen.__next__()
        assert_allclose(new_inputs2, inputs2)
        assert_allclose(new_labels2, labels2)
        new_inputs3, new_labels3 = new_test_gen.__next__()
        assert_allclose(new_inputs3, inputs3)
        assert_allclose(new_labels3, labels3)
        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            new_test_gen.__next__()

    def test_get_test_data_default_character(self):
        get_test_gen, test_size = self.data_manager.get_test_data_from_file(
            [self.TEST_FILE], mode="character")
        test_gen = get_test_gen()
        assert test_size == 3
        inputs1, labels = test_gen.__next__()
        assert_allclose(inputs1[0], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 10, 0, 0, 0],
                                              [6, 9, 2, 7, 8, 3, 5, 4, 9, 4, 1, 10]]))
        assert_allclose(inputs1[1], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 9, 4, 1, 11],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert len(labels) == 0

        inputs2, labels = test_gen.__next__()
        assert_allclose(inputs2[0], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 12, 19, 17, 18],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert_allclose(inputs2[1], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 13, 0, 0, 0],
                                              [6, 9, 2, 7, 8, 3, 5, 4, 9, 4, 1, 12]]))
        assert len(labels) == 0

        inputs3, labels = test_gen.__next__()
        assert_allclose(inputs3[0], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 14, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert_allclose(inputs3[1], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 15, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert len(labels) == 0
        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            test_gen.__next__()

    def test_get_test_data_default_word_and_character(self):
        get_test_gen, test_size = self.data_manager.get_test_data_from_file(
            [self.TEST_FILE], mode="word+character")
        test_gen = get_test_gen()
        assert test_size == 3
        inputs1, labels = test_gen.__next__()
        assert_allclose(inputs1[0], np.array([2, 1]))
        assert_allclose(inputs1[1], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 10, 0, 0, 0],
                                              [6, 9, 2, 7, 8, 3, 5, 4, 9, 4, 1, 10]]))
        assert_allclose(inputs1[2], np.array([1, 0]))
        assert_allclose(inputs1[3], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 9, 4, 1, 11],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert len(labels) == 0

        inputs2, labels = test_gen.__next__()
        assert_allclose(inputs2[0], np.array([4, 0]))
        assert_allclose(inputs2[1], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 12, 19, 17, 18],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert_allclose(inputs2[2], np.array([5, 1]))
        assert_allclose(inputs2[3], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 13, 0, 0, 0],
                                              [6, 9, 2, 7, 8, 3, 5, 4, 9, 4, 1, 12]]))
        assert len(labels) == 0

        inputs3, labels = test_gen.__next__()
        assert_allclose(inputs3[0], np.array([6, 0]))
        assert_allclose(inputs3[1], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 14, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert_allclose(inputs3[2], np.array([7, 0]))
        assert_allclose(inputs3[3], np.array([[6, 9, 2, 7, 8, 3, 5, 4, 15, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
        assert len(labels) == 0

        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            test_gen.__next__()

    def test_get_test_data_pad_with_max_lens(self):
        get_test_gen, test_size = self.data_manager.get_test_data_from_file(
            [self.TEST_FILE],
            max_lengths={"num_sentence_words": 1})
        test_gen = get_test_gen()
        assert test_size == 3

        inputs, labels = test_gen.__next__()
        assert_allclose(inputs[0], np.array([2]))
        assert_allclose(inputs[1], np.array([1]))
        assert len(labels) == 0

        inputs, labels = test_gen.__next__()
        assert_allclose(inputs[0], np.array([4]))
        assert_allclose(inputs[1], np.array([5]))
        assert len(labels) == 0

        inputs, labels = test_gen.__next__()
        assert_allclose(inputs[0], np.array([6]))
        assert_allclose(inputs[1], np.array([7]))
        assert len(labels) == 0

        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            test_gen.__next__()

    def test_get_test_data_with_max_instances(self):
        get_test_gen, test_size = self.data_manager.get_test_data_from_file(
            [self.TEST_FILE],
            max_instances=2)
        test_gen = get_test_gen()
        assert test_size == 2

        inputs, labels = test_gen.__next__()
        assert_allclose(inputs[0], np.array([2, 1]))
        assert_allclose(inputs[1], np.array([1, 0]))
        assert len(labels) == 0

        inputs, labels = test_gen.__next__()
        assert_allclose(inputs[0], np.array([4, 0]))
        assert_allclose(inputs[1], np.array([5, 1]))
        assert len(labels) == 0

        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            test_gen.__next__()

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
        get_test_gen, test_size = self.data_manager.get_test_data_from_file(
            [self.TEST_FILE],
            pad=False)
        test_gen = get_test_gen()
        assert test_size == 3

        inputs, labels = test_gen.__next__()
        assert_allclose(inputs[0], np.array([2, 1, 2]))
        assert_allclose(inputs[1], np.array([1]))
        assert len(labels) == 0

        inputs, labels = test_gen.__next__()
        assert_allclose(inputs[0], np.array([4]))
        assert_allclose(inputs[1], np.array([5, 1]))
        assert len(labels) == 0

        inputs, labels = test_gen.__next__()
        assert_allclose(inputs[0], np.array([6]))
        assert_allclose(inputs[1], np.array([7]))
        assert len(labels) == 0

        # Should raise a StopIteration
        with self.assertRaises(StopIteration):
            test_gen.__next__()

    def test_generate_test_batches(self):
        get_test_gen, test_size = self.data_manager.get_test_data_from_file(
            [self.TEST_FILE])
        batch_gen = self.data_manager.get_batch_generator(get_test_gen, 2)
        new_batch_gen = DataManager.get_batch_generator(get_test_gen, 2)

        # Assert that the new generator is a different object
        # than the old generator.
        assert new_batch_gen != batch_gen
        assert test_size == 3

        first_batch = batch_gen.__next__()
        new_first_batch = new_batch_gen.__next__()
        inputs, labels = first_batch
        new_inputs, new_labels = new_first_batch
        assert len(inputs) == 2
        assert len(labels) == 0

        # Ensure output matches ground truth
        assert_allclose(inputs[0], np.array([[2, 1], [4, 0]]))
        assert_allclose(inputs[1], np.array([[1, 0], [5, 1]]))
        # Ensure both generators produce same results.
        assert_allclose(inputs[0], new_inputs[0])
        assert_allclose(inputs[1], new_inputs[1])

        second_batch = batch_gen.__next__()
        new_second_batch = new_batch_gen.__next__()
        inputs, labels = second_batch
        new_inputs, new_labels = new_second_batch
        assert len(inputs) == 2
        assert len(labels) == 0

        # Ensure output matches ground truth
        assert_allclose(inputs[0], np.array([[6, 0]]))
        assert_allclose(inputs[1], np.array([[7, 0]]))
        # Ensure both generators produce same results.
        assert_allclose(inputs[0], new_inputs[0])
        assert_allclose(inputs[1], new_inputs[1])

        with self.assertRaises(StopIteration):
            batch_gen.__next__()
            new_batch_gen.__next__()
