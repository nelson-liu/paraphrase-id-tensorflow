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
        train_gen, train_size = self.data_manager.get_train_data_from_file(
            [self.TRAIN_FILE])
        assert train_size == 3
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

        # Should cycle back to the start
        inputs4, labels4 = train_gen.__next__()
        assert_allclose(inputs4, inputs1)
        assert_allclose(labels4, labels1)

    def test_get_train_data_default_character(self):
        train_gen, train_size = self.data_manager.get_train_data_from_file(
            [self.TRAIN_FILE], mode="character")
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

        # Should cycle back to the start
        inputs4, labels4 = train_gen.__next__()
        assert_allclose(inputs4, inputs1)
        assert_allclose(labels4, labels1)

    def test_get_train_data_default_word_and_character(self):
        train_gen, train_size = self.data_manager.get_train_data_from_file(
            [self.TRAIN_FILE], mode="word+character")
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
        inputs4, labels4 = train_gen.__next__()
        for idx, val in enumerate(inputs4):
            assert_allclose(inputs4[idx], inputs1[idx])
        assert_allclose(labels4, labels1)

    def test_get_train_data_pad_with_max_lens(self):
        train_gen, train_size = self.data_manager.get_train_data_from_file(
            [self.TRAIN_FILE],
            max_lengths={"num_sentence_words": 1})
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

        # Should cycle back to the start
        inputs4, labels4 = train_gen.__next__()
        assert_allclose(inputs4, inputs1)
        assert_allclose(labels4, labels1)

    def test_get_train_data_with_max_instances(self):
        train_gen, train_size = self.data_manager.get_train_data_from_file(
            [self.TRAIN_FILE],
            max_instances=2)
        assert train_size == 2
        inputs1, labels1 = train_gen.__next__()
        assert_allclose(inputs1[0], np.array([2, 0]))
        assert_allclose(inputs1[1], np.array([3, 4]))
        assert_allclose(labels1[0], np.array([1, 0]))

        inputs2, labels2 = train_gen.__next__()
        assert_allclose(inputs2[0], np.array([5, 0]))
        assert_allclose(inputs2[1], np.array([6, 0]))
        assert_allclose(labels2[0], np.array([0, 1]))

        # Should cycle back to the start
        inputs3, labels3 = train_gen.__next__()
        assert_allclose(inputs3, inputs1)
        assert_allclose(labels3, labels1)

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
        train_gen, train_size = self.data_manager.get_train_data_from_file(
            [self.TRAIN_FILE],
            pad=False)
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

        # Should cycle back to the start
        # Can't just use one assert_allclose since the vectors
        # are of different shape.
        inputs4, labels4 = train_gen.__next__()
        assert_allclose(inputs4[0], inputs1[0])
        assert_allclose(inputs4[1], inputs1[1])
        assert_allclose(labels4[0], labels1[0])

    def test_generate_train_batches(self):
        train_gen, train_size = self.data_manager.get_train_data_from_file(
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
        val_gen, val_size = self.data_manager.get_validation_data_from_file(
            [self.VALIDATION_FILE])
        assert val_size == 3
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

        # Should cycle back to the start
        inputs4, labels4 = val_gen.__next__()
        assert_allclose(inputs4, inputs1)
        assert_allclose(labels4, labels1)

    def test_get_validation_data_default_character(self):
        val_gen, val_size = self.data_manager.get_validation_data_from_file(
            [self.VALIDATION_FILE], mode="character")
        assert val_size == 3
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

        # Should cycle back to the start
        inputs4, labels4 = val_gen.__next__()
        assert_allclose(inputs4, inputs1)
        assert_allclose(labels4, labels1)

    def test_get_validation_data_default_word_and_character(self):
        val_gen, val_size = self.data_manager.get_validation_data_from_file(
            [self.VALIDATION_FILE], mode="word+character")
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

        # Should cycle back to the start
        inputs4, labels4 = val_gen.__next__()
        for idx, val in enumerate(inputs4):
            assert_allclose(inputs4[idx], inputs1[idx])
        assert_allclose(labels4, labels1)

    def test_get_validation_data_pad_with_max_lens(self):
        val_gen, val_size = self.data_manager.get_validation_data_from_file(
            [self.VALIDATION_FILE],
            max_lengths={"num_sentence_words": 1})

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

        # Should cycle back to the start
        inputs4, labels4 = val_gen.__next__()
        assert_allclose(inputs4, inputs1)
        assert_allclose(labels4, labels1)

    def test_get_validation_data_with_max_instances(self):
        val_gen, val_size = self.data_manager.get_validation_data_from_file(
            [self.VALIDATION_FILE],
            max_instances=2)
        val_size == 2
        inputs1, labels1 = val_gen.__next__()
        assert_allclose(inputs1[0], np.array([2, 0]))
        assert_allclose(inputs1[1], np.array([3, 1]))
        assert_allclose(labels1[0], np.array([1, 0]))

        inputs2, labels2 = val_gen.__next__()
        assert_allclose(inputs2[0], np.array([1, 0]))
        assert_allclose(inputs2[1], np.array([1, 0]))
        assert_allclose(labels2[0], np.array([0, 1]))

        # Should cycle back to the start
        inputs3, labels3 = val_gen.__next__()
        assert_allclose(inputs3, inputs1)
        assert_allclose(labels3, labels1)

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
        val_gen, val_size = self.data_manager.get_validation_data_from_file(
            [self.VALIDATION_FILE],
            pad=False)
        assert val_size == 3
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

        # Should cycle back to the start
        inputs4, labels4 = val_gen.__next__()
        assert_allclose(inputs4[0], inputs1[0])
        assert_allclose(inputs4[1], inputs1[1])
        assert_allclose(labels4[0], labels1[0])

    def test_generate_validation_batches(self):
        val_gen, val_size = self.data_manager.get_validation_data_from_file(
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
        test_gen, test_size = self.data_manager.get_test_data_from_file(
            [self.TEST_FILE])
        test_gen = list(test_gen)
        assert len(test_gen) == 3 == test_size

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

    def test_get_test_data_default_character(self):
        test_gen, test_size = self.data_manager.get_test_data_from_file(
            [self.TEST_FILE], mode="character")
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

    def test_get_test_data_default_word_and_character(self):
        test_gen, test_size = self.data_manager.get_test_data_from_file(
            [self.TEST_FILE], mode="word+character")
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

    def test_get_test_data_pad_with_max_lens(self):
        test_gen, test_size = self.data_manager.get_test_data_from_file(
            [self.TEST_FILE],
            max_lengths={"num_sentence_words": 1})
        test_gen = list(test_gen)
        assert len(test_gen) == 3 == test_size

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
        test_gen, test_size = self.data_manager.get_test_data_from_file(
            [self.TEST_FILE],
            max_instances=2)
        test_gen = list(test_gen)
        assert len(test_gen) == 2 == test_size

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
        test_gen, test_size = self.data_manager.get_test_data_from_file(
            [self.TEST_FILE],
            pad=False)
        test_gen = list(test_gen)
        assert len(test_gen) == 3 == test_size

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
        test_gen, test_size = self.data_manager.get_test_data_from_file(
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
        assert_allclose(inputs[0], np.array([[6, 0]]))
        assert_allclose(inputs[1], np.array([[7, 0]]))

        with self.assertRaises(StopIteration):
            batch_gen.__next__()
