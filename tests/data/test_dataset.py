from duplicate_questions.data.dataset import TextDataset
from duplicate_questions.data.instances.sts_instance import STSInstance

from ..common.test_case import DuplicateTestCase


class TestTextDataset(DuplicateTestCase):
    def test_read_from_train_file(self):
        self.write_duplicate_questions_train_file()
        dataset = TextDataset.read_from_file(self.TRAIN_FILE, STSInstance)
        assert len(dataset.instances) == 3
        instance = dataset.instances[0]
        assert instance.first_sentence == "question1"
        assert instance.second_sentence == "question2"
        assert instance.label == 0
        instance = dataset.instances[1]
        assert instance.first_sentence == "question3"
        assert instance.second_sentence == "question4"
        assert instance.label == 1
        instance = dataset.instances[2]
        assert instance.first_sentence == "question5"
        assert instance.second_sentence == "question6"
        assert instance.label == 0

    def test_read_from_test_file(self):
        self.write_duplicate_questions_test_file()
        dataset = TextDataset.read_from_file(self.TEST_FILE, STSInstance)
        assert len(dataset.instances) == 3
        instance = dataset.instances[0]
        assert instance.first_sentence == "question1"
        assert instance.second_sentence == "question2"
        assert instance.label is None
        instance = dataset.instances[1]
        assert instance.first_sentence == "question3"
        assert instance.second_sentence == "question4"
        assert instance.label is None
        instance = dataset.instances[2]
        assert instance.first_sentence == "question5"
        assert instance.second_sentence == "question6"
        assert instance.label is None
