# pylint: disable=invalid-name,protected-access
from unittest import TestCase
import codecs
import logging
import os
import shutil


class DuplicateTestCase(TestCase):
    TEST_DIR = './TMP_TEST/'
    TRAIN_FILE = TEST_DIR + 'train_file'
    VALIDATION_FILE = TEST_DIR + 'validation_file'
    TEST_FILE = TEST_DIR + 'test_file'
    VECTORS_FILE = TEST_DIR + 'vectors_file'

    def setUp(self):
        logging.basicConfig(format=('%(asctime)s - %(levelname)s - '
                                    '%(name)s - %(message)s'),
                            level=logging.INFO)
        os.makedirs(self.TEST_DIR, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.TEST_DIR)

    def write_duplicate_questions_train_file(self):
        with codecs.open(self.TRAIN_FILE, 'w', 'utf-8') as dupe_train_file:
            dupe_train_file.write("\"1\",\"1\",\"2\",\"question1\","
                                  "\"question2 question3\",\"0\"\n")
            dupe_train_file.write("\"2\",\"3\",\"4\",\"question4\","
                                  "\"question5\",\"1\"\n")
            dupe_train_file.write("\"3\",\"5\",\"6\",\"question6\","
                                  "\"question7\",\"0\"\n")

    def write_duplicate_questions_validation_file(self):
        with codecs.open(self.VALIDATION_FILE, 'w',
                         'utf-8') as dupe_val_file:
            dupe_val_file.write("\"1\",\"7\",\"8\",\"question1\","
                                "\"question2 question8\",\"0\"\n")
            dupe_val_file.write("\"2\",\"9\",\"10\",\"question9\","
                                "\"question10\",\"1\"\n")
            dupe_val_file.write("\"3\",\"11\",\"12\",\"question6\","
                                "\"question7 question11 question12\","
                                "\"0\"\n")

    def write_duplicate_questions_test_file(self):
        with codecs.open(self.TEST_FILE, 'w', 'utf-8') as dupe_test_file:
            dupe_test_file.write("\"1\",\"question1 questionunk1 question1\","
                                 "\"questionunk2\"\n")
            dupe_test_file.write("\"2\",\"question3\","
                                 "\"question4 questionunk3\"\n")
            dupe_test_file.write("\"3\",\"question5\",\"question6\"\n")

    def write_vector_file(self):
        with codecs.open(self.VECTORS_FILE, 'w', 'utf-8') as vectors_file:
            vectors_file.write("word1 0.0 1.1 0.2\n")
            vectors_file.write("word2 0.1 0.4 -4.0\n")
