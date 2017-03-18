# pylint: disable=invalid-name,protected-access
from unittest import TestCase
import codecs
import logging
import os
import shutil


class DuplicateTestCase(TestCase):
    TEST_DIR = './TMP_TEST/'
    TRAIN_FILE = TEST_DIR + 'train_file'
    TEST_FILE = TEST_DIR + 'test_file'

    def setUp(self):
        logging.basicConfig(format=('%(asctime)s - %(levelname)s - '
                                    '%(name)s - %(message)s'),
                            level=logging.INFO)
        os.makedirs(self.TEST_DIR, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.TEST_DIR)

    def write_duplicate_questions_train_file(self):
        with codecs.open(self.TRAIN_FILE, 'w', 'utf-8') as dupe_train_file:
            dupe_train_file.write("\"1\",\"2\",\"3\",\"question1\","
                                  "\"question2\",\"0\"\n")
            dupe_train_file.write("\"4\",\"5\",\"6\",\"question3\","
                                  "\"question4\",\"1\"\n")
            dupe_train_file.write("\"7\",\"8\",\"9\",\"question5\","
                                  "\"question6\",\"0\"\n")

    def write_duplicate_questions_test_file(self):
        with codecs.open(self.TEST_FILE, 'w', 'utf-8') as dupe_test_file:
            dupe_test_file.write("\"1\",\"question1\",\"question2\"\n")
            dupe_test_file.write("\"2\",\"question3\",\"question4\"\n")
            dupe_test_file.write("\"3\",\"question5\",\"question6\"\n")
