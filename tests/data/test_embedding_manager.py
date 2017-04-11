import codecs
from numpy.testing import assert_allclose
import numpy as np
from overrides import overrides

from duplicate_questions.data.embedding_manager import EmbeddingManager
from duplicate_questions.data.data_indexer import DataIndexer

from ..common.test_case import DuplicateTestCase


class TestEmbeddingManager(DuplicateTestCase):
    @overrides
    def setUp(self):
        super(TestEmbeddingManager, self).setUp()
        self.write_vector_file()
        self.data_indexer = DataIndexer()
        self.data_indexer.add_word_to_index("word1")
        self.data_indexer.add_word_to_index("word2")
        self.data_indexer.is_fit = True
        self.embedding_dict = {"word1": np.array([5.1, 7.2, -0.2]),
                               "word2": np.array([0.8, 0.1, 0.9])}
        self.embedding_manager = EmbeddingManager(self.data_indexer)

    def test_get_embedding_matrix_reads_data_file(self):
        embed_mat = self.embedding_manager.get_embedding_matrix(
            3,
            pretrained_embeddings_file_path=self.VECTORS_FILE)
        assert_allclose(embed_mat[2], np.array([0.0, 1.1, 0.2]))
        assert_allclose(embed_mat[3], np.array([0.1, 0.4, -4.0]))

    def test_get_embedding_matrix_reads_dict(self):
        embed_mat = self.embedding_manager.get_embedding_matrix(
            3,
            pretrained_embeddings_dict=self.embedding_dict)
        assert_allclose(embed_mat[2], np.array([5.1, 7.2, -0.2]))
        assert_allclose(embed_mat[3], np.array([0.8, 0.1, 0.9]))

    def test_get_embedding_matrix_dict_overrides_file(self):
        embed_mat = self.embedding_manager.get_embedding_matrix(
            3,
            pretrained_embeddings_file_path=self.VECTORS_FILE,
            pretrained_embeddings_dict=self.embedding_dict)
        assert_allclose(embed_mat[2], np.array([5.1, 7.2, -0.2]))
        assert_allclose(embed_mat[3], np.array([0.8, 0.1, 0.9]))

    def test_get_embedding_matrix_reproducible(self):
        embed_mat_1_random = self.embedding_manager.get_embedding_matrix(100)
        embed_mat_2_random = self.embedding_manager.get_embedding_matrix(100)
        assert_allclose(embed_mat_1_random, embed_mat_2_random)

        embed_mat_1_file = self.embedding_manager.get_embedding_matrix(
            3,
            pretrained_embeddings_file_path=self.VECTORS_FILE)
        embed_mat_2_file = self.embedding_manager.get_embedding_matrix(
            3,
            pretrained_embeddings_file_path=self.VECTORS_FILE)
        assert_allclose(embed_mat_1_file, embed_mat_2_file)

        embed_mat_1_dict = self.embedding_manager.get_embedding_matrix(
            3,
            pretrained_embeddings_dict=self.embedding_dict)
        embed_mat_2_dict = self.embedding_manager.get_embedding_matrix(
            3,
            pretrained_embeddings_dict=self.embedding_dict)
        assert_allclose(embed_mat_1_dict, embed_mat_2_dict)

    def test_embedding_manager_errors(self):
        with self.assertRaises(ValueError):
            unfitted_data_indexer = DataIndexer()
            EmbeddingManager(unfitted_data_indexer)
        with self.assertRaises(ValueError):
            EmbeddingManager.initialize_random_matrix((19,))
        with self.assertRaises(ValueError):
            EmbeddingManager.initialize_random_matrix((19, 100, 100))
        with self.assertRaises(ValueError):
            self.embedding_manager.get_embedding_matrix(5.0)
        with self.assertRaises(ValueError):
            self.embedding_manager.get_embedding_matrix("5")
        with self.assertRaises(ValueError):
            self.embedding_manager.get_embedding_matrix(
                5,
                pretrained_embeddings_file_path=["some_path"])
        with self.assertRaises(ValueError):
            self.embedding_manager.get_embedding_matrix(
                5,
                pretrained_embeddings_dict=["list", [0.1, 0.2]])
        with self.assertRaises(ValueError):
            self.embedding_manager.get_embedding_matrix(
                5,
                pretrained_embeddings_file_path=self.VECTORS_FILE)
        with self.assertRaises(ValueError):
            self.embedding_manager.get_embedding_matrix(
                5,
                pretrained_embeddings_dict=self.embedding_dict)
        with self.assertRaises(ValueError):
            bad_dict = {"word1": np.array([0.1, 0.2]),
                        "word2": np.array([0.3, 0.4, 0.5])}
            self.embedding_manager.get_embedding_matrix(
                5,
                pretrained_embeddings_dict=bad_dict)
        with self.assertRaises(ValueError):
            bad_vectors_path = self.TEST_DIR + 'bad_vectors_file'
            with codecs.open(bad_vectors_path, 'w', 'utf-8') as vectors_file:
                vectors_file.write("word1 0.0 1.1 0.2\n")
                vectors_file.write("word2 0.1 0.4\n")
            self.embedding_manager.get_embedding_matrix(
                3,
                pretrained_embeddings_file_path=bad_vectors_path)
        with self.assertRaises(ValueError):
            bad_vectors_path = self.TEST_DIR + 'bad_vectors_file'
            with codecs.open(bad_vectors_path, 'w', 'utf-8') as vectors_file:
                vectors_file.write("word0 0.0\n")
                vectors_file.write("word1 0.0 1.1 0.2\n")
                vectors_file.write("word2 0.1 0.4\n")
            self.embedding_manager.get_embedding_matrix(
                3,
                pretrained_embeddings_file_path=bad_vectors_path)
