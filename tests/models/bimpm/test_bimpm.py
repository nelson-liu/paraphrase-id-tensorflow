from overrides import overrides
import math
import tensorflow as tf

from duplicate_questions.data.data_manager import DataManager
from duplicate_questions.data.embedding_manager import EmbeddingManager
from duplicate_questions.models.bimpm.bimpm import BiMPM
from duplicate_questions.data.instances.sts_instance import STSInstance

from ...common.test_case import DuplicateTestCase


class TestBiMPM(DuplicateTestCase):
    @overrides
    def setUp(self):
        super(TestBiMPM, self).setUp()
        self.write_duplicate_questions_train_file()
        self.write_duplicate_questions_validation_file()
        self.write_duplicate_questions_test_file()
        self.data_manager = DataManager(STSInstance)
        self.batch_size = 3
        self.get_train_gen, self.train_size = self.data_manager.get_train_data_from_file(
            [self.TRAIN_FILE],
            mode="word+character")
        self.get_val_gen, self.val_size = self.data_manager.get_validation_data_from_file(
            [self.VALIDATION_FILE],
            mode="word+character")
        self.get_test_gen, self.test_size = self.data_manager.get_test_data_from_file(
            [self.TEST_FILE],
            mode="word+character")

        self.embedding_manager = EmbeddingManager(self.data_manager.data_indexer)
        self.word_embedding_dim = 5
        self.word_embedding_matrix = self.embedding_manager.get_embedding_matrix(
            self.word_embedding_dim)
        self.char_embedding_dim = 2
        self.char_embedding_matrix = self.embedding_manager.get_embedding_matrix(
            self.char_embedding_dim)
        self.char_rnn_hidden_size = 6
        self.context_rnn_hidden_size = 3
        self.aggregation_rnn_hidden_size = 4
        self.dropout_ratio = 0.1
        self.config_dict = {
            "mode": "train",
            "word_vocab_size": self.data_manager.data_indexer.get_vocab_size(),
            "word_embedding_dim": self.word_embedding_dim,
            "word_embedding_matrix": self.word_embedding_matrix,
            "char_vocab_size": self.data_manager.data_indexer.get_vocab_size(
                namespace="characters"),
            "char_embedding_dim": self.char_embedding_dim,
            "char_embedding_matrix": self.char_embedding_matrix,
            "char_rnn_hidden_size": self.char_rnn_hidden_size,
            "fine_tune_embeddings": False,
            "context_rnn_hidden_size": self.context_rnn_hidden_size,
            "aggregation_rnn_hidden_size": self.aggregation_rnn_hidden_size,
            "dropout_ratio": self.dropout_ratio
        }

        self.num_train_steps_per_epoch = int(math.ceil(self.train_size / self.batch_size))
        self.num_val_steps = int(math.ceil(self.val_size / self.batch_size))
        self.num_test_steps = int(math.ceil(self.test_size / self.batch_size))

    def test_default_does_not_crash(self):
        # Initialize the model
        model = BiMPM(self.config_dict)
        model.build_graph()
        # Train the model
        model.train(get_train_instance_generator=self.get_train_gen,
                    get_val_instance_generator=self.get_val_gen,
                    batch_size=self.batch_size,
                    num_train_steps_per_epoch=self.num_train_steps_per_epoch,
                    num_epochs=2,
                    num_val_steps=self.num_val_steps,
                    save_path=self.TEST_DIR,
                    log_path=self.TEST_DIR,
                    log_period=2,
                    val_period=2,
                    save_period=2,
                    patience=0)

        tf.reset_default_graph()
        # Load and predict with the model
        self.config_dict["mode"] = "test"
        del self.config_dict["word_embedding_matrix"]
        del self.config_dict["char_embedding_matrix"]
        loaded_model = BiMPM(self.config_dict)
        loaded_model.build_graph()
        loaded_model.predict(get_test_instance_generator=self.get_test_gen,
                             model_load_dir=self.TEST_DIR,
                             batch_size=self.batch_size,
                             num_test_steps=self.num_test_steps)
