from overrides import overrides
import math
import tensorflow as tf

from duplicate_questions.data.data_manager import DataManager
from duplicate_questions.data.embedding_manager import EmbeddingManager
from duplicate_questions.models.siamese_bilstm.siamese_bilstm import SiameseBiLSTM
from duplicate_questions.data.instances.sts_instance import STSInstance

from ...common.test_case import DuplicateTestCase


class TestSiameseBiLSTM(DuplicateTestCase):
    @overrides
    def setUp(self):
        super(TestSiameseBiLSTM, self).setUp()
        self.write_duplicate_questions_train_file()
        self.write_duplicate_questions_validation_file()
        self.write_duplicate_questions_test_file()
        self.data_manager = DataManager(STSInstance)
        self.batch_size = 2
        self.get_train_gen, self.train_size = self.data_manager.get_train_data_from_file(
            [self.TRAIN_FILE])
        self.get_val_gen, self.val_size = self.data_manager.get_validation_data_from_file(
            [self.VALIDATION_FILE])
        self.get_test_gen, self.test_size = self.data_manager.get_test_data_from_file(
            [self.TEST_FILE])

        self.embedding_manager = EmbeddingManager(self.data_manager.data_indexer)
        self.word_embedding_dim = 5
        self.embedding_matrix = self.embedding_manager.get_embedding_matrix(
            self.word_embedding_dim)
        self.rnn_hidden_size = 6
        self.rnn_output_mode = "last"
        self.output_keep_prob = 1.0
        self.share_encoder_weights = True
        self.config_dict = {
            "mode": "train",
            "word_vocab_size": self.data_manager.data_indexer.get_vocab_size(),
            "word_embedding_dim": self.word_embedding_dim,
            "fine_tune_embeddings": False,
            "word_embedding_matrix": self.embedding_matrix,
            "rnn_hidden_size": self.rnn_hidden_size,
            "rnn_output_mode": self.rnn_output_mode,
            "output_keep_prob": self.output_keep_prob,
            "share_encoder_weights": self.share_encoder_weights
        }
        self.num_train_steps_per_epoch = int(math.ceil(self.train_size / self.batch_size))
        self.num_val_steps = int(math.ceil(self.val_size / self.batch_size))
        self.num_test_steps = int(math.ceil(self.test_size / self.batch_size))

    def test_default_does_not_crash(self):
        # Initialize the model
        model = SiameseBiLSTM(self.config_dict)
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
        loaded_model = SiameseBiLSTM(self.config_dict)
        loaded_model.build_graph()
        loaded_model.predict(get_test_instance_generator=self.get_test_gen,
                             model_load_dir=self.TEST_DIR,
                             batch_size=self.batch_size,
                             num_test_steps=self.num_test_steps)

    def test_mean_pool_does_not_crash(self):
        # Initialize the model
        self.config_dict["rnn_output_mode"] = "mean_pool"
        model = SiameseBiLSTM(self.config_dict)
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
        loaded_model = SiameseBiLSTM(self.config_dict)
        loaded_model.build_graph()
        loaded_model.predict(get_test_instance_generator=self.get_test_gen,
                             model_load_dir=self.TEST_DIR,
                             batch_size=self.batch_size,
                             num_test_steps=self.num_test_steps)

    def test_non_sharing_encoders_does_not_crash(self):
        # Initialize the model
        self.config_dict["share_encoder_weights"] = False
        model = SiameseBiLSTM(self.config_dict)
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
        loaded_model = SiameseBiLSTM(self.config_dict)
        loaded_model.build_graph()
        loaded_model.predict(get_test_instance_generator=self.get_test_gen,
                             model_load_dir=self.TEST_DIR,
                             batch_size=self.batch_size,
                             num_test_steps=self.num_test_steps)
