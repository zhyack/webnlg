import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
import math
import data_utils
import model_utils

class Seq2SeqModel():
    def __init__(self, config):
        self.learning_rate = config['LR']
        self.batch_size = config['BATCH_SIZE']
        self.input_size = config['INPUT_VOCAB_SIZE']
        self.output_size = config['OUTPUT_VOCAB_SIZE']
        self.hidden_size = config['HIDDEN_SIZE']


        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None, ), name='encoder_inputs_length')
        self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name='decoder_inputs')
        self.decoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None, ), name='decoder_inputs_length')
        with tf.variable_scope("embedding") as scope:
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            self.input_word_embedding_matrix = tf.get_variable(name="input_word_embedding_matrix", shape=[self.input_size, self.hidden_size], initializer=initializer, dtype=tf.float32)
            self.output_word_embedding_matrix = tf.get_variable(name="output_word_embedding_matrix", shape=[self.output_size, self.hidden_size], initializer=initializer, dtype=tf.float32)

            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.input_word_embedding_matrix, self.encoder_inputs)

            self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.output_word_embedding_matrix, self.decoder_inputs)

        config_cell = None
        if config['CELL'] in ['LSTM', 'lstm']:
            config_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
        elif config['CELL'] in ['GRU', 'gru']:
            config_cell = tf.contrib.rnn.GRUCell(self.hidden_size)
        else:
            config_cell = tf.contrib.rnn.BasicRNNCell(self.hidden_size)
        config_cell = tf.contrib.rnn.DropoutWrapper(config_cell, input_keep_prob=config['INPUT_DROPOUT'], output_keep_prob=config['OUTPUT_DROPOUT'], seed=config['SEED'])
        # self.encoder_cell = config_cell
        # self.decoder_cell = config_cell
        self.encoder_cell = tf.contrib.rnn.MultiRNNCell([config_cell] * config['ENCODER_LAYERS'])
        self.decoder_cell = tf.contrib.rnn.MultiRNNCell([config_cell] * config['DECODER_LAYERS'])

        with tf.variable_scope("BidirectionalEncoder") as scope:
            ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_states, encoder_bw_states)) =                 tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_cell, cell_bw=self.encoder_cell, inputs=self.encoder_inputs_embedded, sequence_length=self.encoder_inputs_length, time_major=True, dtype=tf.float32)

            self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

            # encoder_fw_state = encoder_fw_states
            # encoder_bw_state = encoder_bw_states
            encoder_fw_state = encoder_fw_states[-1]
            encoder_bw_state = encoder_bw_states[-1]
            # print encoder_bw_state, encoder_fw_state
            # print encoder_bw_outputs
            encoder_state_c = tf.concat(
                (encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
            encoder_state_h = tf.concat(
                (encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
            self.encoder_state =tuple([tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)]*config['DECODER_LAYERS'])

        with tf.variable_scope("AttentionDecoder") as scope:
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.hidden_size, self.encoder_outputs, memory_sequence_length=self.decoder_inputs_length)
            self.decoder_cell =   tf.contrib.seq2seq.AttentionWrapper(self.decoder_cell, attention_mechanism, attention_layer_size=self.hidden_size, output_attention=False)
            initial_state = self.decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            initial_state = initial_state.clone(cell_state=self.encoder_state)

            # print initial_state

            self.train_helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_inputs_embedded, self.decoder_inputs_length, time_major=True)
            self.decoder_train=tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, helper=self.train_helper, initial_state=initial_state)
            self.train_outputs, self.train_state, self.train_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(self.decoder_train, impute_finished=True, maximum_iterations=config['MAX_OUT_LEN'])
            # def output_fn(outputs):
            #     return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)
            # self.train_logits = output_fn(self.train_outputs)
            self.train_prediction = tf.argmax(self.train_outputs, axis=-1, name='decoder_prediction_train')
            self.train_loss = seq2seq.sequence_loss(logits=outputs, targets=self.decoder_inputs)
            self.train_op = tf.train.AdamOptimizer().minimize(self.train_loss)

            scope.reuse_variables()

            self.infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.output_word_embedding_matrix, self.decoder_inputs[0], config['ID_END'])
            self.decoder_infer=tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, helper=self.infer_helper, initial_state=self.encoder_state)
            self.infer_outputs, self.infer_state, self.infer_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(self.decoder_infer, maximum_iterations=config['MAX_OUT_LEN'])
            self.infer_prediction = tf.argmax(self.infer_outputs, axis=-1, name='decoder_prediction_inference')
            self.eval_loss = seq2seq.sequence_loss(logits=self.infer_outputs, targets=self.decoder_inputs)
        self.saver = model_utils.initGlobalSaver()


    def make_train_feed(self, encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length):
        return {
            self.encoder_inputs: encoder_inputs,
            self.encoder_inputs_length: encoder_inputs_length,
            self.decoder_inputs: encoder_inputs,
            self.decoder_inputs_length: decoder_inputs_length,
        }
    def train_on_batch(self, session, encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length):
        train_feed = make_train_feed(encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length)
        _, loss = session.run([self.train_op, self.train_loss], train_feed)
        return loss
    def eval_on_batch(self, session, encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length):
        infer_feed = make_infer_feed(encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length)
        loss, outputs = session.run([self.eval_loss, self.infer_outputs], infer_feed)
        return loss, outputs
    def predict_on_batch(self, session, encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length):
        infer_feed = make_infer_feed(encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length)
        return
