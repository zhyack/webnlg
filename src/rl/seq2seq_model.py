import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq

class Seq2SeqModel():
    def __init__(self, config):
        self.learning_rate = config['LR']
        self.batch_size = config['BATCH_SIZE']
        self.input_size = config['INPUT_VOCAB_SIZE']
        self.output_size = config['OUTPUT_VOCAB_SIZE']
        self.hidden_size = config['HIDDEN_SIZE']
        self.buckets = config['BUCKETS']
        config_cell = None
        if config['CELL'] in ['LSTM', 'lstm']:
            config_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
        elif config['CELL'] in ['GRU', 'gru']:
            config_cell = tf.contrib.rnn.GRUCell(self.hidden_size)
        else:
            config_cell = tf.contrib.rnn.BasicRNNCell(self.hidden_size)
        config_cell = tf.contrib.rnn.DropoutWrapper(config_cell, input_keep_prob=config['INPUT_DROPOUT'], output_keep_prob=config['OUTPUT_DROPOUT'], seed=CONFIG['SEED'])
        self.encoder_cell = tf.contrib.rnn.MultiRNNCell([config_cell] * config['ENCODER_LAYERS'])
        self.encoder_cell = tf.contrib.rnn.EmbeddingWrapper(self.encoder_cell, self.input_size, self.hidden_size)
        self.decoder_cell = tf.contrib.rnn.MultiRNNCell([config_cell] * config['DECODER_LAYERS'])
        self.decoder_cell = tf.contrib.rnn.EmbeddingWrapper(self.decoder_cell, self.output_size, self.hidden_size)
        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None, ) name='encoder_inputs_length')
        self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, None), name='decoder_inputs')
        self.decoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(None, ) name='decoder_inputs_length')

        with tf.variable_scope("BidirectionalEncoder") as scope:
            (encoder_fw_outputs,encoder_bw_outputs), (encoder_fw_state, encoder_bw_state) =                 tf.nn.bidirectional_dynamic_rnn( cell_fw=self.encoder_cell, cell_bw=self.encoder_cell, inputs=self.encoder_inputs, sequence_length=self.encoder_inputs_length, time_major=True, dtype=tf.float32)

            self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

            if isinstance(encoder_fw_state, LSTMStateTuple):
                encoder_state_c = tf.concat(
                    (encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
                encoder_state_h = tf.concat(
                    (encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
                self.encoder_states = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
            elif isinstance(encoder_fw_state, tf.Tensor):
                self.encoder_states = tf.concat((encoder_fw_state, encoder_bw_state), 1, name='bidirectional_concat')

        with tf.variable_scope("Decoder") as scope:
            def output_fn(outputs):
                return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)

            # attention_states: size [batch_size, max_time, num_units]
            attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])
            attention_keys, attention_values, attention_score_fn, attention_construct_fn = seq2seq.prepare_attention( attention_states=attention_states, attention_option="bahdanau", num_units=self.decoder_hidden_units)

            decoder_fn_train = seq2seq.attention_decoder_fn_train(encoder_state=self.encoder_state, attention_keys=attention_keys, attention_values=attention_values, attention_score_fn=attention_score_fn, attention_construct_fn=attention_construct_fn, name='attention_decoder')

            decoder_fn_inference = seq2seq.attention_decoder_fn_inference( output_fn=output_fn, encoder_state=self.encoder_states, attention_keys=attention_keys, attention_values=attention_values, attention_score_fn=attention_score_fn, attention_construct_fn=attention_construct_fn, embeddings=self.embedding_matrix, start_of_sequence_id=self.EOS, end_of_sequence_id=self.EOS, maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3, num_decoder_symbols=self.vocab_size)

            self.decoder_outputs, self.decoder_states, self.decoder_context_state_train =  seq2seq.dynamic_rnn_decoder( cell=self.decoder_cell, decoder_fn=decoder_fn_train, inputs=self.decoder_train_inputs_embedded, sequence_length=self.decoder_train_length, time_major=True, scope=scope)

            self.decoder_logits_train = output_fn(self.decoder_outputs_train)
            self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_train')

            scope.reuse_variables()

            (self.decoder_logits_inference,
             self.decoder_state_inference,
             self.decoder_context_state_inference) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_inference,
                    time_major=True,
                    scope=scope,
                )
            )
            self.decoder_prediction_inference = tf.argmax(self.decoder_logits_inference, axis=-1, name='decoder_prediction_inference')



    def train_on_batch(self, encoder_inputs, decoder_inputs, decoder_masks, bucket):
        pass
    def eval_on_batch(self, encoder_inputs, decoder_inputs, decoder_masks, bucket):
        pass
    def predict_on_batch(self, encoder_inputs, bucket):
        pass
    def graph_with_buckets(self, encoder_inputs, decoder_inputs, n_bucket):
        pass
