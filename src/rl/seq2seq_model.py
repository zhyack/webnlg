import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
import math
import copy
import data_utils
import model_utils
from contrib_rnn_cell import ExtendedMultiRNNCell
from reward import *
import rlloss

class Seq2SeqModel():
    def __init__(self, config):
        self.rl_enable = config['RL_ENABLE']

        self.learning_rate = tf.Variable(config['LR'], dtype=tf.float32, name='model_learning_rate', trainable=False)

        self.word_embedding_learning_rate = tf.Variable(config['WE_LR'], dtype=tf.float32, name='model_learning_rate', trainable=False)
        self.encoder_learning_rate = tf.Variable(config['ENCODER_LR'], dtype=tf.float32, name='model_learning_rate', trainable=False)
        self.decoder_learning_rate = tf.Variable(config['DECODER_LR'], dtype=tf.float32, name='model_learning_rate', trainable=False)
        if config['SPLIT_LR']:
            def tmp_func():
                self.word_embedding_learning_rate.assign( self.word_embedding_learning_rate * config['LR_DECAY'])
                self.encoder_learning_rate.assign( self.encoder_learning_rate * config['LR_DECAY'])
                self.decoder_learning_rate.assign( self.decoder_learning_rate * config['LR_DECAY'])
            self.lr_decay_op = tmp_func()
        else:
            self.lr_decay_op = self.learning_rate.assign(self.learning_rate * config['LR_DECAY'])

        if config['OPTIMIZER']=='Adam':
            self.optimizer = tf.train.AdamOptimizer
        elif config['OPTIMIZER']=='GD':
            self.optimizer = tf.train.GradientDescentOptimizer
        else:
            raise Exception("Wrong optimizer name...")

        self.global_step = tf.Variable(config['GLOBAL_STEP'], dtype=tf.int32, name='model_global_step', trainable=False)
        self.batch_size = config['BATCH_SIZE']
        self.input_size = config['INPUT_VOCAB_SIZE']
        self.output_size = config['OUTPUT_VOCAB_SIZE']
        self.encoder_hidden_size = config['ENCODER_HIDDEN_SIZE']
        self.decoder_hidden_size = config['DECODER_HIDDEN_SIZE']
        self.embedding_size = config['WORD_EMBEDDING_SIZE']



        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, self.batch_size), name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(self.batch_size), name='encoder_inputs_length')
        self.encoder_inputs_mask = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, None), name='encoder_inputs_mask')
        self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, self.batch_size), name='decoder_inputs')
        self.decoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(self.batch_size), name='decoder_inputs_length')
        self.decoder_inputs_mask = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, None), name='decoder_inputs_mask')
        self.decoder_targets = tf.placeholder(dtype=tf.int32, shape=(None, self.batch_size), name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(dtype=tf.int32, shape=(self.batch_size), name='decoder_targets_length')
        self.decoder_targets_mask = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, None), name='decoder_targets_mask')
        self.rewards = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, None, self.output_size), name='decoder_targets_mask')

        # with tf.variable_scope("model_vars", regularizer = tf.contrib.layers.l2_regularizer(config['VAR_NORM_BETA'])) as global_scope:





        with tf.variable_scope("WordEmbedding") as scope:
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            self.input_word_embedding_matrix = tf.get_variable(name="input_word_embedding_matrix", shape=[self.input_size, self.embedding_size], initializer=initializer, dtype=tf.float32)
            self.output_word_embedding_matrix = tf.get_variable(name="output_word_embedding_matrix", shape=[self.output_size, self.embedding_size], initializer=initializer, dtype=tf.float32)

            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.input_word_embedding_matrix, self.encoder_inputs)

            self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.output_word_embedding_matrix, self.decoder_inputs)
            print('Embedding Trainable Variables')
            self.embedding_variables = scope.trainable_variables()
            print(self.embedding_variables)





        self.encoder_cell = []
        for _ in range(config['ENCODER_LAYERS']):
            config_encoder_cell = None
            if config['CELL'] in ['LSTM', 'lstm']:
                config_encoder_cell = tf.contrib.rnn.BasicLSTMCell(self.encoder_hidden_size)
            elif config['CELL'] in ['GRU', 'gru']:
                onfig_encoder_cell = tf.contrib.rnn.GRUCell(self.encoder_hidden_size)
            else:
                config_encoder_cell = tf.contrib.rnn.BasicRNNCell(self.encoder_hidden_size)
            config_encoder_cell = tf.contrib.rnn.DropoutWrapper(config_encoder_cell, input_keep_prob=config['INPUT_DROPOUT'], output_keep_prob=config['OUTPUT_DROPOUT'])
            self.encoder_cell.append(config_encoder_cell)
        self.encoder_cell = ExtendedMultiRNNCell(self.encoder_cell)


        with tf.variable_scope("DynamicEncoder") as scope:
            if config['BIDIRECTIONAL_ENCODER']:
                self.encoder_fw_cell = copy.deepcopy(self.encoder_cell)
                self.encoder_bw_cell = copy.deepcopy(self.encoder_cell)
                ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_states, encoder_bw_states)) =                 tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_fw_cell, cell_bw=self.encoder_bw_cell, inputs=self.encoder_inputs_embedded, sequence_length=self.encoder_inputs_length, time_major=True, dtype=tf.float32)
                self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
                encoder_fw_state = encoder_fw_states[-1]
                encoder_bw_state = encoder_bw_states[-1]
                # encoder_state_c = tf.concat(
                #     (encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
                # encoder_state_h = tf.concat(
                #     (encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
                # self.encoder_state =tuple([tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)]*config['DECODER_LAYERS'])
                # self.encoder_state = encoder_fw_state
                encoder_state_c = (encoder_fw_state.c+encoder_bw_state.c)/2.0
                encoder_state_h = (encoder_fw_state.h+encoder_bw_state.h)/2.0
                self.encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
            else:
                self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(cell=self.encoder_cell, inputs=self.encoder_inputs_embedded, sequence_length=self.encoder_inputs_length, time_major=True, dtype=tf.float32)
            print('Encoder Trainable Variables')
            self.encoder_variables = scope.trainable_variables()
            print(self.encoder_variables)






        self.decoder_cell = []
        for _ in range(config['DECODER_LAYERS']):
            config_decoder_cell = None
            if config['CELL'] in ['LSTM', 'lstm']:
                config_decoder_cell = tf.contrib.rnn.BasicLSTMCell(self.decoder_hidden_size)
            elif config['CELL'] in ['GRU', 'gru']:
                config_decoder_cell = tf.contrib.rnn.GRUCell(self.decoder_hidden_size)
            else:
                config_decoder_cell = tf.contrib.rnn.BasicRNNCell(self.decoder_hidden_size)
            config_decoder_cell = tf.contrib.rnn.DropoutWrapper(config_decoder_cell, input_keep_prob=config['INPUT_DROPOUT'], output_keep_prob=config['OUTPUT_DROPOUT'])
            self.decoder_cell.append(config_decoder_cell)
        self.decoder_cell = ExtendedMultiRNNCell(self.decoder_cell)
        if config['ATTENTION_DECODER']:
            if config['ATTENTION_MECHANISE']=='LUONG':
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.decoder_hidden_size, tf.transpose(self.encoder_outputs,perm=[1,0,2]), memory_sequence_length=self.encoder_inputs_length, scale=True)
                self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(self.decoder_cell, attention_mechanism, attention_layer_size=self.decoder_hidden_size, output_attention=True)
            elif config['ATTENTION_MECHANISE']=='BAHDANAU':
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.decoder_hidden_size, tf.transpose(self.encoder_outputs,perm=[1,0,2]), memory_sequence_length=self.encoder_inputs_length, normalize=True)
                self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(self.decoder_cell, attention_mechanism, attention_layer_size=self.decoder_hidden_size, output_attention=False)
            else:
                raise Exception('config[\'ATTENTION_MECHANISE\'] should be LUONG or BAHDANAU')

        self.decoder_cell = tf.contrib.rnn.OutputProjectionWrapper( self.decoder_cell, output_size = self.output_size)

        with tf.variable_scope("DynamicDecoder") as scope:
            initial_state = self.decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

            if config['ATTENTION_DECODER']:
                cat_state = tuple([self.encoder_state] + list(initial_state.cell_state)[:-1])
                initial_state.clone(cell_state=cat_state)
            else:
                initial_state = tuple([self.encoder_state] + list(initial_state[:-1]))

            self.train_helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_inputs_embedded, self.decoder_inputs_length, time_major=True)
            self.decoder_train=tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, helper=self.train_helper, initial_state=initial_state)
            self.train_outputs, self.train_state, self.train_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(self.decoder_train, impute_finished=True, maximum_iterations=None)




            self.infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.output_word_embedding_matrix, self.decoder_inputs[0], self.output_size+1)
            self.decoder_infer=tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, helper=self.infer_helper, initial_state=initial_state)
            self.infer_outputs, self.infer_state, self.infer_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(self.decoder_infer, impute_finished=False, maximum_iterations=config['MAX_OUT_LEN'])


            self.train_outputs = self.train_outputs.rnn_output
            self.infer_outputs = self.infer_outputs.rnn_output
            self.eval_outputs = tf.slice(self.infer_outputs, [0,0,0], [-1,tf.reduce_max(self.decoder_inputs_length),-1])

            self.train_loss = seq2seq.sequence_loss(logits=self.train_outputs, targets=tf.transpose(self.decoder_targets, perm=[1,0]), weights=self.decoder_targets_mask)
            self.train_loss_rl = rlloss.sequence_loss_rl(logits=self.train_outputs, rewards=self.rewards, weights=self.decoder_targets_mask)


            self.eval_loss = seq2seq.sequence_loss(logits=self.eval_outputs, targets=tf.transpose(self.decoder_targets, perm=[1,0]), weights=self.decoder_targets_mask)
            print('Decoder Trainable Variables')
            self.decoder_variables = scope.trainable_variables()
            print(self.decoder_variables)

            if config['TRAIN_ON_EACH_STEP']:
                self.final_loss = self.train_loss
                if config['RL_ENABLE']:
                    self.final_loss = self.final_loss + self.train_loss_rl
            else:
                self.final_loss = self.eval_loss

        print('All Trainable Variables:')
        self.all_trainable_variables = tf.trainable_variables()
        print(self.all_trainable_variables)
        if config['CLIP']:
            def updateBP(loss, lr, var_list):

                return [tf.contrib.layers.optimize_loss( loss=loss, global_step=self.global_step, learning_rate=lr[i], optimizer=self.optimizer, clip_gradients=config['CLIP_NORM'], variables=var_list[i], learning_rate_decay_fn=model_utils.create_learning_rate_decay_fn(decay_rate=1-config['LR_DECAY'], decay_steps=config['MAX_STEPS_PER_ITER'])) for i in range(len(lr))]

            if config['SPLIT_LR']:
                self.train_op = updateBP(self.final_loss, [config['WE_LR'], config['ENCODER_LR'], config['DECODER_LR']], [self.embedding_variables, self.encoder_variables, self.decoder_variables])
            else:
                self.train_op = updateBP(self.final_loss, [config['LR']], [self.all_trainable_variables])
        else:
            def updateBP(loss, lr, var_list):
                return                    [tf.train.AdamOptimizer(learning_rate=lr[i]).minimize(loss, var_list=var_list[i]) for i in range(len(lr))]
            if config['SPLIT_LR']:
                self.train_op = updateBP(self.final_loss, [config['WE_LR'], config['ENCODER_LR'], config['DECODER_LR']], [self.embedding_variables, self.encoder_variables, self.decoder_variables])
            else:
                self.train_op = updateBP(self.final_loss, [config['LR']], [self.all_trainable_variables])




    def make_train_feed(self, encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask):
        return {
            self.encoder_inputs: encoder_inputs,
            self.encoder_inputs_length: encoder_inputs_length,
            self.encoder_inputs_mask: encoder_inputs_mask,
            self.decoder_inputs: decoder_inputs,
            self.decoder_inputs_length: decoder_inputs_length,
            self.decoder_inputs_mask: decoder_inputs_mask,
            self.decoder_targets: decoder_targets,
            self.decoder_targets_length: decoder_targets_length,
            self.decoder_targets_mask: decoder_targets_mask,
        }
    def make_infer_feed(self, encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask):
        return {
            self.encoder_inputs: encoder_inputs,
            self.encoder_inputs_length: encoder_inputs_length,
            self.encoder_inputs_mask: encoder_inputs_mask,
            self.decoder_inputs: decoder_inputs,
            self.decoder_inputs_length: decoder_inputs_length,
            self.decoder_inputs_mask: decoder_inputs_mask,
            self.decoder_targets: decoder_targets,
            self.decoder_targets_length: decoder_targets_length,
            self.decoder_targets_mask: decoder_targets_mask,
        }
    def train_on_batch(self, session, encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask, rev_dict_src, dict_dst):
        train_feed = self.make_train_feed(encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask)
        if self.rl_enable:
            [outputs] = session.run([self.train_outputs], train_feed)
            r = contentPenalty(np.transpose(encoder_inputs), outputs, rev_dict_src, dict_dst)
            train_feed[self.rewards] = r
            _, ce_loss, rl_loss = session.run([self.train_op, self.train_loss, self.train_loss_rl], train_feed)
            return [ce_loss, rl_loss]
        else:
            _, loss = session.run([self.train_op, self.final_loss], train_feed)
            return loss
    def eval_on_batch(self, session, encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask):
        infer_feed = self.make_infer_feed(encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask)
        loss, outputs = session.run([self.eval_loss, self.infer_outputs], infer_feed)
        return loss, outputs
    def predict_on_batch(self, session, encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask):
        infer_feed = self.make_infer_feed(encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask)
        return
