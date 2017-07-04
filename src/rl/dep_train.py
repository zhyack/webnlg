from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range

from data_trans import dataSeqs2Onehot, loadDict, dataLogits2Seq

full_dict_src, rev_dict_src = loadDict('../modify/dict_src')
full_dict_dst, rev_dict_dst = loadDict('../modify/dict_dst')
f_x = open('../modify/train-webnlg-all-delex.triple','r')
[x_train] = dataSeqs2Onehot(f_x.readlines()[:500], full_dict_src, buckets=[20])
f_y = open('../modify/train-mod.txt','r')
[y_train] = dataSeqs2Onehot(f_y.readlines()[:500], full_dict_dst, buckets=[20])
f_x.close()
f_y.close()
f_x = open('../modify/dev-webnlg-all-delex.triple','r')
[x_eval] = dataSeqs2Onehot(f_x.readlines()[:100], full_dict_src, buckets=[20])
f_y = open('../modify/dev-mod.txt','r')
[y_eval] = dataSeqs2Onehot(f_y.readlines()[:100], full_dict_dst, buckets=[20])

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_eval.shape)
print(y_eval.shape)

# Try replacing GRU, or SimpleRNN.
RNN = layers.LSTM
HIDDEN_SIZE = 100
BATCH_SIZE = 16
LAYERS = 1

print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length,
# use input_shape=(None, num_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(20, 1338)))
# As the decoder RNN's input, repeatedly provide with the last hidden state of
# RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
# length of output, e.g., when DIGITS=3, max output is 999+999=1998.
model.add(layers.RepeatVector(20))
# The decoder RNN could be multiple layers stacked or a single layer.
for _ in range(LAYERS):
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# Apply a dense layer to the every temporal slice of an input. For each of step
# of the output sequence, decide which character should be chosen.
model.add(layers.TimeDistributed(layers.Dense(5216)))
model.add(layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# Train the model each generation and show predictions against the validation
# dataset.
for iteration in range(1, 500):
    for i in range(10):
        ind = np.random.randint(0, len(x_eval))
        rowx, rowy = x_eval[np.array([ind])], y_eval[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        val_seq = dataLogits2Seq(rowx[0], rev_dict_src)
        pred_seq = dataLogits2Seq(preds[0], rev_dict_dst, calc_argmax=False)
        print(val_seq)
        print(pred_seq+'\n---')
    print()
    print('-' * 50)
    print('Iteration', iteration)
    n = 0
    while True:
        if n + BATCH_SIZE > len(x_train):
            break
        model.fit(x_train[n : n+BATCH_SIZE], y_train[n : n+BATCH_SIZE],
                  batch_size=BATCH_SIZE,
                  epochs=1)
        n += BATCH_SIZE
    # Select 10 samples from the validation set at random so we can visualize
    # errors.
    # model.fit(x_train, y_train,
    #           batch_size=BATCH_SIZE,
    #           epochs=1,
    #           validation_data=(x_eval, y_eval))
