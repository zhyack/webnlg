from model_utils import *
from data_utils import _2uni, _2utf8, _2gbk

CONFIG = dict()
CONFIG['LR'] = 0.001
CONFIG['LR_DECAY'] = 0.3
CONFIG['CELL'] = "lstm"
CONFIG['HIDDEN_SIZE'] = 300
CONFIG['ENCODER_LAYERS'] = 1
CONFIG['DECODER_LAYERS'] = 1
CONFIG['INPUT_DROPOUT'] = 1.0
CONFIG['OUTPUT_DROPOUT'] = 0.8
CONFIG['SEED'] = 233333
CONFIG['SRC_DICT']='../data_utils/modify/dict_src'
CONFIG['DST_DICT']='../data_utils/modify/dict_dst'
CONFIG['TRAIN_INPUT']='../data_utils/modify/train-webnlg-all-delex.triple'
CONFIG['TRAIN_OUTPUT']='../data_utils/modify/train-mod.txt'
CONFIG['DEV_INPUT']='../data_utils/modify/dev-webnlg-all-delex.triple'
CONFIG['DEV_OUTPUT']='../data_utils/modify/dev-mod.txt'
CONFIG['MAX_STEPS_PER_ITER']=300
CONFIG['GLOBAL_STEP']=1
CONFIG['ITERS']=100
CONFIG['BATCH_SIZE']=32
CONFIG['MAX_IN_LEN']=50
CONFIG['MAX_OUT_LEN']=70
CONFIG['BUCKETS']=[[5,10], [10,20], [20,40], [30,50], [40,60], [50,70]]



import argparse
parser = argparse.ArgumentParser(
    description="Train a seq2seq model and save in the specified folder.")
parser.add_argument(
    "--s",
    dest="save_folder",
    type=str,
    default=None,
    help="The specified folder to save. If not specified, the model will not be saved.")
parser.add_argument(
    "--l",
    dest="load_folder",
    type=str,
    help="The specified folder to load saved model. If not specified, the model will be initialized.")
args = parser.parse_args()



full_dict_src, rev_dict_src = loadDict(CONFIG['SRC_DICT'])
full_dict_dst, rev_dict_dst = loadDict(CONFIG['DST_DICT'])
CONFIG['INPUT_VOCAB_SIZE']=len(rev_dict_src)
CONFIG['OUTPUT_VOCAB_SIZE']=len(rev_dict_dst)
print(CONFIG['INPUT_VOCAB_SIZE'],CONFIG['OUTPUT_VOCAB_SIZE'])
CONFIG['ID_END']=full_dict_dst['<EOS>']
CONFIG['ID_BOS']=full_dict_dst['<BOS>']
CONFIG['ID_PAD']=full_dict_dst['<PAD>']
CONFIG['ID_UNK']=full_dict_dst['<UNK>']


f_x = open(CONFIG['TRAIN_INPUT'],'r')
x_train_raw = f_x.readlines()
f_y = open(CONFIG['TRAIN_OUTPUT'],'r')
y_train_raw = f_y.readlines()
train_raw = [ [x_train_raw[i].strip(),y_train_raw[i].strip()] for i in range(len(x_train_raw))]
train_buckets_raw = arrangeBuckets(train_raw, CONFIG['BUCKETS'])
print([len(b) for b in train_buckets_raw])
f_x.close()
f_y.close()
f_x = open(CONFIG['DEV_INPUT'],'r')
x_eval_raw = f_x.readlines()
f_y = open(CONFIG['DEV_OUTPUT'],'r')
y_eval_raw = f_y.readlines()
eval_raw = [ [x_eval_raw[i].strip(),y_eval_raw[i].strip()] for i in range(len(x_eval_raw))]
eval_raw = eval_raw[:256]
eval_buckets_raw = arrangeBuckets(eval_raw, CONFIG['BUCKETS'])
print([len(b) for b in eval_buckets_raw])
f_x.close()
f_y.close()

# print len(train_raw), len(eval_raw)
#
# exit(0)



with tf.Session() as sess:
    print('Loading model...')
    Model = instanceOfInitModel(sess, CONFIG)
    if args.load_folder != None:
        Model, CONFIG = loadModelFromFolder(sess, Model.saver, args.load_folder)

    print('Training Begin...')
    log_losses = []
    for n_iter in range(CONFIG['GLOBAL_STEP']/CONFIG['MAX_STEPS_PER_ITER'], CONFIG['ITERS']):
        while True:
            b = random.randint(0, len(CONFIG['BUCKETS'])-1)
            n_b = len(train_buckets_raw[b])
            train_batch = [ train_buckets_raw[b][random.randint(0, n_b-1)] for _ in range(CONFIG['BATCH_SIZE'])]
            train_batch = map(list, zip(*train_batch))
            model_inputs, len_inputs, inputs_mask = dataSeqs2NpSeqs(train_batch[0], full_dict_src, CONFIG['BUCKETS'][b][0])
            model_outputs, len_outputs, outputs_mask = dataSeqs2NpSeqs(train_batch[1], full_dict_dst, CONFIG['BUCKETS'][b][1])
            model_targets, len_targets, targets_mask = dataSeqs2NpSeqs(train_batch[1], full_dict_dst, CONFIG['BUCKETS'][b][1], bias=1)
            batch_loss = Model.train_on_batch(sess, model_inputs, len_inputs, inputs_mask, model_outputs, len_outputs, outputs_mask, model_targets, len_targets, targets_mask)
            print('Train completed for Iter@%d, Step@%d: Loss=%.6f LR=%.6f'%(n_iter, CONFIG['GLOBAL_STEP'], batch_loss, sess.run(Model.learning_rate)))
            CONFIG['GLOBAL_STEP']+=1
            if (CONFIG['GLOBAL_STEP'] % CONFIG['MAX_STEPS_PER_ITER'] == 0):
                break
        print('Iter@%d completed! Start Evaluating...'%(n_iter))
        eval_losses=[]
        for b in range(len(CONFIG['BUCKETS'])):
            n_b = len(eval_buckets_raw[b])
            for k in range((n_b+CONFIG['BATCH_SIZE']-1)/CONFIG['BATCH_SIZE']):
                eval_batch = [ eval_buckets_raw[b][i%n_b] for i in range(k*CONFIG['BATCH_SIZE'], (k+1)*CONFIG['BATCH_SIZE']) ]
                print('Eval process: [%d/%d] [%d/%d]'%(b, len(CONFIG['BUCKETS']), k*CONFIG['BATCH_SIZE'], n_b))
                eval_batch = map(list, zip(*eval_batch))
                model_inputs, len_inputs, inputs_mask = dataSeqs2NpSeqs(eval_batch[0], full_dict_src, CONFIG['BUCKETS'][b][0])
                model_outputs, len_outputs, outputs_mask = dataSeqs2NpSeqs(eval_batch[1], full_dict_dst, CONFIG['BUCKETS'][b][1])
                model_targets, len_targets, targets_mask = dataSeqs2NpSeqs(eval_batch[1], full_dict_dst, CONFIG['BUCKETS'][b][1], bias=1)
                batch_loss, predict_outputs = Model.eval_on_batch(sess, model_inputs, len_inputs, inputs_mask, model_outputs, len_outputs, outputs_mask, model_targets, len_targets, targets_mask)

                eval_losses.append(batch_loss)
                eval_batch = map(list, zip(*eval_batch))
                for i in range(CONFIG['BATCH_SIZE']):
                    if random.random()<0.01:
                        try:
                            print('Raw input: %s\nExpected output: %s\nModel output: %s' % (eval_batch[i][0], eval_batch[i][1], dataLogits2Seq(predict_outputs[i], rev_dict_dst, calc_argmax=True)))
                            # for j in range(len(predict_outputs[i])):
                            #     ll = eval_batch[i][1].split()
                            #     try:
                            #         print('<PAD:%.6f>, <%s:%.6f>' % (predict_outputs[i][j][full_dict_dst['<PAD>']], 'true', predict_outputs[i][j][full_dict_dst[ll[j]]]))
                            #     except:
                            #         pass
                        except UnicodeDecodeError:
                            pass
        print('Evaluationg completed:\nAverage Loss:%.6f'%(sum(eval_losses)/len(eval_losses)))
        print(log_losses[max(0,len(log_losses)-10):])
        log_losses.append(sum(eval_losses)/len(eval_losses))
        if log_losses[-1]>min(log_losses[max(0,len(log_losses)-3):]):
            sess.run(Model.lr_decay_op)
            print('Learning rate trun down-to %.6f'%(sess.run(Model.learning_rate)))
        if args.save_folder != None:
            saveModelToFolder(sess, Model.saver, args.save_folder, CONFIG, Model)
    print('Training Completed...')
