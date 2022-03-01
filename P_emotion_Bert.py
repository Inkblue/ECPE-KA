# encoding: utf-8


import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import sys, os, time, codecs, pdb

from utils.tf_funcs import *
from utils.prepare_data import *
from utils.load_POS import *

FLAGS = tf.app.flags.FLAGS
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
## embedding parameters ##
tf.app.flags.DEFINE_string('w2v_file', 'data_combine/w2v_200.txt', 'embedding file')
tf.app.flags.DEFINE_string('SC_LIWC_file', 'data_combine/sc_liwc.dic', 'Simplified Chinese LIWC file')  ##
tf.app.flags.DEFINE_string('POS_result', 'data_combine/Word_seg.txt', 'Word segmentation and POS results')  ##
##tf.app.flags.DEFINE_integer('embedding_dim', 768, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('embedding_dim_word', 768, 'dimension of word embedding')  ##
tf.app.flags.DEFINE_integer('embedding_dim_pos', 50, 'dimension of position embedding')
tf.app.flags.DEFINE_integer('embedding_dim_LIWC', 71, 'dimension of LIWC embedding')  ##
tf.app.flags.DEFINE_integer('embedding_dim_all', 848, 'dimension of word embedding')  ##768+71+9(词向量+LIWC+词性)
## input struct ##
tf.app.flags.DEFINE_integer('max_sen_len', 30, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_doc_len', 75, 'max number of tokens per documents')
## model struct ##
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
# >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_string('log_file_name', '', 'name of log file')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_integer('training_iter', 15, 'number of train iter')
tf.app.flags.DEFINE_string('scope', 'RNN', 'RNN scope')
# not easy to tune , a good posture of using data to train model is very important
tf.app.flags.DEFINE_integer('batch_size', 16, 'number of example per batch')
tf.app.flags.DEFINE_float('learning_rate', 0.0025, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.50, 'word embedding training dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'softmax layer dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')
tf.app.flags.DEFINE_float('cause', 1.000, 'lambda1')
tf.app.flags.DEFINE_float('pos', 1.00, 'lambda2')


def build_model(wordPOS_embedding, LIWC_embedding, word_embedding, x, sen_len, doc_len, keep_prob1, keep_prob2, y_position, y_cause, RNN=biLSTM):
    ##x = tf.nn.embedding_lookup(word_embedding, x)  # 选取一个张量里面索引对应的元素。x为索引
    embedding_tmp = tf.concat([word_embedding,LIWC_embedding],1)  ##
    embedding_all = tf.concat([embedding_tmp, wordPOS_embedding], 1)  ##
    x = tf.nn.embedding_lookup(embedding_all, x)  ##
    ##inputs = tf.reshape(x,[-1, FLAGS.max_sen_len, FLAGS.embedding_dim])  # 将x变成行数不确定，但列数为max_sen_len，深度为embedding_dim的张量
    inputs = tf.reshape(x, [-1, FLAGS.max_sen_len,FLAGS.embedding_dim_all])  ## 将x变成行数不确定，但列数为max_sen_len，深度为embedding_dim_all的张量
    inputs = tf.nn.dropout(inputs, keep_prob=keep_prob1)
    sen_len = tf.reshape(sen_len, [-1])  # 将sen_len变成1维

    def get_s(inputs, name):
        with tf.name_scope('word_encode'):
            inputs = RNN(inputs, sen_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'word_layer' + name)
        with tf.name_scope('word_attention'):
            sh2 = 2 * FLAGS.n_hidden
            w1 = get_weight_varible('word_att_w1' + name, [sh2, sh2])
            b1 = get_weight_varible('word_att_b1' + name, [sh2])
            w2 = get_weight_varible('word_att_w2' + name, [sh2, 1])
            s = att_var(inputs, sen_len, w1, b1, w2)
        s = tf.reshape(s, [-1, FLAGS.max_doc_len, 2 * FLAGS.n_hidden])
        return s

    s = get_s(inputs, name='cause_word_encode')
    s = RNN(s, doc_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'cause_sentence_layer')
    with tf.name_scope('sequence_prediction'):
        s1 = tf.reshape(s, [-1, 2 * FLAGS.n_hidden])
        s1 = tf.nn.dropout(s1, keep_prob=keep_prob2)

        w_cause = get_weight_varible('softmax_w_cause', [2 * FLAGS.n_hidden, FLAGS.n_class])
        b_cause = get_weight_varible('softmax_b_cause', [FLAGS.n_class])
        pred_cause = tf.nn.softmax(tf.matmul(s1, w_cause) + b_cause)
        pred_cause = tf.reshape(pred_cause, [-1, FLAGS.max_doc_len, FLAGS.n_class])

    s = get_s(inputs, name='pos_word_encode')
    s = tf.concat([s, pred_cause], 2)
    s = RNN(s, doc_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope + 'pos_sentence_layer')
    with tf.name_scope('sequence_prediction'):
        s1 = tf.reshape(s, [-1, 2 * FLAGS.n_hidden])
        s1 = tf.nn.dropout(s1, keep_prob=keep_prob2)

        w_pos = get_weight_varible('softmax_w_pos', [2 * FLAGS.n_hidden, FLAGS.n_class])
        b_pos = get_weight_varible('softmax_b_pos', [FLAGS.n_class])
        pred_pos = tf.nn.softmax(tf.matmul(s1, w_pos) + b_pos)
        pred_pos = tf.reshape(pred_pos, [-1, FLAGS.max_doc_len, FLAGS.n_class])

    reg = tf.nn.l2_loss(w_cause) + tf.nn.l2_loss(b_cause)
    reg += tf.nn.l2_loss(w_pos) + tf.nn.l2_loss(b_pos)
    return pred_cause, pred_pos, reg


def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, lr-{}, kb1-{}, kb2-{}, l2_reg-{}'.format(
        FLAGS.batch_size, FLAGS.learning_rate, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.l2_reg))
    print('training_iter-{}, scope-{}\n'.format(FLAGS.training_iter, FLAGS.scope))


def get_batch_data(x, sen_len, doc_len, keep_prob1, keep_prob2, y_position, y_cause, batch_size, test=False):
    for index in batch_index(len(y_cause), batch_size, test):
        feed_list = [x[index], sen_len[index], doc_len[index], keep_prob1, keep_prob2, y_position[index],
                     y_cause[index]]
        yield feed_list, len(index)


def run():
    save_dir = 'pair_data/{}/'.format(FLAGS.scope)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if FLAGS.log_file_name:
        sys.stdout = open(save_dir + FLAGS.log_file_name, 'w')
    print_time()
    tf.reset_default_graph()
    # Model Code Block
    word_idx_rev, word_id_mapping, word_embedding, pos_embedding = load_bert_embedding(FLAGS.embedding_dim_word,FLAGS.embedding_dim_pos,'data_combine/clause_keywords.csv')
    word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
    pos_embedding = tf.constant(pos_embedding, dtype=tf.float32, name='pos_embedding')

    LIWC_embedding = load_LIWC(FLAGS.SC_LIWC_file, 'data_combine/clause_keywords.csv', FLAGS.embedding_dim_LIWC)  ##
    LIWC_embedding = tf.constant(LIWC_embedding, dtype=tf.float32, name='LIWC_embedding')  ##

    wordPOS_embedding = load_POS(FLAGS.POS_result, 'data_combine/clause_keywords.csv')
    wordPOS_embedding = tf.constant(wordPOS_embedding, dtype=tf.float32, name='wordPOS_embedding')

    print('build model...')

    x = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len, FLAGS.max_sen_len])
    sen_len = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
    doc_len = tf.placeholder(tf.int32, [None])
    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)
    y_position = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    y_cause = tf.placeholder(tf.float32, [None, FLAGS.max_doc_len, FLAGS.n_class])
    placeholders = [x, sen_len, doc_len, keep_prob1, keep_prob2, y_position, y_cause]

    pred_pos, pred_cause, reg = build_model(wordPOS_embedding, LIWC_embedding, word_embedding, x, sen_len, doc_len, keep_prob1, keep_prob2, y_position,y_cause)
    valid_num = tf.cast(tf.reduce_sum(doc_len), dtype=tf.float32)
    loss_pos = - tf.reduce_sum(y_position * tf.log(pred_pos)) / valid_num
    loss_cause = - tf.reduce_sum(y_cause * tf.log(pred_cause)) / valid_num
    loss_op = loss_cause * FLAGS.cause + loss_pos * FLAGS.pos + reg * FLAGS.l2_reg
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_op)

    true_y_cause_op = tf.argmax(y_cause, 2)  # tf.argmax(input,axis)根据axis取值的不同返回每行或者每列最大值的索引。axis=0为列，axis=1为行
    pred_y_cause_op = tf.argmax(pred_cause, 2)  # 得到每个句子是否原因子句的0-1标签
    true_y_pos_op = tf.argmax(y_position, 2)
    pred_y_pos_op = tf.argmax(pred_pos, 2)  # 得到每个句子是否情感子句的0-1标签
    pred_all0_pos_op = pred_pos  ##
    pred_all0_cause_op = pred_cause  ##
    print('build model done!\n')

    # Training Code Block
    print_training_info()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        acc_cause_list, p_cause_list, r_cause_list, f1_cause_list = [], [], [], []
        acc_pos_list, p_pos_list, r_pos_list, f1_pos_list = [], [], [], []
        p_pair_list, r_pair_list, f1_pair_list = [], [], []

        for fold in range(1, 11):
            sess.run(tf.global_variables_initializer())
            # train for one fold
            print('############# fold {} begin ###############'.format(fold))
            # Data Code Block
            train_file_name = 'fold{}_train.txt'.format(fold)
            test_file_name = 'fold{}_test.txt'.format(fold)
            tr_doc_id, tr_y_position, tr_y_cause, tr_y_pairs, tr_x, tr_sen_len, tr_doc_len = load_data('data_combine/' + train_file_name, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len)
            te_doc_id, te_y_position, te_y_cause, te_y_pairs, te_x, te_sen_len, te_doc_len = load_data('data_combine/' + test_file_name, word_id_mapping, FLAGS.max_doc_len, FLAGS.max_sen_len)
            max_f1_cause, max_f1_pos, max_f1_avg = [-1.] * 3
            epoch_loss = []  # 定义每一个epoch的损失，情感子句预测F1值，原因子句预测F1值
            epoch_cause = []
            epoch_emo = []
            print('train docs: {}    test docs: {}'.format(len(tr_x), len(te_x)))
            for i in range(FLAGS.training_iter):
                start_time, step = time.time(), 1
                # train
                for train, _ in get_batch_data(tr_x, tr_sen_len, tr_doc_len, FLAGS.keep_prob1, FLAGS.keep_prob2,
                                               tr_y_position, tr_y_cause, FLAGS.batch_size):
                    _, loss, pred_y_cause, true_y_cause, pred_y_pos, true_y_pos, doc_len_batch = sess.run(
                        [optimizer, loss_op, pred_y_cause_op, true_y_cause_op, pred_y_pos_op, true_y_pos_op, doc_len],
                        feed_dict=dict(zip(placeholders, train)))
                    if step % 10 == 0:
                        print('step {}: train loss {:.4f} '.format(step, loss))
                    step = step + 1
                epoch_loss.append(loss)
                # test
                test = [te_x, te_sen_len, te_doc_len, 1., 1., te_y_position, te_y_cause]
                loss, pred_y_cause, true_y_cause, pred_y_pos, true_y_pos, doc_len_batch, pred_pos, pred_cause = sess.run(
                    [loss_op, pred_y_cause_op, true_y_cause_op, pred_y_pos_op, true_y_pos_op, doc_len, pred_all0_pos_op,pred_all0_cause_op], feed_dict=dict(zip(placeholders, test)))
                print('\nepoch {}: test loss {:.4f} cost time: {:.1f}s\n'.format(i, loss, time.time() - start_time))

                # 处理全0的数据集标签，用以计算PRF值----开始
                pred_sum_cause = []
                pred_sum_pos = []
                for r in range(len(te_doc_id)):
                    pred_sum_cause_tmp = tf.reduce_sum(pred_y_cause[r, :te_doc_len[r]])  # 得到该文本的标签值总和
                    pred_sum_cause.append(pred_sum_cause_tmp)
                    pred_sum_pos_tmp = tf.reduce_sum(pred_y_pos[r, :te_doc_len[r]])  # 得到该文本的标签值总和
                    pred_sum_pos.append(pred_sum_pos_tmp)
                pred_sum_cause, pred_sum_pos = sess.run([pred_sum_cause, pred_sum_pos])

                pred_all0_pos_index = [k for (k, v) in enumerate(pred_sum_pos) if v == 0]  # 得到情感子句预测为全0的文本索引
                pred_all0_cause_index = [k for (k, v) in enumerate(pred_sum_cause) if v == 0]  # 得到原因子句预测为全0的文本索引

                for r in pred_all0_pos_index:
                    all0_pred_pos = pred_pos[r][:te_doc_len[r], 1]  # 得到全0文本中是情感子句的预测概率
                    all0_max_pos_index = all0_pred_pos.argsort()[-2:]  ## 得到该文本中情感子句概率前2的子句索引号

                    pred_y_pos[r][all0_max_pos_index] = 1  # 将情感子句概率最大的子句标签改为1
                    # print('pred_y_pos[r]',pred_y_pos[r])  # 修改完标签后该文本的标签内容
                for r in pred_all0_cause_index:
                    all0_pred_cause = pred_cause[r][:te_doc_len[r], 1]
                    all0_max_cause_index = all0_pred_cause.argsort()[-2:]  ## 得到该文本中原因子句概率前2的子句索引号

                    pred_y_cause[r][all0_max_cause_index] = 1
                    # print('pred_y_cause[r]',r,pred_y_cause[r])  # 修改完标签后该文本的标签内容
                # 处理全0的数据集标签----完成

                acc, p, r, f1 = acc_prf(pred_y_cause, true_y_cause, doc_len_batch)
                result_avg_cause = [acc, p, r, f1]
                if f1 > max_f1_cause:
                    max_acc_cause, max_p_cause, max_r_cause, max_f1_cause = acc, p, r, f1
                print('cause_predict: test acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))
                epoch_cause.append(f1)
                print('max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(max_acc_cause, max_p_cause,
                                                                                        max_r_cause, max_f1_cause))

                acc, p, r, f1 = acc_prf(pred_y_pos, true_y_pos, doc_len_batch)
                result_avg_pos = [acc, p, r, f1]
                if f1 > max_f1_pos:
                    max_acc_pos, max_p_pos, max_r_pos, max_f1_pos = acc, p, r, f1
                print('position_predict: test acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}'.format(acc, p, r, f1))
                epoch_emo.append(f1)
                print('max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(max_acc_pos, max_p_pos, max_r_pos,
                                                                                      max_f1_pos))

                if (result_avg_cause[-1] + result_avg_pos[-1]) / 2. > max_f1_avg:
                    max_f1_avg = (result_avg_cause[-1] + result_avg_pos[-1]) / 2.
                    result_avg_cause_max = result_avg_cause
                    result_avg_pos_max = result_avg_pos

                    te_pred_y_cause, te_pred_y_pos = pred_y_cause, pred_y_pos
                    tr_pred_y_cause, tr_pred_y_pos = [], []
                    for train, _ in get_batch_data(tr_x, tr_sen_len, tr_doc_len, 1., 1., tr_y_position, tr_y_cause, 200,
                                                   test=True):
                        pred_y_cause, pred_y_pos = sess.run([pred_y_cause_op, pred_y_pos_op],
                                                            feed_dict=dict(zip(placeholders, train)))
                        tr_pred_y_cause.extend(list(pred_y_cause))
                        tr_pred_y_pos.extend(list(pred_y_pos))
                print('Average max cause: max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}'.format(result_avg_cause_max[0], result_avg_cause_max[1], result_avg_cause_max[2], result_avg_cause_max[3]))
                print('Average max pos: max_acc {:.4f} max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(result_avg_pos_max[0], result_avg_pos_max[1], result_avg_pos_max[2], result_avg_pos_max[3]))


            def get_pair_data(file_name, doc_id, doc_len, y_pairs, pred_y_cause, pred_y_pos, pred_cause, pred_pos, x, sen_len, word_idx_rev):  #1#

                g = open(file_name, 'w', encoding='utf-8')
                for i in range(len(doc_id)):
                    g.write(doc_id[i] + ' ' + str(doc_len[i]) + '\n')
                    g.write(str(y_pairs[i]) + '\n')
                    for j in range(doc_len[i]):
                        clause = ''
                        for k in range(sen_len[i][j]):
                            clause = clause + word_idx_rev[x[i][j][k]] + ' '
                        g.write(str(j + 1) + ', ' + str(pred_y_pos[i][j]) + ', ' + str(pred_y_cause[i][j]) + ', ' + str(pred_pos[i][j][1]) + ', ' + str(pred_cause[i][j][1]) + ', ' + clause + '\n')  #1#

                print('write {} done'.format(file_name))

            get_pair_data(save_dir + test_file_name, te_doc_id, te_doc_len, te_y_pairs, te_pred_y_cause, te_pred_y_pos,pred_cause, pred_pos, te_x, te_sen_len, word_idx_rev)  # 1#
            get_pair_data(save_dir + train_file_name, tr_doc_id, tr_doc_len, tr_y_pairs, tr_pred_y_cause, tr_pred_y_pos,tr_y_cause, tr_y_position, tr_x, tr_sen_len, word_idx_rev)  # 1#

            print('Optimization Finished!\n')
            print('epoch_loss: ', epoch_loss)
            print('epoch_cause: ', epoch_cause)
            print('epoch_emo: ', epoch_emo)
            print('############# fold {} end ###############'.format(fold))
            # fold += 1
            acc_cause_list.append(result_avg_cause_max[0])
            p_cause_list.append(result_avg_cause_max[1])
            r_cause_list.append(result_avg_cause_max[2])
            f1_cause_list.append(result_avg_cause_max[3])
            acc_pos_list.append(result_avg_pos_max[0])
            p_pos_list.append(result_avg_pos_max[1])
            r_pos_list.append(result_avg_pos_max[2])
            f1_pos_list.append(result_avg_pos_max[3])

        print_training_info()
        all_results = [acc_cause_list, p_cause_list, r_cause_list, f1_cause_list, acc_pos_list, p_pos_list, r_pos_list,f1_pos_list]
        acc_cause, p_cause, r_cause, f1_cause, acc_pos, p_pos, r_pos, f1_pos = map(lambda x: np.array(x).mean(),
                                                                                   all_results)
        print('\ncause_predict: test f1 in 10 fold: {}'.format(np.array(f1_cause_list).reshape(-1, 1)))
        print('average : acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}\n'.format(acc_cause, p_cause, r_cause, f1_cause))
        print('position_predict: test f1 in 10 fold: {}'.format(np.array(f1_pos_list).reshape(-1, 1)))
        print('average : acc {:.4f} p {:.4f} r {:.4f} f1 {:.4f}\n'.format(acc_pos, p_pos, r_pos, f1_pos))
        print_time()


def main(_):
    FLAGS.scope = 'P_emotion_Bert_1'
    run()

    FLAGS.scope = 'P_emotion_Bert_2'
    run()


if __name__ == '__main__':
    tf.app.run()