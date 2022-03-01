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
## model struct ##
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
# >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_string('log_file_name', '', 'name of log file')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_integer('training_iter', 10, 'number of train iter')
tf.app.flags.DEFINE_string('scope', 'Ind_BiLSTM', 'RNN scope')
# not easy to tune , a good posture of using data to train model is very important
tf.app.flags.DEFINE_integer('batch_size', 32, 'number of example per batch')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 0.5, 'word embedding training dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'softmax layer dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')


def build_model(wordPOS_embedding, LIWC_embedding, word_embedding, pos_embedding, clause_pro, x, sen_len, keep_prob1, keep_prob2, distance, y, RNN=biLSTM):
    ##x = tf.nn.embedding_lookup(word_embedding, x)  # 选取一个张量里面索引对应的元素。x为索引
    embedding_tmp = tf.concat([word_embedding,LIWC_embedding],1)  ##
    embedding_all = tf.concat([embedding_tmp, wordPOS_embedding], 1)  ##
    x = tf.nn.embedding_lookup(embedding_all, x)  ##
    ##inputs = tf.reshape(x,[-1, FLAGS.max_sen_len, FLAGS.embedding_dim])  # 将x变成行数不确定，但列数为max_sen_len，深度为embedding_dim的张量
    inputs = tf.reshape(x, [-1, FLAGS.max_sen_len, FLAGS.embedding_dim_all])  ## 将x变成行数不确定，但列数为max_sen_len，深度为embedding_dim_all的张量
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
        s = tf.reshape(s, [-1, 2 * 2 * FLAGS.n_hidden])
        return s

    s = get_s(inputs, name='cause_word_encode')
    dis = tf.nn.embedding_lookup(pos_embedding, distance)
    s = tf.concat([s, dis], 1)
    s = tf.concat([s, clause_pro], 1)  ## 加入子句对的概率值

    s1 = tf.nn.dropout(s, keep_prob=keep_prob2)
    w_pair = get_weight_varible('softmax_w_pair', [4 * FLAGS.n_hidden + FLAGS.embedding_dim_pos +2, FLAGS.n_class])
    b_pair = get_weight_varible('softmax_b_pair', [FLAGS.n_class])
    pred_pair = tf.nn.softmax(tf.matmul(s1, w_pair) + b_pair)

    reg = tf.nn.l2_loss(w_pair) + tf.nn.l2_loss(b_pair)
    return pred_pair, reg


def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, lr-{}, kb1-{}, kb2-{}, l2_reg-{}'.format(
        FLAGS.batch_size, FLAGS.learning_rate, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.l2_reg))
    print('training_iter-{}, scope-{}\n'.format(FLAGS.training_iter, FLAGS.scope))


def get_batch_data(x, sen_len, keep_prob1, keep_prob2, distance, y, batch_size, clause_pro, test=False):
    for index in batch_index(len(y), batch_size, test):
        feed_list = [x[index], sen_len[index], keep_prob1, keep_prob2, distance[index], y[index], clause_pro[index]]
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
    word_idx_rev, word_id_mapping, word_embedding, pos_embedding = load_bert_embedding(FLAGS.embedding_dim_word, FLAGS.embedding_dim_pos,'data_combine/clause_keywords.csv')
    word_embedding = tf.constant(word_embedding, dtype=tf.float32, name='word_embedding')
    pos_embedding = tf.constant(pos_embedding, dtype=tf.float32, name='pos_embedding')

    LIWC_embedding = load_LIWC(FLAGS.SC_LIWC_file, 'data_combine/clause_keywords.csv', FLAGS.embedding_dim_LIWC)  ##
    LIWC_embedding = tf.constant(LIWC_embedding, dtype=tf.float32, name='LIWC_embedding')  ##

    wordPOS_embedding = load_POS(FLAGS.POS_result, 'data_combine/clause_keywords.csv')
    wordPOS_embedding = tf.constant(wordPOS_embedding, dtype=tf.float32, name='wordPOS_embedding')

    print('build model...')

    x = tf.placeholder(tf.int32, [None, 2, FLAGS.max_sen_len])
    sen_len = tf.placeholder(tf.int32, [None, 2])
    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)
    distance = tf.placeholder(tf.int32, [None])
    y = tf.placeholder(tf.float32, [None, FLAGS.n_class])
    clause_pro = tf.placeholder(tf.float32, [None, 2])  ## 定义子句对概率值
    placeholders = [x, sen_len, keep_prob1, keep_prob2, distance, y, clause_pro]  ##
    ##placeholders = [x, sen_len, keep_prob1, keep_prob2, distance, y]

    pred_pair, reg = build_model(wordPOS_embedding, LIWC_embedding, word_embedding, pos_embedding, clause_pro, x, sen_len, keep_prob1, keep_prob2, distance, y)
    loss_op = - tf.reduce_mean(y * tf.log(pred_pair)) + reg * FLAGS.l2_reg
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_op)

    true_y_op = tf.argmax(y, 1)
    pred_y_op = tf.argmax(pred_pair, 1)
    acc_op = tf.reduce_mean(tf.cast(tf.equal(true_y_op, pred_y_op), tf.float32))
    print('build model done!\n')

    # Training Code Block
    print_training_info()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        keep_rate_list, acc_subtask_list, p_pair_list, r_pair_list, f1_pair_list = [], [], [], [], []
        o_p_pair_list, o_r_pair_list, o_f1_pair_list = [], [], []

        for fold in range(1, 11):
            sess.run(tf.global_variables_initializer())
            # train for one fold
            print('############# fold {} begin ###############'.format(fold))
            # Data Code Block
            train_file_name = 'fold{}_train.txt'.format(fold, FLAGS)
            test_file_name = 'fold{}_test.txt'.format(fold)
            tr_pair_id_all, tr_pair_id, tr_y, tr_x, tr_sen_len, tr_distance, tr_clause_pro = load_data_2nd_step(save_dir + train_file_name, word_id_mapping, max_sen_len=FLAGS.max_sen_len)
            te_pair_id_all, te_pair_id, te_y, te_x, te_sen_len, te_distance, te_clause_pro = load_data_2nd_step(save_dir + test_file_name, word_id_mapping, max_sen_len=FLAGS.max_sen_len)

            max_acc_subtask, max_f1 = [-1.] * 2
            epoch_loss = []  # 定义每一个epoch的损失
            epoch_pair = []
            print('train docs: {}    test docs: {}'.format(len(tr_x), len(te_x)))
            for i in range(FLAGS.training_iter):
                start_time, step = time.time(), 1
                # train
                for train, _ in get_batch_data(tr_x, tr_sen_len, FLAGS.keep_prob1, FLAGS.keep_prob2, tr_distance, tr_y,FLAGS.batch_size, tr_clause_pro):
                    _, loss, pred_y, true_y, acc = sess.run(
                        [optimizer, loss_op, pred_y_op, true_y_op, acc_op], feed_dict=dict(zip(placeholders, train)))
                    print('step {}: train loss {:.4f} acc {:.4f}'.format(step, loss, acc))
                    step = step + 1
                epoch_loss.append(loss)
                # test
                test = [te_x, te_sen_len, 1., 1., te_distance, te_y, te_clause_pro]
                loss, pred_y, true_y, acc = sess.run([loss_op, pred_y_op, true_y_op, acc_op],feed_dict=dict(zip(placeholders, test)))
                print('\nepoch {}: test loss {:.4f}, acc {:.4f}, cost time: {:.1f}s\n'.format(i, loss, acc, time.time() - start_time))
                if acc > max_acc_subtask:
                    max_acc_subtask = acc
                print('max_acc_subtask: {:.4f} \n'.format(max_acc_subtask))

                # p, r, f1, o_p, o_r, o_f1, keep_rate = prf_2nd_step(te_pair_id_all, te_pair_id, pred_y, fold, save_dir)
                p, r, f1, o_p, o_r, o_f1, keep_rate = prf_2nd_step(te_pair_id_all, te_pair_id, pred_y)
                if f1 > max_f1:
                    max_keep_rate, max_p, max_r, max_f1 = keep_rate, p, r, f1
                print('original o_p {:.4f} o_r {:.4f} o_f1 {:.4f}'.format(o_p, o_r, o_f1))
                print('pair filter keep rate: {}'.format(keep_rate))
                print('test p {:.4f} r {:.4f} f1 {:.4f}'.format(p, r, f1))
                epoch_pair.append(f1)

                print('max_p {:.4f} max_r {:.4f} max_f1 {:.4f}\n'.format(max_p, max_r, max_f1))

            print('Optimization Finished!\n')
            print('epoch_loss: ', epoch_loss)
            print('epoch_pair: ', epoch_pair)
            print('############# fold {} end ###############'.format(fold))
            # fold += 1
            acc_subtask_list.append(max_acc_subtask)
            keep_rate_list.append(max_keep_rate)
            p_pair_list.append(max_p)
            r_pair_list.append(max_r)
            f1_pair_list.append(max_f1)
            o_p_pair_list.append(o_p)
            o_r_pair_list.append(o_r)
            o_f1_pair_list.append(o_f1)

        print_training_info()
        all_results = [acc_subtask_list, keep_rate_list, p_pair_list, r_pair_list, f1_pair_list, o_p_pair_list,o_r_pair_list, o_f1_pair_list]
        acc_subtask, keep_rate, p_pair, r_pair, f1_pair, o_p_pair, o_r_pair, o_f1_pair = map(lambda x: np.array(x).mean(), all_results)
        print('\nOriginal pair_predict: test f1 in 10 fold: {}'.format(np.array(o_f1_pair_list).reshape(-1, 1)))
        print('average : p {:.4f} r {:.4f} f1 {:.4f}\n'.format(o_p_pair, o_r_pair, o_f1_pair))
        print('\nAverage keep_rate: {:.4f}\n'.format(keep_rate))
        print('\nFiltered pair_predict: test f1 in 10 fold: {}'.format(np.array(f1_pair_list).reshape(-1, 1)))
        print('average : p {:.4f} r {:.4f} f1 {:.4f}\n'.format(p_pair, r_pair, f1_pair))
        print_time()


def main(_):
    FLAGS.training_iter = 20

    # 将控制台结果输出到文件里
    f = open('pair_result.txt', 'w+', encoding='utf-8')
    sys.stdout = f

    for scope_name in ['P_emotion_Bert']:
        FLAGS.scope = scope_name + '_1'
        run()
        FLAGS.scope = scope_name + '_2'
        run()


if __name__ == '__main__':
    tf.app.run()