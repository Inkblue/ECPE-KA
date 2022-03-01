#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf
import os

#读入词嵌入
def load_w2v(w2v_file, embedding_dim, debug=False):
    fp = open(w2v_file)
    words, _ = map(int, fp.readline().split())  # 单个独立下划线是用作一个名字，来表示某个变量是临时的或无关紧要的 或 最近一个表达式的结果

    w2v = []
    # [0,0,...,0] represent absent words
    w2v.append([0.] * embedding_dim) # 初始化w2v，使其维度为embedding_dim，初始值为[0.]
    word_dict = dict()
    print('load word_embedding...')
    print('word: {} embedding_dim: {}'.format(words, embedding_dim))
    cnt = 0
    for line in fp:
        cnt += 1  # 计数，文件的行数
        line = line.split()  # 空格切分
        if len(line) != embedding_dim + 1:  # 判断该行维度
            print('a bad word embedding: {}'.format(line[0]))
            continue
        word_dict[line[0]] = cnt  # line[0]表示该行表示的word，将word及其对应的向量所在行存入字典中
        w2v.append([float(v) for v in line[1:]])  # 把向量存入w2v中
    print('done!')
    w2v = np.asarray(w2v, dtype=np.float32)  # asarray将数据类型转换为ndarray类型，并且不会复制原数据w2v
    #w2v -= np.mean(w2v, axis = 0) # zero-center
    #w2v /= np.std(w2v, axis = 0)
    if debug:
        print('shape of w2v:',np.shape(w2v))
        word='the'
        print('id of \''+word+'\':',word_dict[word])
        print('vector of \''+word+'\':',w2v[word_dict[word]])
    return word_dict, w2v

#用于生成minibatch训练数据
def batch_index(length, batch_size, test=False):
    #index = range(length)
    index = list(range(length))
    if not test: np.random.shuffle(index)  # 如果不是测试数据集，打乱数据集索引
    #for i in xrange(int( (length + batch_size -1) / batch_size ) ):
    for i in range(int((length + batch_size - 1) / batch_size)):
        ret = index[i * batch_size : (i + 1) * batch_size]  # 生成一个batch_size大小的数据集
        if not test and len(ret) < batch_size : break
        yield ret  # yield关键字很像return，所不同的是，它返回的是一个生成器。然后再用for去遍历这个生成器。

# tf functions
class Saver(object):
    def __init__(self, sess, save_dir, max_to_keep=10):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.sess = sess
        self.save_dir = save_dir
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=max_to_keep)
        '''
        Saver类:保存和恢复变量
        write_version：控制保存检查点时使用的格式。它还会影响某些文件路径匹配逻辑。V2格式是推荐的选择：它在恢复期间所需的内存和延迟方面比V1更加优化。无论此标志如何，Saver都能够从V2和V1检查点恢复。
        max_to_keep表示要保留的最近文件的最大数量。创建新文件时，将删除旧文件。如果为None或0，则不会从文件系统中删除任何Checkpoint，但只有最后一个Checkpoint保留在checkpoint文件中。默认为5
        '''

    def save(self, step):  # 保存训练好的模型
        self.saver.save(self.sess, self.save_dir, global_step=step)  # save_dir保存的路径和名字,global_step将训练的次数作为后缀加入到模型名字中

    def restore(self, idx=''):  # 恢复变量
        ckpt = tf.train.get_checkpoint_state(self.save_dir)
        model_path = self.save_dir+idx if idx else ckpt.model_checkpoint_path # 'dir/-110'
        print("Reading model parameters from %s" % model_path)
        self.saver.restore(self.sess, model_path)


def get_weight_varible(name, shape):  # 初始化权重
        return tf.get_variable(name, initializer=tf.random_uniform(shape, -0.01, 0.01))  # 从均匀分布中输出随机值

def tf_load_w2v(w2v_file, embedding_dim, embedding_type):
    print('\n\n>>>>>>>>>>>>>>>>>>>>MODEL INFO:\n\n## embedding parameters ##')
    print('w2v_file-{}'.format(w2v_file))
    word_id_mapping, w2v = load_w2v(w2v_file, embedding_dim)
    print('embedding_type-{}\n'.format(embedding_type))
    if embedding_type == 0:  # Pretrained and Untrainable
        word_embedding = tf.constant(w2v, dtype=tf.float32, name='word_embedding')  # constant创建常量，将
    elif embedding_type == 1:  # Pretrained and Trainable
        word_embedding = tf.Variable(w2v, dtype=tf.float32, name='word_embedding')
    elif embedding_type == 2:  # Random and Trainable
        word_embedding = get_weight_varible(shape=w2v.shape, name='word_embedding')
    return word_id_mapping, word_embedding

# def tf_load_w2v(w2v_file, embedding_dim, embedding_type):
#     print('\n\n>>>>>>>>>>>>>>>>>>>>MODEL INFO:\n\n## embedding parameters ##')
#     print('w2v_file-{}'.format(w2v_file))
#     word_id_mapping, w2v = load_w2v(w2v_file, embedding_dim)
#     print('embedding_type-{}\n'.format(embedding_type))
#     if embedding_type == 0:  # Pretrained and Untrainable
#         return word_id_mapping, tf.constant(w2v, dtype=tf.float32, name='word_embedding')
#     w2v = w2v[1:]
#     if embedding_type == 1:  # Pretrained and Trainable
#         word_embedding = tf.Variable(w2v, dtype=tf.float32, name='word_embedding')
#     else:  # Random and Trainable
#         word_embedding = get_weight_varible(shape=w2v.shape, name='word_embedding')
#     embed0 = tf.Variable(np.zeros([1, embedding_dim]), dtype=tf.float32, name="embed0", trainable=False)
#     return word_id_mapping, tf.concat((embed0, word_embedding), 0) 

def getmask(length, max_len, out_shape):
    ''' 
    length shape:[batch_size]
    '''
    ret = tf.cast(tf.sequence_mask(length, max_len), tf.float32) # cast数据类型转换
    '''
    sequence_mask数据填充
    mask_data = tf.sequence_mask(lengths=4, maxlen=6)
    # 输出结果,输出结果是长度为6的array，前四个True
    array([True, True, True, True, False, False])
    '''
    return tf.reshape(ret, out_shape)

#实际运行比biLSTM更快
def biLSTM_multigpu(inputs,length,n_hidden,scope):
    ''' 
    input shape:[batch_size, max_len, embedding_dim]
    length shape:[batch_size]
    return shape:[batch_size, max_len, n_hidden*2]
    '''
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=tf.contrib.rnn.LSTMCell(n_hidden),  # 前向RNN
        cell_bw=tf.contrib.rnn.LSTMCell(n_hidden),  # 后向RNN
        inputs=inputs,  # 输入
        # sequence_length=length,  # 输入序列的实际长度
        dtype=tf.float32,  # 初始化和输出的数据类型
        scope=scope
    )
    
    max_len = tf.shape(inputs)[1]
    mask = getmask(length, max_len, [-1, max_len, 1])
    return tf.concat(outputs, 2) * mask

def LSTM_multigpu(inputs,length,n_hidden,scope):
    ''' 
    input shape:[batch_size, max_len, embedding_dim]
    length shape:[batch_size]
    return shape:[batch_size, max_len, n_hidden*2]
    '''
    outputs, state = tf.nn.dynamic_rnn(
        cell=tf.contrib.rnn.LSTMCell(n_hidden),
        inputs=inputs,
        # sequence_length=length,
        dtype=tf.float32,
        scope=scope
    )
    
    max_len = tf.shape(inputs)[1]
    mask = getmask(length, max_len, [-1, max_len, 1])
    return outputs * mask

def biLSTM_multigpu_last(inputs,length,n_hidden,scope):
    ''' 
    input shape:[batch_size, max_len, embedding_dim]
    length shape:[batch_size]
    return shape:[batch_size, max_len, n_hidden*2]
    '''
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=tf.contrib.rnn.LSTMCell(n_hidden),
        cell_bw=tf.contrib.rnn.LSTMCell(n_hidden),
        inputs=inputs,
        # sequence_length=length,
        dtype=tf.float32,
        scope=scope
    )

    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]

    index = tf.range(0, batch_size) * max_len + tf.maximum((length - 1), 0)
    fw_last = tf.gather(tf.reshape(outputs[0], [-1, n_hidden]), index)  # batch_size * n_hidden
    index = tf.range(0, batch_size) * max_len 
    bw_last = tf.gather(tf.reshape(outputs[1], [-1, n_hidden]), index)  # batch_size * n_hidden
    
    return tf.concat([fw_last, bw_last], 1)

   


def biLSTM(inputs,length,n_hidden,scope):
    ''' 
    input shape:[batch_size, max_len, embedding_dim]
    length shape:[batch_size]
    return shape:[batch_size, max_len, n_hidden*2]
    '''
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=tf.contrib.rnn.LSTMCell(n_hidden),
        cell_bw=tf.contrib.rnn.LSTMCell(n_hidden),
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32,
        scope=scope
    )
    
    return tf.concat(outputs, 2)

def LSTM(inputs,sequence_length,n_hidden,scope):
    outputs, state = tf.nn.dynamic_rnn(
        cell=tf.contrib.rnn.LSTMCell(n_hidden),
        inputs=inputs,
        sequence_length=sequence_length,
        dtype=tf.float32,
        scope=scope
    )
    return outputs

def att_avg(inputs, length):
    ''' 
    input shape:[batch_size, max_len, n_hidden]
    length shape:[batch_size]
    return shape:[batch_size, n_hidden]
    '''
    max_len = tf.shape(inputs)[1]
    inputs *= getmask(length, max_len, [-1, max_len, 1])
    inputs = tf.reduce_sum(inputs, 1, keepdims =False)  # 计算张量沿着某一维度的和。keepdims是否保持原有张量的维度,设置为False，结果会降低维度
    length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9  # 把length重组为若干个1维的向量
    return inputs / length





def softmax_by_length(inputs, length):
    ''' 
    input shape:[batch_size, 1, max_len]
    length shape:[batch_size]
    return shape:[batch_size, 1, max_len]
    '''
    inputs = tf.exp(tf.cast(inputs, tf.float32))
    inputs *= getmask(length, tf.shape(inputs)[2], tf.shape(inputs))
    _sum = tf.reduce_sum(inputs, reduction_indices=2, keepdims =True) + 1e-9
    return inputs / _sum

def att_var(inputs,length,w1,b1,w2):
    ''' 
    input shape:[batch_size, max_len, n_hidden]
    length shape:[batch_size]
    return shape:[batch_size, n_hidden]
    '''
    max_len, n_hidden = (tf.shape(inputs)[1], tf.shape(inputs)[2])
    tmp = tf.reshape(inputs, [-1, n_hidden])
    u = tf.tanh(tf.matmul(tmp, w1) + b1)
    alpha = tf.reshape(tf.matmul(u, w2), [-1, 1, max_len])
    alpha = softmax_by_length(alpha, length)
    return tf.reshape(tf.matmul(alpha, inputs), [-1, n_hidden]) 

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        # Average over the 'tower' dimension.
        grad = tf.stack(grads, 0)  # 将秩为R的张量列表堆叠成一个秩为(R+1)的张量
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
