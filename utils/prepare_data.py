# encoding:utf-8

import codecs
import random
import numpy as np
import pickle as pk
from sklearn.metrics import precision_score, recall_score, f1_score
import pdb, time
from bert_serving.client import BertClient
from fnmatch import fnmatch  #通配符匹配

def print_time():
    print ('\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))

def load_w2v(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
    print('\nload embedding...')
    # 把每个词列出索引并排序
    words = []
    inputFile1 = open(train_file_path, 'r', encoding='utf-8')
    for line in inputFile1.readlines():  # 读取文件中的每一行
        line = line.strip().split(',')  # 移除每一行中的空格，并以逗号划分
        emotion, clause = line[2], line[-1]
        ##words.extend( [emotion] + clause.split())  # extend在列表末尾一次性追加另一个序列中的多个值
        words.extend(clause.split())  ## extend在列表末尾一次性追加另一个序列中的多个值
    words = set(words)  # 所有不重复词的集合
    word_idx = dict((c, k + 1) for k, c in enumerate(words)) # 每个词及词的位置，k表示词的索引(从0开始)，c表示词本身
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words))
    # enumerate将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标

    # 给每个词创建向量
    w2v = {}
    inputFile2 = open(embedding_path, 'r',  encoding='utf-8')
    inputFile2.readline()
    for line in inputFile2.readlines():
        line = line.strip().split(' ')
        w, ebd = line[0], line[1:]  # w保存词，ebd保存其向量表示
        w2v[w] = ebd

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0  # 记录在向量表中存在的单词个数
    for item in words:  # 取词
        if item in w2v:  # 取该词的向量表示
            vec = list(map(float, w2v[item]))  # map(float, w2v[item])将向量值转为float型  vec为list列表类型
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1) # 从均匀分布[-0.1,0.1]中随机取
        embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))

    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend( [list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(200)] )
    ##embedding_pos.extend([list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(768)])   ##
    '''
    np.random.normal正态分布
    loc均值为0，以Y轴为对称轴
    scale标准差为0.1，越小曲线越高瘦
    size(int 或者整数元组)输出的值赋在embedding_dim_pos里
    '''

    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)   ## embedding_pos维度是201*50
    
    print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")
    return word_idx_rev, word_idx, embedding, embedding_pos

def load_bert_embedding(embedding_dim, embedding_dim_pos, train_file_path):
    print('\nload bert embedding...')
    # 把每个词列出索引并排序
    words = []
    inputFile1 = open(train_file_path, 'r', encoding='utf-8')
    for line in inputFile1.readlines():  # 读取文件中的每一行
        line = line.strip().split(',')  # 移除每一行中的空格，并以逗号划分
        emotion, clause = line[2], line[-1]
        ##words.extend( [emotion] + clause.split())  # extend在列表末尾一次性追加另一个序列中的多个值
        words.extend(clause.split())  ## extend在列表末尾一次性追加另一个序列中的多个值
    words = set(words)  # 所有不重复词的集合
    word_idx = dict((c, k + 1) for k, c in enumerate(words))  # 每个词及词的位置，k表示词的索引(从0开始)，c表示词本身
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words))
    # enumerate将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标

    # 给每个词创建向量
    w = list(words)
    bc = BertClient()
    ebd = bc.encode(w)  # 获取词的向量表示
    ebd = list(ebd)

    bert_ebd = {}
    bert_ebd = dict(zip(w, ebd))# w保存词，ebd保存其向量表示

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0  # 记录在向量表中存在的单词个数
    for item in words:  # 取词
        if item in bert_ebd:  # 取该词的向量表示
            vec = list(map(float, bert_ebd[item]))  # map(float, w2v[item])将向量值转为float型  vec为list列表类型
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1)  # 从均匀分布[-0.1,0.1]中随机取
        embedding.append(vec)
    print('all_words: {} hit_words: {}'.format(len(words), hit))

    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend([list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(200)])
    '''
    np.random.normal正态分布
    loc均值为0，以Y轴为对称轴
    scale标准差为0.1，越小曲线越高瘦
    size(int 或者整数元组)输出的值赋在embedding_dim_pos里
    '''
    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)

    print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")
    return word_idx_rev, word_idx, embedding, embedding_pos

def load_data(input_file, word_idx, max_doc_len = 75, max_sen_len = 45):
    print('load data_file: {}'.format(input_file))
    y_position, y_cause, y_pairs, x, sen_len, doc_len = [], [], [], [], [], []
    doc_id = []
    
    n_cut = 0
    inputFile = open(input_file, 'r',  encoding='utf-8')
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id.append(line[0])  # 文本的索引
        d_len = int(line[1])  # 该文本句子个数
        pairs = eval('[' + inputFile.readline().strip() + ']')  # 返回  情感-原因对，句子的序号 [(1, 4), (2, 5), (3, 6)]
        doc_len.append(d_len)
        y_pairs.append(pairs)
        pos, cause = zip(*pairs)  # 解压pairs，将情感句序号存储在pos，原因句序号存储在cause中[(1, 2, 3), (4, 5, 6)]
        y_po, y_ca, sen_len_tmp, x_tmp = np.zeros((max_doc_len, 2)), np.zeros((max_doc_len, 2)), np.zeros(max_doc_len,dtype=np.int32), np.zeros((max_doc_len, max_sen_len),dtype=np.int32)
        for i in range(d_len):
            y_po[i][int(i+1 in pos)]=1  # 该文本中各句子是否为情感句的标签。标注当前句子是否为情感句。[int(i+1 in pos)]  i+1=pos，则表示该句为情感句，返回1，否则返回0
            y_ca[i][int(i+1 in cause)]=1  # 该文本中各句子是否为原因句的标签。
            words = inputFile.readline().strip().split(',')[-1]  # 存储该句的内容
            sen_len_tmp[i] = min(len(words.split()), max_sen_len)  # 存储当前句子的单词数
            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    n_cut += 1  # 当前句超出最大句子长度的单词个数
                    break
                x_tmp[i][j] = int(word_idx[word])  # 将当前句的每个词存储在x_tmp中
        
        y_position.append(y_po)
        y_cause.append(y_ca)
        x.append(x_tmp)
        sen_len.append(sen_len_tmp)
    
    y_position, y_cause, x, sen_len, doc_len = map(np.array, [y_position, y_cause, x, sen_len, doc_len])  # 将所有变量变成数组
    for var in ['y_position', 'y_cause', 'x', 'sen_len', 'doc_len']:
        print('{}.shape {}'.format( var, eval(var).shape ))
    print('n_cut {}'.format(n_cut))
    print('load data done!\n')
    return doc_id, y_position, y_cause, y_pairs, x, sen_len, doc_len

def bert_word2id(words, max_sen_len_bert, tokenizer, i, x_tmp, sen_len_tmp):
    # 首先转换成unicode
    tokens_a, ret = tokenizer.tokenize(words), 0
    if len(tokens_a) > max_sen_len_bert - 2:
        ret += 1
        tokens_a = tokens_a[0:(max_sen_len_bert - 2)]
    tokens_a = ["[CLS]"] + tokens_a + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens_a)
    sen_len_tmp[i] = len(input_ids)
    for j in range(len(input_ids)):
        x_tmp[i][j] = input_ids[j]
    return ret

def load_data_bert(input_file, tokenizer, word_idx, max_doc_len, max_sen_len_bert, max_sen_len):
    print('load data_file: {}'.format(input_file))
    doc_id, y_emotion, y_cause, y_pairs, x_bert, sen_len_bert, x, sen_len, doc_len = [[] for i in range(9)]

    n_cut = 0
    inputFile = open(input_file, 'r', encoding='utf-8')
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id.append(line[0])
        d_len = int(line[1])
        pairs = eval('[' + inputFile.readline().strip() + ']')
        doc_len.append(d_len)
        y_pairs.append(pairs)
        emo, cause = zip(*pairs)
        y_emotion_tmp, y_cause_tmp = np.zeros((max_doc_len, 2)), np.zeros((max_doc_len, 2))
        x_bert_tmp, sen_len_bert_tmp = np.zeros((max_doc_len, max_sen_len_bert), dtype=np.int32), np.zeros(max_doc_len,dtype=np.int32)
        x_tmp, sen_len_tmp = np.zeros((max_doc_len, max_sen_len), dtype=np.int32), np.zeros(max_doc_len, dtype=np.int32)
        for i in range(d_len):
            y_emotion_tmp[i][int(i + 1 in emo)] = 1
            y_cause_tmp[i][int(i + 1 in cause)] = 1
            words = inputFile.readline().strip().split(',')[-1]
            n_cut += bert_word2id(words, max_sen_len_bert, tokenizer, i, x_bert_tmp, sen_len_bert_tmp)
            # pdb.set_trace()
            sen_len_tmp[i] = min(len(words.split()), max_sen_len)
            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    break
                x_tmp[i][j] = int(word_idx[word])

        y_emotion.append(y_emotion_tmp)
        y_cause.append(y_cause_tmp)
        x_bert.append(x_bert_tmp)
        sen_len_bert.append(sen_len_bert_tmp)
        x.append(x_tmp)
        sen_len.append(sen_len_tmp)

    y_emotion, y_cause, x_bert, sen_len_bert, x, sen_len, doc_len = map(np.array, [y_emotion, y_cause, x_bert, sen_len_bert, x, sen_len, doc_len])  # 将所有变量变成数组
    for var in ['y_emotion', 'y_cause', 'x_bert', 'sen_len_bert', 'x', 'sen_len', 'doc_len']:
        print('{}.shape {}'.format(var, eval(var).shape))
    print('n_cut {}'.format(n_cut))
    print('load data_bert done!\n')
    return doc_id, y_emotion, y_cause, y_pairs, x_bert, sen_len_bert, x, sen_len, doc_len

def load_data_2nd_step(input_file, word_idx, max_doc_len = 75, max_sen_len = 45):
    print('load data_file: {}'.format(input_file))
    pair_id_all, pair_id, y, x, sen_len, distance = [], [], [], [], [], []
    clause_pair_pro = []  ##定义存储子句对概率的数组
    
    n_cut = 0
    inputFile = open(input_file, 'r', encoding='utf-8')
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id = int(line[0])
        d_len = int(line[1])
        pairs = eval(inputFile.readline().strip())
        pair_id_all.extend([doc_id*10000+p[0]*100+p[1] for p in pairs])
        sen_len_tmp, x_tmp = np.zeros(max_doc_len,dtype=np.int32), np.zeros((max_doc_len, max_sen_len),dtype=np.int32)
        pos_list, cause_list = [], []
        pos_clause_pro, cause_clause_pro = [], []  ##定义存储子句概率的数组
        for i in range(d_len):  #读取一个文本中的每个句子
            line = inputFile.readline().strip().split(',')
            if int(line[1].strip())>0:
                pos_list.append(i+1)  # 将情感子句的序号加入pos_list
                pos_clause_pro.append(float(line[3].strip()))  ##将该情感子句的概率值加入pos_clause_pro
            if int(line[2].strip())>0:
                cause_list.append(i+1)  # 将原因子句的序号加入cause_list
                cause_clause_pro.append(float(line[4].strip()))  ##将该原因子句的概率值加入cause_clause_pro
            words = line[-1]  #存储该行句子内容
            sen_len_tmp[i] = min(len(words.split()), max_sen_len)  # 存储该行句子(下标比句子序号小1)的单词数，最大为max_sen_len
            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    n_cut += 1
                    break
                x_tmp[i][j] = int(word_idx[word])  # 在该句第j个单词的位置存储该单词的embedding索引号
        '''笛卡儿积'''
        #for i in pos_list:
            #for j in cause_list:
        for i_num,i in enumerate(pos_list): ##
            for j_num,j in enumerate(cause_list): ##
                pair_id_cur = doc_id*10000+i*100+j
                pair_id.append(pair_id_cur)
                y.append([0,1] if pair_id_cur in pair_id_all else [1,0])  # 给笛卡儿积结果打标，是正确的pari则标为[0,1]，否则为[1,0]
                x.append([x_tmp[i-1],x_tmp[j-1]])  # 存储所有句子对
                clause_pair_pro.append([pos_clause_pro[i_num], cause_clause_pro[j_num]])  ## 存储子句对的概率值
                sen_len.append([sen_len_tmp[i-1], sen_len_tmp[j-1]])  # 存储句子对的长度
                distance.append(j-i+100)  # 存储句子对距离
    y, x, sen_len, distance = map(np.array, [y, x, sen_len, distance])
    clause_pair_pro = np.array(clause_pair_pro)  ##
    for var in ['y', 'x', 'sen_len', 'distance']:
        print('{}.shape {}'.format( var, eval(var).shape ))
    print('n_cut {}, (y-negative, y-positive): {}'.format(n_cut, y.sum(axis=0)))
    print('load data done!\n')
    return pair_id_all, pair_id, y, x, sen_len, distance, clause_pair_pro

def acc_prf(pred_y, true_y, doc_len, average='binary'): 
    tmp1, tmp2 = [], []
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            tmp1.append(pred_y[i][j])
            tmp2.append(true_y[i][j])
    y_pred, y_true = np.array(tmp1), np.array(tmp2)
    acc = precision_score(y_true, y_pred, average='micro')
    p = precision_score(y_true, y_pred, average=average)
    r = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    return acc, p, r, f1

def prf_2nd_step(pair_id_all, pair_id, pred_y, fold = 0, save_dir = ''):
    pair_id_filtered = []
    for i in range(len(pair_id)):
        if pred_y[i]:
            pair_id_filtered.append(pair_id[i])
    def write_log():
        pair_to_y = dict(zip(pair_id, pred_y))
        g = open(save_dir+'pair_log_fold{}.txt'.format(fold), 'w')
        doc_id_b, doc_id_e = pair_id_all[0]/10000, pair_id_all[-1]/10000  # 文档的序号，最小序号为doc_id_b，最大序号为doc_id_e
        idx_1, idx_2 = 0, 0
        for doc_id in range(doc_id_b, doc_id_e+1):  # 依次读取每个文档
            true_pair, pred_pair, pair_y = [], [], []
            line = str(doc_id) + ' '
            while True:
                p_id = pair_id_all[idx_1]
                d, p1, p2 = p_id/10000, p_id%10000/100, p_id%100
                if d != doc_id: break  # 判断读取的句子对是否属于当前文档
                true_pair.append((p1, p2))  # 保存当前文档的句子对
                line += '({}, {}) '.format(p1,p2)
                idx_1 += 1
                if idx_1 == len(pair_id_all): break
            line += '|| '
            while True:
                p_id = pair_id[idx_2]
                d, p1, p2 = p_id/10000, p_id%10000/100, p_id%100
                if d != doc_id: break
                if pred_y[idx_2]:
                    pred_pair.append((p1, p2))
                pair_y.append(pred_y[idx_2])
                line += '({}, {}) {} '.format(p1, p2, pred_y[idx_2])
                idx_2 += 1
                if idx_2 == len(pair_id): break
            if len(true_pair)>1:
                line += 'multipair '
                if true_pair == pred_pair:
                    line += 'good '
            line += '\n'
            g.write(line)
    if fold:
        write_log()
    keep_rate = len(pair_id_filtered)/(len(pair_id)+1e-8)  # 1e-8表示10的-8次方
    s1, s2, s3 = set(pair_id_all), set(pair_id), set(pair_id_filtered)  # set()函数创建一个无序不重复元素集
    o_acc_num = len(s1 & s2)  # 原始所有文本中共有多少个正确的句子对
    acc_num = len(s1 & s3)  # 预测出的文本中共有多少个句子对  TP
    o_p, o_r = o_acc_num/(len(s2)+1e-8), o_acc_num/(len(s1)+1e-8)
    p, r = acc_num/(len(s3)+1e-8), acc_num/(len(s1)+1e-8)
    f1, o_f1 = 2*p*r/(p+r+1e-8), 2*o_p*o_r/(o_p+o_r+1e-8)
    
    return p, r, f1, o_p, o_r, o_f1, keep_rate

def load_LIWC(LIWC_filename, train_file_path, embedding_dim_LIWC):
    inputFile = open(LIWC_filename, 'r', encoding='utf-8')
    inputFile.readline()  # 读取首行的%
    lines = inputFile.readlines()  #读取文件所有行
    type2num = dict()
    word2type = dict()  # 存储词典内一般的词
    word2type_wlid = dict()  # 存储词典内带*的词

    # 读取LIWC词典的种类
    for i, line in enumerate(lines):
        if '%' in line:
            loc_num = i
            break
        tmp = line.strip().split()
        #print(loc_num)  #输出lc=72
        #print('tmp',tmp)  #输出['1', 'funct', '功能词', '或许、许多、那些']
        #type2name[int(tmp[0])] = tmp[1]  #{种类编号:种类名字}，如{1:funct}
        type2num[int(tmp[0])] = i  #{种类编号：embedding索引}，如{1：0}
        #print(type2num)
    #print(type2num.get(139))  # 输出39。即第131种类别对应第39个位置

    # 将LIWC词典中所有词所属类别存为字典
    for line in lines[loc_num + 1:]:
        tmp = line.strip().split()
        #print(type(tmp[0]))  # 输出<class 'str'>
        if '*' in tmp[0]:
            #tmp1 = tmp[0][:-1]  # 去掉*号。不应该去掉*，应该将其保留作为通配符
            word2type_wlid[tmp[0]] = list(map(int, tmp[1:]))
            #print(word2type_wlid)  # 输出{'一千*': [1, 21]}
        else:
            word2type[tmp[0]] = list(map(int, tmp[1:]))
            #print(word2type)  # 输出{'只': [1, 16, 34, 131, 139]}

    # 读取情感原因数据集中所有的单词，去重
    words = []
    inputFile1 = open(train_file_path, 'r', encoding='utf-8')
    for line in inputFile1.readlines():  # 读取文件中的每一行
        line = line.strip().split(',')  # 移除每一行中的空格，并以逗号划分
        emotion, clause = line[2], line[-1]
        ##words.extend( [emotion] + clause.split())  # extend在列表末尾一次性追加另一个序列中的多个值
        words.extend(clause.split())  ## extend在列表末尾一次性追加另一个序列中的多个值
    words = set(words)  # 所有不重复词的集合

    # 对数据集中每个词进行LIWC的embedding
    LIWC_embedding = [list(np.zeros(embedding_dim_LIWC))]
    hit = 0  # 记录在向量表中存在的单词个数
    keys = list(word2type_wlid.keys())  # 存储*号词典的所有词
    for item in words:  # 取词
        vec = list(np.zeros(embedding_dim_LIWC))  # 初始化

        if item in word2type:  # 取非*号词的种类
            kinds = list(map(int, word2type[item]))  # map(float, word2type[item])将向量值转为int型  kinds为list列表类型
            for i, value in enumerate(kinds):
                vec[type2num.get(value)] = 1
            hit += 1
        else:  #*号词或者不存在的词
            for key in keys:
                if fnmatch(item,key):  # 判断是否为*号词
                    kinds = list(map(int, word2type_wlid[key]))  # map(float, word2type_wild[key])将向量值转为int型  kinds为list列表类型
                    for i, value in enumerate(kinds):
                        vec[type2num.get(value)] = 1
                    hit += 1
        # else:
        #     vec = list(np.zeros(embedding_dim_LIWC))  # 若单词不存在于LIWC词典中，则用全0代替
        LIWC_embedding.append(vec)
    LIWC_embedding = np.array(LIWC_embedding)
    print('all_words: {} LIWC_hit_words: {}'.format(len(words), hit))

    return LIWC_embedding

