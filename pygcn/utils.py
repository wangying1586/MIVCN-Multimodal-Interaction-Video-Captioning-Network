import numpy as np
import scipy.sparse as sp
import torch
import pickle

def encode_onehot(labels):
    # set():创建无序不重复的序列 7类
    classes = set(labels)
    # np.identity():创建方阵的one-hot向量 主对角线为1，其余为0数组
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    # (2708,7) 每个paper用one-hot标示类别
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

"""
    2708个paper 7类 有1433个关键词 关键词频小于10会删掉
    core.content:paper_id + 单词是否出现（出现为1，否则0） + 类标签
    core.cites: 被引论文的ID + 引用前面论文ID
"""
def load_data(path="../data/msrvtt/", dataset="msrvtt"):
    print('Loading {} dataset...'.format(dataset))

    x = pickle.load(open("../data/msrvtt/corpus_msr_vtt.p", "rb"), encoding='bytes')
    embedding_vec = x[5]
    del x
    feature = embedding_vec

    labels = pickle.load(open(path + dataset + "_word_labels.p", "rb"), encoding='bytes')

    f = open('../data/youtube2text/key_words.txt')
    key_words = f.read()
    key_words = key_words.split('\n')
    key_words = list(filter(None, key_words))
    x = pickle.load(open('../data/youtube2text/corpus_youtube2text.p', 'rb'), encoding='bytes')
    wordtoix, ixtoword = x[3], x[4]
    idx = []
    for i in range(labels.shape[1]):
        b = key_words[i].encode('utf-8')
        idx.append(wordtoix[b])
    idx_map = {j: i for i, j in enumerate(idx)}
    features = []
    for _, i in enumerate(idx):
        features.append(feature[i])

    x = pickle.load(open("../data/youtube2text/youtube2text_cos_similarity.p", "rb"), encoding='bytes')
    adj = x
    del x

    features = normalize(features)
    adj = normalize(adj)

    idx_train = range(1200)
    idx_val = range(1200, 1300)
    idx_test = range(1300, 1970)

    features = torch.FloatTensor(np.array(features))
    adj = torch.FloatTensor(np.array(adj))
    labels = torch.LongTensor(np.array(labels))

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix 正则化稀疏矩阵"""
    """
        数据标准化的原因：1.数据量纲不同，数量级差别大 2.避免太大的数引发数值问题 3.平衡各特征的贡献 4.加快梯度下降求最优解速度
        方法：
        1.Min-Max：x=(x-min)/(max-min) 对数据每一维度重新调节，使得数据向量在[0,1]区间
        适合数据比较集中，不涉及距离度量、协方差计算、数据不符合正态分布情况，如min和max不稳定，使得归一化结果不稳定
        2.标准差标准化 z-scores 处理后符合正态分布
        3.非线性归一化 log对数函数转换 atan反正切函数转换 L2范数归一化
    """
    mx = np.array(mx)
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor. 矩阵->张量"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
