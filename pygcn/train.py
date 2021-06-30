# coding：utf-8

from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN


# Training settings
# 创建解析器
parser = argparse.ArgumentParser()
# 添加程序参数信息
# action： 当参数在命令行中出现时使用的动作基本类型。
# default：当参数未在命令行中出现时使用的值。
# help：   一个此选项作用的简单描述。
# cuda fastmode seed epochs lr weight hidden dropout
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0008,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

# 通过parse_args()解析参数
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# 每次生成随机数相同
np.random.seed(args.seed)
# 为cpu设置随机数种子 以使得结果是确定的
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data 邻接矩阵 特征矩阵 标签 训练集 校验集 测试集
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            embed_size=300,
            nhid=args.hidden,
            nclass=300,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    """
    input:(N,C) C=类别个数或在例子2D-loss中的(N,C,H,W)
    target: (N) 大小为0 <= targets[i] <= C-1
    """
    #  MultiLabelMarginLoss用于一个样本属于多个类别的分类任务
    # MultiLbaleSoftMarginLoss用于多分类 但每个样本只属于一个类
    criterion = torch.nn.MultiLabelMarginLoss()
    loss_train = criterion(output, labels)
    # loss_train = F.nll_loss(output, labels)
    # acc_train = accuracy(output, labels)
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          # 'acc_train: {:.4f}'.format(acc_train.item()),
          # 'loss_val: {:.4f}'.format(loss_val.item()),
          # 'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return output

def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    output = (train(epoch)).detach().numpy().tolist()

if epoch == 1999:
    file2 = open("../data/youtube2text/youtubeb2text_word_GCN_result_2_epoch2000.txt", 'w+')
    for i in range(len(output)):
        for j in range(len(output[i])):
            file2.write(str(output[i][j]))  # write函数不能写int类型的参数，所以使用str()转化
            file2.write('\t')  # 相当于Tab一下，换一个单元格
        file2.write('\n')  # 写完一行立马换行
    file2.close()

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


# Testing
# test()
