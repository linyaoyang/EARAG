from vllm import LLM, SamplingParams
from tqdm import tqdm
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial.distance import cdist

seed = 123


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(seed)


def find_row_positions(X, Y):
    positions = []
    for row in Y:
        # Find the index of the row in X
        index = np.where((X == row).all(axis=1))[0]
        if len(index) > 0:
            positions.append(index[0])  # Append the index of the matching row
        else:
            positions.append(None)  # Append None if the row is not found
    return positions


data_dir = 'data/zh_en/'

ill_ents = np.loadtxt(data_dir + 'ref_ent_ids', delimiter='\t', dtype=int)
train_ills = np.load(data_dir + 'train_ills.npy')
test_ills = np.load(data_dir + 'test_ills.npy')
train_indices = find_row_positions(ill_ents, train_ills)
test_indices = find_row_positions(ill_ents, test_ills)
string_sim = np.load(data_dir + 'string_sim.npy')
struct_sim = np.load(data_dir + 'struct_sim_1.npy')
name_emb = np.load(data_dir + 'name_emb.npy')
name_dist = cdist(name_emb[ill_ents[:, 0]], name_emb[ill_ents[:, 1]], 'braycurtis')
name_sim = 1 - name_dist
ngh_sim = np.load(data_dir + 'common_nghs.npy')
print('dy_nb loaded')

top_k = 35

train_feat, train_label = [], []
test_feat, test_label = [], []
overall_sim = struct_sim + name_sim
train_sim, test_sim = overall_sim[train_indices, :][:, train_indices], overall_sim[test_indices, :][:, test_indices]
train_string_sim = string_sim[train_indices, :][:, train_indices]
train_struct_sim = struct_sim[train_indices, :][:, train_indices]
train_name_sim = name_sim[train_indices, :][:, train_indices]
train_ngh_sim = ngh_sim[train_indices, :][:, train_indices]
test_string_sim = string_sim[test_indices, :][:, test_indices]
test_struct_sim = struct_sim[test_indices, :][:, test_indices]
test_name_sim = name_sim[test_indices, :][:, test_indices]
test_ngh_sim = ngh_sim[test_indices, :][:, test_indices]
train_sort, test_sort = np.argsort(-train_sim, axis=1), np.argsort(-test_sim, axis=1)
test_acc, hits10 = 0., 0.
for i in range(len(test_sim)):
    if i == test_sort[i, 0]:
        test_acc += 1
    if i in test_sort[i, :10]:
        hits10 += 1

print('baseline acc: ', test_acc / len(test_sim))
print('baseline hits10: ', hits10 / len(test_sim))

test_struct_sort = np.argsort(-test_struct_sim, axis=1)
test_string_sort = np.argsort(-test_string_sim, axis=1)
test_name_sort = np.argsort(-test_name_sim, axis=1)
test_ngh_sort = np.argsort(-test_ngh_sim, axis=1)
struct_hits1, struct_hits10 = 0., 0.
string_hits1, string_hits10 = 0., 0.
name_hits1, name_hits10 = 0., 0.
ngh_hits1, ngh_hits10 = 0., 0.
for i in range(len(test_sim)):
    if i == test_struct_sort[i, 0]:
        struct_hits1 += 1
    if i in test_struct_sort[i, :10]:
        struct_hits10 += 1
    if i == test_string_sort[i, 0]:
        string_hits1 += 1
    if i in test_string_sort[i, :10]:
        string_hits10 += 1
    if i == test_name_sort[i, 0]:
        name_hits1 += 1
    if i in test_name_sort[i, :10]:
        name_hits10 += 1
    if i == test_ngh_sort[i, 0]:
        ngh_hits1 += 1
    if i in test_ngh_sort[i, :10]:
        ngh_hits10 += 1
print('struct hits1: ', struct_hits1 / len(test_sim))
print('struct hits10: ', struct_hits10 / len(test_sim))
print('string hits1: ', string_hits1 / len(test_sim))
print('string hits10: ', string_hits10 / len(test_sim))
print('name hits1: ', name_hits1 / len(test_sim))
print('name hits10: ', name_hits10 / len(test_sim))
print('ngh hits1: ', ngh_hits1 / len(test_sim))
print('ngh hits10: ', ngh_hits10 / len(test_sim))

print('prepared')

for i in range(len(train_sim)):
    indices = train_sort[i, :top_k]
    feats = []
    if i not in indices:
        indices[-1] = i
    for j in range(len(indices)):
        feats.append([train_struct_sim[i, indices[j]], train_name_sim[i, indices[j]], train_ngh_sim[i, indices[j]]])
        if indices[j] == i:
            train_label.append(j)
    feats = np.array(feats).T
    # train_feat = train_feat.append({'dy_nb': feats}, ignore_index=True)
    train_feat.append(feats)

test_cand_indices = []

for i in range(len(test_sim)):
    indices = test_sort[i, :top_k]
    # 将indices转为list
    indices = indices.tolist()
    # # 将i添加到indices中
    # if i in indices:
    #     indices.remove(i)
    #     indices.insert(0, i)
    # else:
    #     indices[0] = i
    feats = []
    for j in range(len(indices)):
        feats.append([test_struct_sim[i, indices[j]], test_name_sim[i, indices[j]], test_ngh_sim[i, indices[j]]])
    if i in indices:
        test_label.append(indices.index(i))
    else:
        test_label.append(-1)
    feats = np.array(feats).T
    test_cand_indices.append(indices)
    # test_feat = test_feat.append({'dy_nb': feats}, ignore_index=True)
    test_feat.append(feats)

train_feat = np.array(train_feat)
test_feat = np.array(test_feat)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
train_feat = torch.FloatTensor(train_feat).to(device)
train_label = torch.LongTensor(train_label).to(device)
test_feat = torch.FloatTensor(test_feat).to(device)
test_label = torch.LongTensor(test_label).to(device)
train_feat = train_feat.unsqueeze(1)


# 定义CNN模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # 第一个卷积层: in_channels=1 (灰度图)，out_channels=16，卷积核大小 (4x1)
        # 这个卷积核会将输入的 4x10 特征高度缩减到 1，保持宽度 10 不变
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 1), stride=1)

        # 第二个卷积层: 保持输出宽度不变，进一步提取特征，使用 kernel_size=(1,1) 只是用来增加特征通道数
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), stride=1)

        # 全连接层，将卷积结果映射到最终的 1x10 的隐特征
        self.fc1 = nn.Linear(32 * top_k, top_k)  # 将卷积后的特征展平成一维，然后映射到 1x10 输出

    def forward(self, x):
        # 卷积层1: 输入 (batch_size, 1, 4, 10) -> 输出 (batch_size, 16, 1, 10)
        x = F.relu(self.conv1(x))

        # 卷积层2: 输入 (batch_size, 16, 1, 10) -> 输出 (batch_size, 32, 1, 10)
        x = F.relu(self.conv2(x))

        # 展平，将输出的每个样本展平成一个向量， (batch_size, 32 * 10)
        x = x.view(x.size(0), -1)

        # 全连接层: 将展平的特征映射到 1x10 的输出 (batch_size, 10)
        x = self.fc1(x)

        return x


dataset = TensorDataset(train_feat, train_label)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

# 创建模型
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()  # 定义损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 定义优化器
predicted_right_idx, predicted_right_logit = [], []

# 训练模型
epochs = 1000
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, target in dataloader:
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, target)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计损失
        running_loss += loss.item()
        train_input = inputs
        model.eval()  # 设置模型为评估模式
        correct = 0
        total = 0

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}")
    if (epoch + 1) % 10 == 0:
        test_input = test_feat.unsqueeze(1)

        model.eval()  # 设置模型为评估模式
        correct = 0
        total = 0
        with torch.no_grad():  # 在测试时不需要计算梯度
            outputs = model(test_input)
            equal_logits = F.softmax(outputs, dim=1)

            for i in range(len(equal_logits)):
                predicted_right_logit.append(torch.max(equal_logits[i]).item())
                predicted_right_idx.append(test_cand_indices[i][torch.argmax(equal_logits[i]).item()])
                if torch.argmax(equal_logits[i]) != 0:
                    print('the prediction output is not index', i)
            _, predicted = torch.max(outputs, 1)  # 获取每个样本的预测标签
            total += test_label.size(0)
            correct += (predicted == test_label).sum().item()  # 计算正确预测的数量
            equal_logits = equal_logits.cpu().numpy()
            # 将预测结果保存到npy文件中
            np.save(data_dir + 'equal_logits.npy', equal_logits)

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy * 100:.2f}%')

import os
import numpy as np
import logging
import time
import random
from datetime import datetime
from module.utils import load_dwynb_ents, load_rels


# 配置日志记录器
def setup_logger():
    # 获取当前时间作为文件名的一部分
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # 创建日志记录器Z
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG，以便记录所有级别的日志

    # 创建控制台处理器，并设置日志级别为INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建文件处理器，将日志输出到文件
    log_file = f"log_{current_time}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # 文件中记录DEBUG及以上级别的日志

    # 创建日志格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 将格式器添加到处理器
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def remove_random_element(x):
    if not x:  # Check if the list is empty
        return x
    elif len(x) == 1:  # Check if the list has only one element
        return []
    else:
        # Remove a random element
        element_to_remove = random.choice(x)
        x.remove(element_to_remove)
        return x


idx2ent, ents1, ents2 = load_dwynb_ents(data_dir)
idx2rel = load_rels(data_dir)
ill_ents = np.loadtxt(data_dir + 'ref_ent_ids', delimiter='\t', dtype=int)
triples_1 = np.loadtxt(data_dir + 'triples_1', delimiter='\t', dtype=int)
triples_2 = np.loadtxt(data_dir + 'triples_2', delimiter='\t', dtype=int)
np.random.shuffle(ill_ents)

ill_ents_dict1 = {ill_ents[i, 0]: ill_ents[i, 1] for i in range(len(ill_ents))}
ill_ents_dict2 = {ill_ents[i, 1]: ill_ents[i, 0] for i in range(len(ill_ents))}

idx2nghs1, idx2nghs2 = {}, {}
for i in range(len(triples_1)):
    h, r, t = triples_1[i]
    if h not in idx2nghs1:
        idx2nghs1[h] = []
    if t in ill_ents_dict1:
        idx2nghs1[h].append(t)
    if t not in idx2nghs1:
        idx2nghs1[t] = []
    if h in ill_ents_dict1:
        idx2nghs1[t].append(h)

for i in range(len(triples_2)):
    h, r, t = triples_2[i]
    if h not in idx2nghs2:
        idx2nghs2[h] = []
    if t in ill_ents_dict2:
        idx2nghs2[h].append(t)
    if t not in idx2nghs2:
        idx2nghs2[t] = []
    if h in ill_ents_dict2:
        idx2nghs2[t].append(h)

idx2nghs = {}
for e1 in idx2nghs1:
    if e1 in ill_ents_dict1:
        nghs1 = set([e for e in idx2nghs1[e1]])
        if len(nghs1) > 0:
            nghs2 = set([ill_ents_dict2[e] for e in idx2nghs2[ill_ents_dict1[e1]]])
            remain_nghs = nghs1.intersection(nghs2)
            idx2nghs[e1] = list(remain_nghs)

for e2 in idx2nghs2:
    if e2 in ill_ents_dict2:
        nghs2 = set([e for e in idx2nghs2[e2]])
        if len(nghs2) > 0:
            nghs1 = set([ill_ents_dict1[e] for e in idx2nghs1[ill_ents_dict2[e2]]])
            remain_nghs = nghs1.intersection(nghs2)
            idx2nghs[e2] = list(remain_nghs)

idx2triples = {}
for i in range(len(triples_1)):
    h, r, t = triples_1[i]
    if h not in idx2triples:
        idx2triples[h] = []
    if t in idx2nghs:
        idx2triples[h].append((idx2ent[h], idx2rel[r], idx2ent[t]))
    if t not in idx2triples:
        idx2triples[t] = []
    if h in idx2nghs:
        idx2triples[t].append((idx2ent[h], idx2rel[r], idx2ent[t]))
for i in range(len(triples_2)):
    h, r, t = triples_2[i]
    if h not in idx2triples:
        idx2triples[h] = []
    if t in idx2nghs:
        idx2triples[h].append((idx2ent[h], idx2rel[r], idx2ent[t]))
    if t not in idx2triples:
        idx2triples[t] = []
    if h in idx2nghs:
        idx2triples[t].append((idx2ent[h], idx2rel[r], idx2ent[t]))

logger = setup_logger()
acc = 0.

# 1. 初始化模型
model = LLM(
    model="/math_a100/data/yly/LLM-Research/Meta-Llama-3___1-8B-Instruct",
    dtype="bfloat16",
    trust_remote_code=True,
    max_num_seqs=8  # 并行生成数量，可视显存调大
)

# 2. 设置采样参数
sampling_params = SamplingParams(
    temperature=0.1  # 稳定输出
)

# 3. 构造 prompt 列表与 meta 信息
prompts = []
meta_info = []

for i in range(len(test_ills)):
    e1, e2 = test_ills[i, 0], test_ills[predicted_right_idx[i], 1]
    name1, name2 = idx2ent[e1], idx2ent[e2]
    triple1, triple2 = idx2triples[e1], idx2triples[e2]
    prompt = f"""You are an expert assistant. Analyze the entities from two different knowledge graphs and determine if they refer to the same real-world object based on their names, triples, and your own knowledge. Respond in the exact format: “result: Yes, reason: ” or “result: No, reason: ” depending on whether they represent the same object. Your explanation should only include essential information, with no line breaks or extra spaces. Be concise and accurate in your reasoning.

    entity_1: {{name: {name1}, triples: {triple1}}}
    entity_2: {{name: {name2}, triples: {triple2}}}

    Note that the estimated similarity between these two entities is: {predicted_right_logit[i]}"""

    try:
        outputs = model.generate([prompt], sampling_params)
    except Exception as e:
        print(f"Batch error: {e}")
        continue
    response = outputs[0].outputs[0].text.strip()
    print(response)
    logger.info(name1 + '\t' + name2 + '\n' + response + '\n')
    print(response)
    if 'yes' in response.lower() and predicted_right_idx[i] == i:
        acc += 1.

print(f"Accuracy: {acc}/100 = {acc / 100:.2%}")