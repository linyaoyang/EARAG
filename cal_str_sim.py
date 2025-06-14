import numpy as np
from scipy.spatial.distance import cdist
import json
import warnings
warnings.filterwarnings('ignore')

import os
import random
import re
from unidecode import unidecode
from tqdm import trange
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import torch
import torch.nn.functional as F
from module.utilization import load_data, direct_test, confident_paris
from module.encoder import XGAT
from module.utils import load_dwynb_ents
from concurrent.futures import ProcessPoolExecutor, as_completed  # 用于并行计算

data_dir = 'data/zh_en/'

name_emb = np.load(data_dir + 'name_emb.npy')
ill_ents = np.loadtxt(data_dir + 'ref_ent_ids', dtype=int, delimiter='\t')
# np.random.shuffle(ill_ents)

name_dist = cdist(name_emb[ill_ents[:, 0]], name_emb[ill_ents[:, 1]], 'braycurtis')
name_sim = 1 - name_dist
np.save(data_dir + 'name_sim.npy', name_sim)
name_sort = np.argsort(-name_sim, axis=1)

name_hits1, name_hits10 = 0., 0.
for i in range(len(name_sort)):
    if i == name_sort[i, 0]:
        name_hits1 += 1
    if i in name_sort[i, :10]:
        name_hits10 += 1
print('name_hits1: %.4f, name_hits10: %.4f' % (name_hits1 / len(name_sort), name_hits10 / len(name_sort)))

idx2ent, ents1, ents2 = load_dwynb_ents(data_dir)

def calculate_string_similarity_block(i_block, j_block):
    block_size_i = len(i_block)
    block_size_j = len(j_block)
    block_sim = np.zeros((block_size_i, block_size_j))
    for i_idx, i in enumerate(i_block):
        for j_idx, j in enumerate(j_block):
            s1 = idx2ent[ill_ents[i, 0]]
            s1 = s1.lower().replace('(', '').replace(')', '')
            s1 = unidecode(s1)
            s1 = re.sub("[\s+\.\!\/_,$%^*_\-(+\"\')]+|[+—?【】“”！，。？、~@#￥%……&*（）]+'", "", s1)
            s2 = idx2ent[ill_ents[j, 1]]
            s2 = s2.lower().replace('(', '').replace(')', '')
            s2 = unidecode(s2)
            s2 = re.sub("[\s+\.\!\/_,$%^*_\-(+\"\')]+|[+—?【】“”！，。？、~@#￥%……&*（）]+'", "", s2)

            block_sim[i_idx, j_idx] = fuzz.token_sort_ratio(s1, s2) / 100
    return block_sim, i_block, j_block

def calculate_string_accuracy_hits(i):
    global str_sim_sort
    acc = 0
    hits10 = 0
    if str_sim_sort[i, 0] == i:
        acc += 1
    if i in str_sim_sort[i, :10]:
        hits10 += 1
    return acc, hits10

block_size = 1000
blocks = [(list(range(i, min(i + block_size, len(ill_ents)))), list(range(j, min(j + block_size, len(ill_ents)))))
          for i in range(0, len(ill_ents), block_size)
          for j in range(0, len(ill_ents), block_size)]

string_sim = np.zeros((len(ill_ents), len(ill_ents)))

with ProcessPoolExecutor() as executor:
    future_to_block = {executor.submit(calculate_string_similarity_block, i_block, j_block): (i_block, j_block)
                       for i_block, j_block in blocks}
    for future in as_completed(future_to_block):
        block_sim, i_block, j_block = future.result()
        for i_idx, i in enumerate(i_block):
            for j_idx, j in enumerate(j_block):
                string_sim[i, j] = block_sim[i_idx, j_idx]

np.save(data_dir + 'string_sim.npy', string_sim)

str_acc, str_hits10 = 0., 0.
str_sim_sort = np.argsort(-string_sim, axis=1)

with ProcessPoolExecutor() as executor:
    future_to_acc_hits = {executor.submit(calculate_string_accuracy_hits, i): i
                          for i in range(len(ill_ents))}
    for future in as_completed(future_to_acc_hits):
        acc, hits10 = future.result()
        str_acc += acc
        str_hits10 += hits10

print('字符串相似度准确率: {:.4f}, Hits@10: {:.4f}'.format(str_acc / len(ill_ents), str_hits10 / len(ill_ents)))
