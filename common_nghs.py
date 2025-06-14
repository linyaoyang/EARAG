"""计算公共邻居相似度"""

import numpy as np
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed


def compute_equivalent_neighbors_for_pair(i, A, B, C):
    n, m = A.shape[0], B.shape[1]
    neighbors_A = np.where(A[i] == 1)[0]
    result_row = np.zeros(m)

    for j in range(m):
        neighbors_B = np.where(B[j] == 1)[0]

        equivalent_neighbor_count = 0
        for neighbor_A in neighbors_A:
            for neighbor_B in neighbors_B:
                if C[neighbor_A][neighbor_B] == 1:
                    equivalent_neighbor_count += 1

        result_row[j] = equivalent_neighbor_count

    return i, result_row


def build_equivalent_neighbor_matrix_parallel(A, B, C, n_jobs=-1):
    n = A.shape[0]
    results = Parallel(n_jobs=n_jobs)(delayed(compute_equivalent_neighbors_for_pair)(i, A, B, C) for i in range(n))

    result_matrix = np.zeros((n, B.shape[1]))

    for i, result_row in results:
        result_matrix[i] = result_row

    return result_matrix


# 示例使用
data_dir = 'data/zh_en/'

ill_ents = np.loadtxt(data_dir + 'ref_ent_ids', delimiter='\t', dtype=int)
left_ills, right_ills = ill_ents[:, 0].tolist(), ill_ents[:, 1].tolist()
A1 = np.zeros((len(ill_ents), len(ill_ents)))
A2 = np.zeros((len(ill_ents), len(ill_ents)))
triples_1 = np.loadtxt(data_dir + 'triples_1', delimiter='\t', dtype=int)
triples_2 = np.loadtxt(data_dir + 'triples_2', delimiter='\t', dtype=int)
string_sim = np.load(data_dir + 'string_sim.npy')
struct_sim = np.load(data_dir + 'struct_sim_1.npy')

print('step 1')
# 填充 A1 和 A2
for i in range(len(triples_1)):
    if triples_1[i, 0] in left_ills and triples_1[i, 2] in left_ills:
        A1[left_ills.index(triples_1[i, 0]), left_ills.index(triples_1[i, 2])] = 1
for i in range(len(triples_2)):
    if triples_2[i, 0] in right_ills and triples_2[i, 2] in right_ills:
        A2[right_ills.index(triples_2[i, 0]), right_ills.index(triples_2[i, 2])] = 1

E = np.zeros((len(ill_ents), len(ill_ents)))
name_emb = np.load(data_dir + 'name_emb.npy')
name_dist = cdist(name_emb[ill_ents[:, 0]], name_emb[ill_ents[:, 1]], 'braycurtis')
name_sim = 1 - name_dist
overalll_sim = name_sim + string_sim + struct_sim
name_sort_idx, name_sort_jdx = np.argsort(-overalll_sim, axis=1), np.argsort(-overalll_sim, axis=0)

print('step 2')
for i in range(len(ill_ents)):
    j = name_sort_idx[i, 0]
    if name_sort_jdx[0, j] == i:
        E[i, j] = 1

common_nghs = build_equivalent_neighbor_matrix_parallel(A1, A2, E)

# 计算 hits@1
hits1 = 0.
for i in range(len(common_nghs)):
    if np.argmax(common_nghs[i]) == i:
        hits1 += 1

common_nghs = common_nghs / (np.max(common_nghs, axis=1, keepdims=True) + 1)
np.save(data_dir + 'common_nghs.npy', common_nghs)

print(common_nghs[0])
print(np.sum(common_nghs))
print(hits1 / len(common_nghs))
