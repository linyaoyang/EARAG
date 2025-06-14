import scipy
import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial.distance import cdist


def load_triples(file):
    triples = []
    entity = set()
    rel = set([0])
    for line in open(file, 'r', encoding='utf-8'):
        head, r, tail = [int(item) for item in line.strip().split()]
        entity.add(head)
        entity.add(tail)
        rel.add(r + 1)
        triples.append((head, r + 1, tail))
    return entity, rel, triples

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T

def get_matrix(triples, entity, rel):
    ent_size = max(entity) + 1
    rel_size = (max(rel) + 1)
    print(ent_size, rel_size)
    adj_matrix = sp.lil_matrix((ent_size, ent_size))
    adj_features = sp.lil_matrix((ent_size, ent_size))
    radj = []
    rel_in = np.zeros((ent_size, rel_size))
    rel_out = np.zeros((ent_size, rel_size))

    for i in range(max(entity) + 1):
        adj_features[i, i] = 1

    for h, r, t in triples:
        adj_matrix[h, t] = 1
        adj_matrix[t, h] = 1
        adj_features[h, t] = 1
        adj_features[t, h] = 1
        radj.append([h, t, r])
        radj.append([t, h, r + rel_size])
        rel_out[h][r] += 1
        rel_in[t][r] += 1

    count = -1
    s = set()
    d = {}
    r_index, r_val = [], []
    for h, t, r in sorted(radj, key=lambda x: x[0] * 10e10 + x[1] * 10e5):
        if ' '.join([str(h), str(t)]) in s:
            r_index.append([count, r])
            r_val.append(1)
            d[count] += 1
        else:
            count += 1
            d[count] = 1
            s.add(' '.join([str(h), str(t)]))
            r_index.append([count, r])
            r_val.append(1)
    for i in range(len(r_index)):
        r_val[i] /= d[r_index[i][0]]

    # r_index 表示，r_val 表示关系 r 在实体 h 和 t 之间出现的次数
    rel_features = np.concatenate([rel_in, rel_out], axis=1)
    adj_features = normalize_adj(adj_features)  # 就是简单的邻接矩阵
    rel_features = normalize_adj(sp.lil_matrix(rel_features))  # 表示每个实体进入和出去的每种关系的数量
    return adj_matrix, r_index, r_val, adj_features, rel_features

def load_data(data_dir, init_seeds):
    ents1, rels1, triples1 = load_triples(data_dir + 'triples_1')
    ents2, rels2, triples2 = load_triples(data_dir + 'triples_2')
    sup_triples = [(item[0], 0, item[1]) for item in init_seeds]
    adj_matrix, r_index, r_val, adj_features, rel_features = get_matrix(triples1 + triples2 + sup_triples,
                                                                        ents1.union(ents2), rels1.union(rels2))
    return adj_matrix, np.array(r_index), np.array(r_val), adj_features, rel_features


def direct_test(Lvec, Rvec):
    hits1, hits10 = 0., 0.
    Lvec, Rvec = Lvec.numpy(), Rvec.numpy()
    dist = cdist(Lvec, Rvec, 'braycurtis')
    sorted_idx = np.argsort(dist, axis=1)
    for i in range(len(sorted_idx)):
        if sorted_idx[i, 0] == i:
            hits1 += 1
        if i in sorted_idx[i, :10]:
            hits10 += 1
    return hits1 / len(sorted_idx), hits10 / len(sorted_idx)

def confident_paris(sim, test_ills):
    new_pair = []
    sorted_idx = np.argmin(-sim, axis=1)
    sorted_jdx = np.argmin(-sim, axis=0)
    for i in range(len(sorted_idx)):
        j = sorted_idx[i]
        if sorted_jdx[j] == i:
            new_pair.append((test_ills[i, 0], test_ills[j, 1]))
    return np.array(new_pair)