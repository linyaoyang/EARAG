"""Step 1: learn structural embeddings."""

import json
import warnings
warnings.filterwarnings('ignore')

import os
import random
import numpy as np
from tqdm import trange
from scipy.spatial.distance import cdist
import torch
import torch.nn.functional as F
from module.utilization import load_data, direct_test, confident_paris
from module.encoder import XGAT
from module.utils import load_dwynb_ents
# from module.evaluate import Evaluate

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

set_seed(3407)
ratio = 0.3
data_dir = 'data/zh_en/'
# if not os.path.exists(res_dir):
#     os.makedirs(res_dir)

# obtain pseudo alignment seeds from pretrained name embeddings
idx2ent, ents1, ents2 = load_dwynb_ents(data_dir)

name_emb = np.load(data_dir + 'name_emb.npy')
ill_ents = np.loadtxt(data_dir + 'ref_ent_ids', dtype=int, delimiter='\t')
np.random.shuffle(ill_ents)

train_num = int(len(ill_ents) * ratio)
train_ills = [(ill_ents[i, 0], ill_ents[i, 1]) for i in range(train_num)]
test_ills = ill_ents[train_num:, :]
test_indices = find_row_positions(ill_ents, test_ills)

left_emb, right_emb = name_emb[ill_ents[train_num:, 0]], name_emb[ill_ents[train_num:, 1]]
dist = cdist(left_emb, right_emb, 'braycurtis')
sorted_idx, sorted_jdx = np.argsort(dist, axis=1), np.argsort(dist, axis=0)
sup_ills = []
hits1, hits10 = 0., 0.
for i in range(len(dist)):
    j = sorted_idx[i, 0]
    if i == sorted_idx[i, 0]:
        hits1 += 1
    if i in sorted_idx[i, :10]:
        hits10 += 1
    if sorted_jdx[0, j] == i:
        sup_ills.append((test_ills[i, 0], test_ills[j, 1]))
print('hit@1 of name similarity: %.4f, hit@10 of name similarity: %.4f' % (hits1 / len(dist), hits10 / len(dist)))

adj_matrix, r_index, r_val, adj_features, rel_features = load_data(data_dir, train_ills + sup_ills)
init_seeds = np.array(train_ills + sup_ills)
adj_matrix = np.stack(adj_matrix.nonzero(), axis=1)
rel_matrix, rel_val = np.stack(rel_features.nonzero(), axis=1), rel_features.data
ent_matrix, ent_val = np.stack(adj_features.nonzero(), axis=1), adj_features.data
train_ills = np.array(train_ills)
np.save(data_dir + 'train_ills.npy', train_ills)
np.save(data_dir + 'test_ills.npy', test_ills)

string_sim = np.load(data_dir + 'string_sim.npy')
name_sim = 1- dist

# hyperparameters
node_size = adj_features.shape[0]
rel_size = rel_features.shape[1]
triple_size = len(adj_matrix)
node_hidden = 100
rel_hidden =100
batch_size = 1024
dropout_rate = 0.3
lr = 0.005
gamma = 1
depth = 2
epoch = 12
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# load dy_nb into cuda
adj_matrix = torch.from_numpy(np.transpose(adj_matrix))  # 2 * N
rel_matrix = torch.from_numpy(np.transpose(rel_matrix))
ent_matrix = torch.from_numpy(np.transpose(ent_matrix))
r_index = torch.from_numpy(np.transpose(r_index))
r_val = torch.from_numpy(r_val)

model = XGAT(node_hid=node_hidden, rel_hid=rel_hidden, triple_size=triple_size, node_size=node_size,
             new_node_size=node_size, rel_size=rel_size, device=device, adj_matrix=adj_matrix, r_index=r_index,
             r_val=r_val, rel_matrix=rel_matrix, ent_matrix=ent_matrix, dropout_rate=dropout_rate, gamma=gamma,
             depth=depth)
model = model.to(device)
opt = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=0.0005)
print('model developed!')


model.eval()
with torch.no_grad():
    Lvec, Rvec = model.get_embeddings(test_ills[:, 0], test_ills[:, 1])
    # output = model(inputs)
    # Lvec, Rvec = get_embedding(dev_pair[:, 0], dev_pair[:, 1], output.cpu())
    # evaluator.test(Lvec, Rvec)  # 测试初始化的嵌入向量对齐效果, 特别差, 约等于0
    hits1, hits10 = direct_test(Lvec, Rvec)
    print('Hits@1 of structure emb similarity: {:.4f}, Hits@10 of structure emb similarity: {:.4f} before training'.format(hits1, hits10))

for turn in range(5):
    for i in trange(epoch):
        model.train()
        np.random.shuffle(init_seeds)
        for pairs in [init_seeds[i * batch_size:(i + 1) * batch_size] for i in
                      range(len(init_seeds) // batch_size + 1)]:
            inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix]
            # output = model(inputs)
            # loss = align_loss(pairs, output)
            pairs = torch.from_numpy(pairs).to(device)
            loss = model(pairs)
            # print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        Lvec, Rvec = model.get_embeddings(test_ills[:, 0], test_ills[:, 1])
        # left_name_emb = torch.FloatTensor(name_emb[ill_ents[:, 0]])
        # right_name_emb = torch.FloatTensor(name_emb[ill_ents[:, 1]])
        # Lvec = torch.cat((Lvec, left_name_emb), dim=1)
        # Rvec = torch.cat((Rvec, right_name_emb), dim=1)
        hits1, hits10 = direct_test(Lvec, Rvec)
        print('Hits@1 of name emb similarity: {:.4f}, Hits@10 of name emb similarity: {:.4f} after training'.format(hits1, hits10))
        Lvec, Rvec = Lvec.numpy(), Rvec.numpy()
        dist = cdist(Lvec, Rvec, 'braycurtis')
        sct_sim = 1 - dist
        str_sim = string_sim[test_indices, :][:, test_indices]
        n_sim = name_sim
        new_pair = confident_paris(sct_sim + str_sim + n_sim, test_ills)
        init_seeds = np.concatenate((train_ills, new_pair), axis=0)
        epoch = 5

    if turn < 1:
        model.eval()
        with torch.no_grad():
            Lvec, Rvec = model.get_embeddings(ill_ents[:, 0], ill_ents[:, 1])
        struct_dist = cdist(Lvec, Rvec, 'braycurtis')
        np.save(data_dir + 'struct_sim_0.npy', 1 - struct_dist)

# Lvec, Rvec = Lvec.numpy(), Rvec.numpy()
model.eval()
with torch.no_grad():
    Lvec, Rvec = model.get_embeddings(ill_ents[:, 0], ill_ents[:, 1])
struct_dist = cdist(Lvec, Rvec, 'braycurtis')
np.save(data_dir + 'struct_sim_1.npy', 1 - struct_dist)
print('structural learning finished')
hits1, hits5, hits10 = 0.0, 0.0, 0.0
overall_sim = 1 - struct_dist[test_indices, :][:, test_indices] + str_sim + n_sim
overall_idx = np.argsort(-overall_sim, axis=1)
for i in range(len(overall_idx)):
    if i == overall_idx[i, 0]:
        hits1 += 1
    if i in overall_idx[i, :5]:
        hits5 += 1
    if i in overall_idx[i, :10]:
        hits10 += 1
print('Hits@1 of overall similarity: {:.4f}, Hits@5 of overall similarity: {:.4f}, Hits@10 of overall similarity: {:.4f}'.format(hits1 / len(overall_idx), hits5 / len(overall_idx), hits10 / len(overall_idx)))