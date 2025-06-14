import torch
import torch.nn as nn
from torch_scatter import scatter_sum
import torch.nn.functional as F


class GraphAttention(nn.Module):
    def __init__(self, node_size, rel_size, triple_size, node_dim, depth=1):
        super(GraphAttention, self).__init__()
        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.node_dim = node_dim
        self.activation = torch.nn.Tanh()
        self.depth = depth
        self.attn_kernels = nn.ParameterList()

        self.w1 = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.w2 = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.w3 = nn.Parameter(torch.tensor(1.0, requires_grad=True))

        # attention kernel
        for l in range(self.depth):
            attn_kernel = torch.nn.Parameter(data=torch.empty(self.node_dim, 1, dtype=torch.float32))  # 128 * 1
            torch.nn.init.xavier_uniform_(attn_kernel)
            self.attn_kernels.append(attn_kernel)

    def forward(self, inputs):
        outputs = []
        features = inputs[0]  # rel_feature  38960 * 128
        rel_emb = inputs[1]  # rel_embedding  6050 * 128
        adj = inputs[2]  # adj_list  2 * 259545
        r_index = inputs[3]  # r_index  2 * 331112
        r_val = inputs[4]  # r_val  331112

        features = self.activation(features)
        outputs.append(features)

        for l in range(self.depth):
            attention_kernel = self.attn_kernels[l]
            # matrix shape: [N_tri x N_rel]
            tri_rel = torch.sparse_coo_tensor(indices=r_index, values=r_val,
                                              size=[self.triple_size, self.rel_size], dtype=torch.float32)
            # shape: [N_tri x dim]
            tri_rel = torch.sparse.mm(tri_rel, rel_emb)
            # shape: [N_tri x dim]
            neighs = features[adj[1, :].long()]

            tri_rel = F.normalize(tri_rel, dim=1, p=2)
            neighs = neighs - 2 * torch.sum(neighs * tri_rel, dim=1, keepdim=True) * tri_rel

            att = torch.squeeze(torch.mm(tri_rel, attention_kernel), dim=-1)  # 259545
            att = torch.sparse_coo_tensor(indices=adj, values=att, size=[self.node_size, self.node_size])
            att = torch.sparse.softmax(att, dim=1)

            new_features = scatter_sum(src=neighs * torch.unsqueeze(att.coalesce().values(), dim=-1), dim=0,
                                       index=adj[0, :].long())

            features = self.activation(new_features)
            outputs.append(features)

        outputs = torch.cat(outputs, dim=-1)

        # proxy_att = torch.mm(F.normalize(outputs, p=2, dim=-1), torch.transpose(F.normalize(self.proxy, p=2, dim=-1), 0, 1))
        # proxy_att = F.softmax(proxy_att, dim=-1)
        # proxy_feature = outputs - torch.mm(proxy_att, self.proxy)

        # gate_rate = torch.sigmoid(self.gate(proxy_feature))

        # final_outputs = gate_rate * outputs + (1-gate_rate) * proxy_feature
        final_outputs = outputs

        return final_outputs


class XGAT(nn.Module):
    def __init__(self, node_hid, rel_hid, triple_size, node_size, new_node_size, rel_size, device, adj_matrix, r_index,
                 r_val, rel_matrix, ent_matrix, dropout_rate=0.0, gamma=3, depth=2):
        super(XGAT, self).__init__()
        self.e_encoder = GraphAttention(node_size=new_node_size, rel_size=rel_size, triple_size=triple_size,
                                        node_dim=node_hid, depth=depth)
        self.r_encoder = GraphAttention(node_size=new_node_size, rel_size=rel_size, triple_size=triple_size,
                                        node_dim=node_hid, depth=depth)
        self.ent_embedding = nn.Embedding(node_size, node_hid)
        self.rel_embedding = nn.Embedding(rel_size, rel_hid)
        torch.nn.init.xavier_uniform_(self.ent_embedding.weight)
        torch.nn.init.xavier_uniform_(self.rel_embedding.weight)
        self.node_size = node_size
        self.rel_size = rel_size
        self.adj_list = adj_matrix.to(device)
        self.r_index = r_index.to(device)
        self.gamma = gamma
        self.r_val = r_val.to(device)
        self.rel_adj = rel_matrix.to(device)
        self.ent_adj = ent_matrix.to(device)
        self.dropout = nn.Dropout(dropout_rate)

    def avg(self, adj, emb, size: int):
        adj = torch.sparse_coo_tensor(indices=adj, values=torch.ones_like(adj[0, :], dtype=torch.float),
                                      size=[self.node_size, size])
        adj = torch.sparse.softmax(adj, dim=1)
        return torch.sparse.mm(adj, emb)

    def gcn_forward(self):
        ent_feature = self.avg(self.ent_adj, self.ent_embedding.weight, self.node_size)
        rel_feature = self.avg(self.rel_adj, self.rel_embedding.weight, self.rel_size)
        opt = [self.rel_embedding.weight, self.adj_list, self.r_index, self.r_val]
        out_rel_feature = self.r_encoder([rel_feature] + opt)
        out_ent_feature = self.e_encoder([ent_feature] + opt)
        return out_ent_feature, out_rel_feature

    def forward(self, train_paris: torch.Tensor):
        out_ent_feature, out_rel_feature = self.gcn_forward()

        out_feature_join = torch.cat((out_ent_feature, out_rel_feature), dim=-1)

        loss = self.align_loss(train_paris, out_feature_join)
        return loss

    def align_loss(self, pairs, emb):
        def squared_dist(A, B):
            row_norms_A = torch.sum(torch.square(A), dim=1)
            row_norms_A = torch.reshape(row_norms_A, [-1, 1])
            row_norms_B = torch.sum(torch.square(B), dim=1)
            row_norms_B = torch.reshape(row_norms_B, [1, -1])
            return row_norms_A + row_norms_B - 2 * torch.matmul(A, B.t())

        l, r = pairs[:, 0].long(), pairs[:, 1].long()
        l_emb, r_emb = emb[l], emb[r]

        pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
        l_neg_dis = squared_dist(l_emb, emb)
        r_neg_dis = squared_dist(r_emb, emb)

        del l_emb, r_emb

        l_loss = pos_dis - l_neg_dis + self.gamma
        l_loss = l_loss * (1 - F.one_hot(l, num_classes=self.node_size) - F.one_hot(r, num_classes=self.node_size))
        r_loss = pos_dis - r_neg_dis + self.gamma
        r_loss = r_loss * (1 - F.one_hot(l, num_classes=self.node_size) - F.one_hot(r, num_classes=self.node_size))

        del r_neg_dis, l_neg_dis

        r_loss = (r_loss - torch.mean(r_loss, dim=-1, keepdim=True).detach()) / torch.std(r_loss, dim=-1,
                                                                                          unbiased=False,
                                                                                          keepdim=True).detach()
        l_loss = (l_loss - torch.mean(l_loss, dim=-1, keepdim=True).detach()) / torch.std(l_loss, dim=-1,
                                                                                          unbiased=False,
                                                                                          keepdim=True).detach()

        lamb, tau = 20, 8

        l_loss = torch.logsumexp(lamb * l_loss + tau, dim=-1)
        r_loss = torch.logsumexp(lamb * r_loss + tau, dim=-1)

        return torch.mean(l_loss + r_loss)

    def get_embeddings(self, index_a, index_b):
        # forward
        out_ent_feature, out_rel_feature = self.gcn_forward()
        # out_feature_join= out_ent_feature
        out_feature_join = torch.cat((out_ent_feature, out_rel_feature), dim=-1)
        out_feature = out_feature_join
        out_feature = out_feature.cpu()

        # get embeddings
        index_a = torch.Tensor(index_a).long()
        index_b = torch.Tensor(index_b).long()
        Lvec = out_feature[index_a]
        Rvec = out_feature[index_b]
        Lvec = Lvec / (torch.linalg.norm(Lvec, dim=-1, keepdim=True) + 1e-5)
        Rvec = Rvec / (torch.linalg.norm(Rvec, dim=-1, keepdim=True) + 1e-5)

        return Lvec, Rvec

    # def get_align_sim(self, left_ind, right_ind):
    #     out_ent_feature, out_rel_feature = self.gcn_forward()
    #     left_ind = torch.Tensor(left_ind).long()
    #     right_ind = torch.Tensor(right_ind).long()
    #     aep_left, aep_right = self.co_attention(out_ent_feature, out_rel_feature, self.img_feature,
    #                                             left_ind, right_ind)
    #     return aep_left, aep_right