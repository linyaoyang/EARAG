import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Evaluate(nn.Module):
    def __init__(self, dev_pair, k=10):
        super(Evaluate, self).__init__()
        self.dev_pair = dev_pair
        self.k = k

    def dot_product(self, tensor):
        A, B = tensor
        A_sim = torch.matmul(A, B.t())
        return A_sim

    def top_k_avg(self, x):
        top_k_values = torch.topk(x, self.k, dim=-1)[0]
        return torch.mean(top_k_values, dim=-1)

    def CSLS(self, tensor):
        sim, LR, RL = tensor
        sim = 2 * sim - LR.t()
        sim = sim - RL
        rank = torch.argsort(-sim, dim=-1)
        val = torch.sort(-sim, dim=-1)[0]
        return rank[:, 0], val[:, 0]

    def CSLS_with_ans(self, tensor):
        sim, LR, RL, ans_rank = tensor
        sim = 2 * sim - LR.t()
        sim = sim - RL
        rank = torch.argsort(-sim, dim=-1)
        results = (rank == ans_rank).nonzero(as_tuple=True)
        return results

    def forward(self, Lvec, Rvec, evaluate=True, batch_size=1024):
        batch_size = len(Lvec)
        L_sim, R_sim = [], []

        for epoch in range((len(Lvec) // batch_size) + 1):
            L_sim.append(self.dot_product([Lvec[epoch * batch_size:(epoch + 1) * batch_size], Rvec]))
            R_sim.append(self.dot_product([Rvec[epoch * batch_size:(epoch + 1) * batch_size], Lvec]))

        LR, RL = [], []
        for epoch in range((len(Lvec) // batch_size) + 1):
            LR.append(self.top_k_avg(L_sim[epoch]))
            RL.append(self.top_k_avg(R_sim[epoch]))

        if evaluate:
            results = []
            for epoch in range((len(Lvec) // batch_size) + 1):
                ans_rank = torch.arange(epoch * batch_size, min((epoch + 1) * batch_size, len(Lvec))).long()
                result = self.CSLS_with_ans([R_sim[epoch], torch.stack(LR, dim=1), RL[epoch], ans_rank])
                results.append(result)
            return torch.cat(results, dim=1)
        else:
            l_rank, r_rank = [], []
            l_rank_val_list, r_rank_val_list = [], []

            for epoch in range((len(Lvec) // batch_size) + 1):
                ans_rank = torch.arange(epoch * batch_size, min((epoch + 1) * batch_size, len(Lvec))).long()
                r_rank_index, r_rank_val = self.CSLS([R_sim[epoch], torch.stack(LR, dim=1), RL[epoch]])
                l_rank_index, l_rank_val = self.CSLS([L_sim[epoch], torch.stack(RL, dim=1), LR[epoch]])

                r_rank.append(r_rank_index)
                l_rank.append(l_rank_index)

                r_rank_val_list.append(r_rank_val)
                l_rank_val_list.append(l_rank_val)

            return torch.cat(r_rank, dim=1), torch.cat(l_rank, dim=1), torch.cat(r_rank_val_list, dim=1), torch.cat(l_rank_val_list, dim=1)

    def test(self, Lvec, Rvec):
        results = self.forward(Lvec, Rvec)

        def cal(results):
            hits1, hits5, hits10, mrr = 0, 0, 0, 0
            for x in results:
                if x < 1:
                    hits1 += 1
                if x < 5:
                    hits5 += 1
                if x < 10:
                    hits10 += 1
                mrr += 1 / (x + 1)
            return hits1, hits5, hits10, mrr

        hits1, hits5, hits10, mrr = cal(results)
        print(f"Hits@1: {hits1 / len(Lvec)} Hits@5: {hits5 / len(Lvec)} Hits@10: {hits10 / len(Lvec)} MRR: {mrr / len(Lvec)}")
        return results

    def elect(self, Lvec, Rvec):
        batch_size = len(Lvec)
        L_sim, R_sim = [], []

        for epoch in range((len(Lvec) // batch_size) + 1):
            L_sim.append(self.dot_product([Lvec[epoch * batch_size:(epoch + 1) * batch_size], Rvec]))
            R_sim.append(self.dot_product([Rvec[epoch * batch_size:(epoch + 1) * batch_size], Lvec]))

        return L_sim[0].squeeze()
