import torch
import torch.nn as nn
import torch.nn.functional as F


class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):  # isinstance判断model是否是nn.Linear类型
                nn.init.xavier_normal_(model.weight, gain=1.414)
    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)  # 求z1的-1范数
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_mp, z_sc, pos):
        # 把两视角下的嵌入映射到同一个空间
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)

        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()

        matrix_mp2sc = matrix_mp2sc / (torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        # lori_mp = -torch.log(matrix_mp2sc.mul(pos.to_dense()).sum(dim=-1)).mean()  # log,求完loss
        lori_mp = -torch.log(matrix_mp2sc.mul(pos).sum(dim=-1)).mean()

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        # lori_sc = -torch.log(matrix_sc2mp.mul(pos.to_dense()).sum(dim=-1)).mean()
        lori_sc = -torch.log(matrix_sc2mp.mul(pos).sum(dim=-1)).mean()
        return self.lam * lori_mp + (1 - self.lam) * lori_sc  # 返回损失函数



# if __name__=='__main__':
#     import pickle
#     from sklearn.model_selection import train_test_split
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     import time
#     import csv
#     import xgboost as xgb
#     from sklearn.model_selection import StratifiedKFold
#     import numpy  as np
#
#     with open("/mnt/data1/security_train.csv.pkl", "rb") as f:
#         labels = pickle.load(f)
#         files = pickle.load(f)#files是序列数据
#     k=5
#
#     def get_cos_sim_of_matrix(v1, v2):  # 计算两矩阵余弦相似度
#         num = np.dot(v1, np.array(v2).T)  # 向量点乘
#         denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
#         res = num / denom
#         res[np.isneginf(res)] = 0
#         return 0.5 + 0.5 * res  # 返回余弦相似度矩阵
#     def get_whether_pos(sim_matrix, k):  # 得到正样本矩阵，一行中1表示互为正样本，0表示互为负样本
#         res = []  # 提前定义tfidf向量相似度前k大的互为正样本
#         for i in sim_matrix:
#             arg = np.argsort(i)
#             threshold = i[arg[len(i) - k:]][0]
#             cur = i >= threshold
#             res.append(cur)
#         res = np.array(res).astype(int)
#         res = torch.Tensor(res)
#         return res  # 返回正样本判断矩阵
#
#     vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
#     train_features = vectorizer.fit_transform(files)
#     m = train_features.toarray()
#     sim_matrix = get_cos_sim_of_matrix(m, m)  # 得到tfidf余弦相似度矩阵
#     pos = get_whether_pos(sim_matrix, k) # 正样本判断矩阵(i,j)为true，代表第i，j个样本互为正样本
#     pos1 = pos[:, :256]
#     z_str = torch.rand(13887, 64)
#     z_seq = torch.rand(256, 64)
#     model = Contrast(64, 0.5, 0.5)
#     loss = model(z_str, z_seq, pos1)
#     print(loss)






