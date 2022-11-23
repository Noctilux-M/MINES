import pickle
from sklearn.metrics import f1_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from Transition_encoder import transition_encoder
from collections import Counter
import math
from tqdm import tqdm
from Existence_encoder import existence_encoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import KMeans
import heapq

with open("/mnt/data1/MINES/acmd_seqs.pkl", "rb") as f1:
    acmd_labels = pickle.load(f1).astype(int)
    acmd_files = pickle.load(f1)  # files是序列数据

#得到pos矩阵的2个函数
def get_cos_sim_of_matrix(v1, v2):  # 计算两矩阵余弦相似度
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res  # 返回余弦相似度矩阵
def get_cos_similar_multi(v1: list, v2: list):
    num = np.dot([v1], np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res
def get_whether_pos(sim_matrix, k):  # 得到正样本矩阵，一行中1表示互为正样本，0表示互为负样本
    res = []  # 提前定义tfidf向量相似度前k大的互为正样本
    for i in sim_matrix:
        arg = np.argsort(i)
        threshold = i[arg[len(i) - k:]][0]
        cur = i >= threshold
        res.append(cur)
    res = np.array(res).astype(int)
    res = torch.Tensor(res)
    return res  # 返回正样本判断矩阵
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

batch_size = 100
n_epoch = 2000
lr = 0.0001
window_size = 5
n_hid = 60  # 100
n_inp = 128  # 256
n_out = 64  # embedding dimension d / 4
tau = 0.5
lam = 0.5
k=32 # the size of positive sample set
classCount = 200 # number of API clusters

num_classes =len(np.bincount(acmd_labels))
num_train = int(len(acmd_files)*0.7)
num_test = len(acmd_files) - num_train

# 3.3.1 Initialization phase
## 3.3.1.1 Word embedding
sentences = [s.split() for s in acmd_files]
model = Word2Vec(sentences, min_count=1)
keys = model.wv.index_to_key
# len(keys): 295
print('Word embedding ready')

## 3.3.1.2 Calculate similarity between APIs
sm_api = np.zeros((len(keys),len(keys)))
for i in range(len(keys)):
    for j in range(len(keys)):
        sm_api[i,j] = model.wv.similarity(keys[i],keys[j]) # sm是API嵌入的相似度矩阵
# sm.shape: (295, 295)
print('similarity between APIs ready')

clf = KMeans(n_clusters=classCount)
s = clf.fit(sm_api) #k-means聚类
labels = clf.labels_
cluster_vocab = {}
for i in range(len(keys)):
    cluster_vocab[keys[i]]=str(labels[i]) # cluster_vocab：API对应的簇{API：cluster}

## 3.3.2.1. Find API function in cluster
acmd_clustered = [] # acmd_clustered是把API替换为cluster的序列
for i in range(len(acmd_files)):
    cur_list = list(map(cluster_vocab.get, acmd_files[i].split()))
    cur = ' '.join(cur_list)
    acmd_clustered.append(cur)
clustered_ids = [[int(num) for num in acmd_clustered[i].split()] for i in range(len(acmd_clustered))]

## get top n index
topn_idx=[]
n = 10
for i in range(len(sm_api)):
    b=heapq.nlargest(n, range(len(sm_api[i])), sm_api[i].take)
    topn_idx.append(b)
emb_dict={}
for i in range(len(topn_idx)):
    for j in range(n):
        if sm_api[i][j]>0:
            emb_dict[(i,j)]=sm_api[i][j]
        else:
            emb_dict[(i, j)] = 1e-4
vocab = {}
for i in range(len(keys)):
    vocab[keys[i]]=int(i)
x_train_word_ids = []
for i in range(len(acmd_files)):
    cur_list = list(map(vocab.get, acmd_files[i].split()))
    x_train_word_ids.append(cur_list)
num_apis = len(vocab)
num_inputs = len(acmd_files)
num_nodes = num_apis + num_inputs
#
#emb_api, emb_file
onehot = np.eye(len(keys))#onehot+pca 得到api初始矩阵
pca=PCA(n_components=n_inp)
pca.fit(onehot)
onehot_pca = pca.transform(onehot)
emb_api=nn.Parameter(torch.tensor(onehot_pca), requires_grad=False).to(torch.float32).to(device)
emb_files = nn.Parameter(torch.Tensor(len(x_train_word_ids), n_inp), requires_grad=False)
emb_file = nn.init.xavier_uniform_(emb_files).to(device)

def markov(n, n_nodes, seqs_ids):
    x = seqs_ids[n]
    markov_matrix = np.zeros([n_nodes, n_nodes])
    for (i, j) in zip(x, x[1:]):
        markov_matrix[i - 1][j - 1] += 1
    for row in markov_matrix:
        s = sum(row)
        if s > 0:
            row[:] = [f / s for f in row]
    return markov_matrix

clustered_ids = [[int(num) for num in acmd_clustered[i].split()] for i in range(len(acmd_clustered))]
res = []
for i in range(len(clustered_ids)):
    cur = markov(i, classCount, clustered_ids)
    res.append(cur)
    # print(len(res))
mat = np.array(res).reshape(len(acmd_labels),1,classCount,classCount)
# 二阶maekov邻接矩阵
mat2 = np.multiply(mat, mat)
# 三阶maekov邻接矩阵
mat3 = np.multiply(mat, mat2)
markov_mat = np.concatenate((mat, mat2, mat3), axis=1)
markov_mat1 = torch.tensor(markov_mat).to(torch.float32) #out1

# 调整转移矩阵的API顺序
markov_all = np.mean(np.array(res), axis=0)
markov_order = [0]
for i in range(len(markov_all)):
    argsot_cur = np.argsort(markov_all[markov_order[-1]])[::-1]
    for i in argsot_cur:
        if i not in markov_order:
            markov_order.append(i)
            break
markov_order = np.array(markov_order)
reorder_dict = {}
for i in range(len(markov_order)):
    reorder_dict[markov_order[i]] = int(i)
acmd_reordered = []
for i in range(len(clustered_ids)):
    cur_list = list(map(reorder_dict.get, clustered_ids[i]))
    acmd_reordered.append(cur_list)
markov_mat_1 = []
for i in range(len(acmd_reordered)):
    cur = markov(i, classCount, acmd_reordered)
    markov_mat_1.append(cur)
    print(len(markov_mat_1))
mat = np.array(markov_mat_1).reshape(len(acmd_labels),1,classCount,classCount)

# 二阶maekov邻接矩阵
mat2 = np.multiply(mat, mat)
# 三阶maekov邻接矩阵
mat3 = np.multiply(mat, mat2)
markov_mat = np.concatenate((mat, mat2, mat3), axis=1)
markov_mat2 = torch.tensor(markov_mat).to(torch.float32) #out2

# g_PMI
#pmi
window_size = 5
windows = []
for x in x_train_word_ids:
    if len(x)<=window_size:
        windows.append(x)
    else:
        windows+=[x[i:i + window_size]
                  for i in range(len(x) - window_size + 1)]
windows = [list(win) for win in windows]
num_windows = len(windows)
print(f"num_windows={num_windows}")
word2window = Counter([w for win in windows for w in set(win)])
word_word2window = Counter([p for win in tqdm(windows)
                                for i, x in enumerate(win)
                                for y in win[i + 1:]
                                for p in ((x, y), (y, x)) if x != y])
word_word2pmi = {(x, y): math.log(c * num_windows / (word2window[x] * word2window[y]))
                     for (x, y), c in word_word2window.items()}
api_api2pmi = {(x, y): v for (x, y), v in word_word2pmi.items() if v > 0}

pmi_adj = []
pmi_weight = []
for adj, w in api_api2pmi.items():
    pmi_adj.append(list(adj))
    pmi_weight.append(w)
pmi_adj = np.array(pmi_adj)
pmi_adj = torch.from_numpy(pmi_adj.astype(int))
f_a = []
f_a_w=[] # software与API间的tfidf边权重
for i in range(len(x_train_word_ids)):
    for j in list(set(x_train_word_ids[i])):
        f_a.append([i, j])
        f_a_w.append(1)
f_a = np.array(f_a)
f_a = torch.from_numpy(f_a.astype(int))

data_dict = {
        ('api', 'called_by', 'file'): (f_a[:, 1], f_a[:, 0]),
    ('api', 'related_with', 'api'): (pmi_adj[:,1], pmi_adj[:, 0]),
('api', 'related_by', 'api'): (pmi_adj[:,0], pmi_adj[:, 1])
}
g_pmi = dgl.heterograph(data_dict).to(device)
g_pmi.nodes['api'].data['feature'] = emb_api
g_pmi.nodes['file'].data['feature'] = emb_file

g_pmi['related_with'].edata['w'] = torch.tensor(pmi_weight,dtype = torch.float32).to(device)
g_pmi['related_by'].edata['w'] = torch.tensor(pmi_weight,dtype = torch.float32).to(device)
g_pmi['called_by'].edata['w'] = torch.tensor(f_a_w,dtype = torch.float32).to(device)
print('g_pmi ready!')

# g_emb
a_a = []
for i in range(len(topn_idx)):
    for j in topn_idx[i]:
        a_a.append([i, j])

a_a = np.array(a_a)
a_a = torch.from_numpy(a_a.astype(int))

data_dict = {
        ('api', 'called_by', 'file'): (f_a[:, 1], f_a[:, 0]),
    ('api', 'related_with', 'api'): (a_a[:,1], a_a[:, 0]),
('api', 'related_by', 'api'): (a_a[:,0], a_a[:, 1])
}
g_emb = dgl.heterograph(data_dict).to(device)
g_emb.nodes['api'].data['feature'] = emb_api
g_emb.nodes['file'].data['feature'] = emb_file
g_emb['related_with'].edata['w'] = torch.tensor(list(emb_dict.values()),dtype = torch.float32).to(device)
g_emb['related_by'].edata['w'] = torch.tensor(list(emb_dict.values()),dtype = torch.float32).to(device)
g_emb['called_by'].edata['w'] = torch.tensor(f_a_w,dtype = torch.float32).to(device)

# pos
vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
train_features = vectorizer.fit_transform(acmd_files)
m = train_features.toarray()
sim_matrix = get_cos_sim_of_matrix(m, m)  # 得到tfidf余弦相似度矩阵
pos = get_whether_pos(sim_matrix, k)

print('pos tuning!')
for i in range(num_train):
    for j in range(num_train):
        if acmd_labels[i]==acmd_labels[j]:
            pos[i][j] = 1
            pos[j][i] = 1
        else:
            pos[i][j] = 0
            pos[j][i] = 0
print('pos tune finished!')

pos = pos.to(device)
pos_simclr = torch.eye(batch_size).to(device)

class MINES(nn.Module):
    def __init__(self, n_inp, n_hid,  n_out, n_class, tau, lam, pmi_rel_names,emb_rel_names):
        super(MINES, self).__init__()
        self.existence_encoder = existence_encoder(n_inp,  n_hid, n_out, tau, lam, pmi_rel_names,emb_rel_names)
        self.transition_encoder = transition_encoder(n_out, batch_size, tau)
        self.classifier = nn.Linear(4*n_out, n_class)
    def forward(self, pos, pos_simclr, g_pmi, g_emb, node_features, markov_mat1,markov_mat2,step):
        global batch_size
        z_e1,z_e2, loss_e = self.existence_encoder(pos, g_pmi, g_emb, node_features)
        if not step:
            z_e1 = z_e1[-(step + 1) * batch_size: ]
            z_e2 = z_e2[-(step + 1) * batch_size:]
        else:
            z_e1 = z_e1[-(step + 1) * batch_size: -step * batch_size]
            z_e2 = z_e2[-(step + 1) * batch_size: -step * batch_size]

        z_e = torch.cat((z_e1, z_e2), dim=1)
        z_t1,z_t2, loss_t = self.transition_encoder(markov_mat1, markov_mat2, pos_simclr)
        z_t = torch.cat((z_t1, z_t2), dim=1)
        z = torch.cat((z_e, z_t), dim=1)
        res = self.classifier(z)
        return res, loss_e + loss_t

#emb_nodes
onehot = np.eye(num_nodes)#onehot+pca 得到api+file初始矩阵
pca=PCA(n_components=n_inp)
pca.fit(onehot)
onehot_pca = pca.transform(onehot)
emb_nodes=nn.Parameter(torch.tensor(onehot_pca), requires_grad=False).to(torch.float32)
emb_nodes = emb_nodes.to(device)
node_features = {'api': emb_api, 'file': emb_file}

model = MINES(n_inp, n_hid, n_out, num_classes, tau, lam, g_pmi.etypes, g_emb.etypes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_batch = num_train//batch_size
test_batch = num_train//batch_size
num = len(acmd_files) // batch_size
for i in range(n_epoch):
    loss_all = 0
    f1_train = 0
    model.train()
    for step in range(train_batch):
        if not step:
            markov_mat1_batch = markov_mat1[-(step + 1) * batch_size: ]
            markov_mat2_batch = markov_mat2[-(step + 1) * batch_size:]
        else:
            markov_mat1_batch = markov_mat1[-(step + 1) * batch_size: -step * batch_size]
            markov_mat2_batch = markov_mat2[-(step + 1) * batch_size: -step * batch_size]
        markov_mat1_batch = markov_mat1_batch.to(device)
        markov_mat2_batch = markov_mat2_batch.to(device)

        pre, loss_c = model(pos, pos_simclr, g_pmi, g_emb, node_features, markov_mat1_batch,markov_mat2_batch,step)
        if not step:
            label_array = acmd_labels[-(step + 1) * batch_size:].astype(int)
        else:
            label_array = acmd_labels[-(step + 1) * batch_size: -step * batch_size].astype(int)
        label = torch.tensor(label_array, dtype=torch.long).to(device)
        loss_s =F.cross_entropy(pre, label).requires_grad_(True)
        loss = loss_c + loss_s
        optimizer.zero_grad()
        loss.backward()
        loss_all += loss
        optimizer.step()
        pre_label = pre.argmax(1)
        del markov_mat1_batch, markov_mat2_batch
    #测试集
    f1_micro_all = 0
    f1_macro_all = 0
    model.eval()
    for step in range(test_batch,num):
        markov_mat1_batch = markov_mat1[step * batch_size:(step + 1) * batch_size]
        markov_mat2_batch = markov_mat2[step * batch_size:(step + 1) * batch_size]
        markov_mat1_batch = markov_mat1_batch.to(device)
        markov_mat2_batch = markov_mat2_batch.to(device)
        pre, _ = model(pos, pos_simclr, g_pmi, g_emb, node_features, markov_mat1_batch, markov_mat2_batch, step)
        pre_label = pre.argmax(1)
        label_array = acmd_labels[step * batch_size:(step + 1) * batch_size].astype(int)
        label = torch.tensor(label_array)
        f1_micro = f1_score(label.cpu(), pre_label.cpu(), average='micro')
        f1_micro_all += f1_micro / (num - test_batch)
        f1_macro = f1_score(label.cpu(), pre_label.cpu(), average='macro')
        f1_macro_all += f1_macro / (num - test_batch)
        del pre
    print('MINES acmd epoch', i, 'f1_micro', f1_micro_all, 'f1_macro', f1_macro_all)
