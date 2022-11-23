import torch
import torch.nn as nn

class CNN123(nn.Module):
    def __init__(self, out_dim):
        super(CNN123, self).__init__()
        ## 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2,
            ),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        ## 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.output = nn.Linear(in_features=32 * 51 * 51, out_features=out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.output(x)
        return output
class Loss(torch.nn.Module):
    def __init__(self, batch_size, t = 0.5):
        super(Loss,self).__init__()
        self.batch_size = batch_size
        self.temperature = t
    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.temperature)
        return sim_matrix
    def forward(self, z1, z2, pos):
        matrix_mp2sc = self.sim(z1, z2)
        matrix_sc2mp = matrix_mp2sc.t()
        matrix_mp2sc = matrix_mp2sc / (torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        lori_mp = -torch.log(matrix_mp2sc.mul(pos).sum(dim=-1)).mean()
        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log(matrix_sc2mp.mul(pos).sum(dim=-1)).mean()
        return 0.5 * lori_mp + 0.5 * lori_sc  # 返回损失函数
class transition_encoder(nn.Module):
    def __init__(self,  n_out,  batch_size, tau):
        super(transition_encoder, self).__init__()
        self.cnn1 = CNN123(n_out)
        self.cnn2 = CNN123(n_out)
        self.contrast = Loss(batch_size, tau)
        self.mlp1 = nn.Linear(n_out, n_out)
        self.mlp2 = nn.Linear(n_out, n_out)
    def forward(self, m1, m2, pos):
        z1 = self.cnn1(m1)
        z2 = self.cnn2(m2)
        z1c = self.mlp1(z1)
        z2c = self.mlp2(z2)
        loss = self.contrast(z1c, z2c, pos)
        return z1, z2, loss