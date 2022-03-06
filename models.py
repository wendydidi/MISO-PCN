import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder01(nn.Module):
    def __init__(self):
        super(Encoder01, self).__init__()      
        # first shared mlp
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        ### first shared mlp
        x = F.relu(self.bn1(self.conv1(x)))           # (B, 128, N)
        x = self.bn2(self.conv2(x))                   # (B, 256, N)
        f1 = x
        f1_gt = torch.max(x, dim=2, keepdim=True)[0]      # (B, 256, 1)
        #####################
        return f1_gt, f1    #(B, 256, 1) / (B, 256, N)

class Encoder02(nn.Module):
    def __init__(self):
        super(Encoder02, self).__init__() 

        self.conv1 = nn.Conv1d(512, 512, 1)
        self.conv2 = nn.Conv1d(512, 1024, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
    def forward(self, f1_gt, x):
        n = x.size()[2]
        # expand and concat
        f = torch.cat([f1_gt.repeat(1, 1, n), x], dim=1)  # (B, 512, N)
        f = F.relu(self.bn1(self.conv1(f)))           # (B, 512, N)
        f = self.bn2(self.conv2(f))                   # (B, 1024, N)
        f2 = f
        # point-wise maxpool
        f2_gt = torch.max(f, dim=2, keepdim=True)[0]  # (B, 1024, 1)   
        return f2_gt, f2     # (B, 1024, 1) / (B, 1024, N)

class Decoder(nn.Module):
    def __init__(self, num_coarse=512, num_dense=2048):
        super(Decoder, self).__init__()
        self.num_coarse = num_coarse      
        # fully connected layers
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 3 * num_coarse)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
        # shared mlp
        self.conv1 = nn.Conv1d(3+2+1024, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 3, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        # 2D grid
        grids = np.meshgrid(np.linspace(-0.05, 0.05, 2, dtype=np.float32),
                            np.linspace(-0.05, 0.05, 2, dtype=np.float32))                               # (2, 4, 4)
        self.grids = torch.Tensor(grids).view(2, -1)  # (2, 2) -> (2, 4)
    def forward(self, x):
        b = x.size()[0]
        # global features
        v = x.view(b, -1)  # (B, 1024)
        # fully connected layers to generate the coarse output
        x = F.relu(self.bn1(self.linear1(v)))  # B, 1024
        x = F.relu(self.bn2(self.linear2(x))) # B, 1024
        x = self.linear3(x) # B, 3*512
        y_coarse = x.view(-1, 3, self.num_coarse)  # (B, 3, 512)   

        repeated_centers = y_coarse.unsqueeze(3).repeat(1, 1, 1, 4).view(b, 3, -1)  #  (b, 3, 4*512)
        repeated_v = v.unsqueeze(2).repeat(1, 1, 4 * self.num_coarse)               #  (b, 1024, 4*512)
        grids = self.grids.to(x.device)  # (2, 4)
        grids = grids.unsqueeze(0).repeat(b, 1, self.num_coarse)                     # (B, 2, 4*512)

        x = torch.cat([repeated_v, grids, repeated_centers], dim=1)                  # (B, 2+3+1024, 4*512)
        x = F.relu(self.bn3(self.conv1(x))) # -> b, 512, n'
        x = F.relu(self.bn4(self.conv2(x))) # -> b, 256, n'
        x = self.conv3(x)                # (B, 3, 16x1024)
        y_detail = x + repeated_centers  # (B, 3, 16x1024)

        return y_coarse, y_detail

class DecoderMI(nn.Module):
    def __init__(self, num_coarse=512, num_dense=2048):
        super(DecoderMI, self).__init__()
        self.num_coarse = num_coarse      
        # fully connected layers
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 3 * num_coarse)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
        # shared mlp
        self.conv1 = nn.Conv1d(3+2+1024, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.convadd = nn.Conv1d(256, 128, 1)
        self.conv3 = nn.Conv1d(256, 3, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        # 2D grid
        grids = np.meshgrid(np.linspace(-0.05, 0.05, 2, dtype=np.float32),
                            np.linspace(-0.05, 0.05, 2, dtype=np.float32))                               # (2, 4, 4)
        self.grids = torch.Tensor(grids).view(2, -1)  # (2, 2) -> (2, 4)
    def forward(self, x):
        b = x.size()[0]
        # global features
        v = x.view(b, -1)  # (B, 1024)
        # fully connected layers to generate the coarse output
        x = F.relu(self.bn1(self.linear1(v)))  # B, 1024
        x = F.relu(self.bn2(self.linear2(x))) # B, 1024
        x = self.linear3(x) # B, 3*512
        y_coarse = x.view(-1, 3, self.num_coarse)  # (B, 3, 512)   

        repeated_centers = y_coarse.unsqueeze(3).repeat(1, 1, 1, 4).view(b, 3, -1)  #  (b, 3, 4*512)
        repeated_v = v.unsqueeze(2).repeat(1, 1, 4 * self.num_coarse)               #  (b, 1024, 4*512)
        grids = self.grids.to(x.device)  # (2, 4)
        grids = grids.unsqueeze(0).repeat(b, 1, self.num_coarse)                     # (B, 2, 4*512)

        x = torch.cat([repeated_v, grids, repeated_centers], dim=1)                  # (B, 2+3+1024, 4*512)
        x = F.relu(self.bn3(self.conv1(x))) # -> b, 512, n'
        x = F.relu(self.bn4(self.conv2(x))) # -> b, 256, n'
        ##############################################################
        # xadd = x
        # fh_3 = self.convadd(xadd)
        # fh_2 = F.adaptive_max_pool1d(fh_3, 1)
        fh_3 = x # b, 256, n'
        fh_2 = F.adaptive_max_pool1d(fh_3, 1) # b, 256, 1
        ###############################################################
        x = self.conv3(x)                # (B, 3, 16x1024)
        y_detail = x + repeated_centers  # (B, 3, 16x1024)

        return fh_3, fh_2, y_coarse, y_detail

class AutoEncoder_gt(nn.Module):
    def __init__(self):
        super(AutoEncoder_gt, self).__init__()

        self.encoder01 = Encoder01()
        self.encoder02 = Encoder02()
        self.decoder = Decoder()

    def forward(self, x):
        b = x.size()[0]
        f1_gt, f1 = self.encoder01(x)
        f2_gt, f2 = self.encoder02(f1_gt, f1) # f2_gt 32,1024,1
        # print(f2_gt.size())
        y_coarse, y_detail = self.decoder(f2_gt)
        f1_gt = f1_gt.view(b, -1)
        f2_gt = f2_gt.view(b, -1)
        return f1_gt, f1, f2_gt, f2, y_coarse, y_detail


class AutoEncoder_part(nn.Module):
    def __init__(self):
        super(AutoEncoder_part, self).__init__()

        self.encoder01 = Encoder01()
        self.encoder02 = Encoder02()
        self.decoder = Decoder()

    def forward(self, x):
        b = x.size()[0]
        f1_part, f1 = self.encoder01(x)
        f2_part, f2 = self.encoder02(f1_part, f1) # f2_gt b,1024,1
        # print(f2_gt.size())
        out_coarse, out_detail = self.decoder(f2_part)
        f1_part = f1_part.view(b, -1)
        f2_part = f2_part.view(b, -1)
        return f1_part, f1, f2_part, f2, out_coarse, out_detail
            
class AutoEncoder_part_MI(nn.Module):
    def __init__(self):
        super(AutoEncoder_part_MI, self).__init__()
        self.encoder01 = Encoder01()
        self.encoder02 = Encoder02()
        self.decoder = DecoderMI()

    def forward(self, x):
        b = x.size()[0]
        f1_part, f1 = self.encoder01(x)
        f2_part, f2 = self.encoder02(f1_part, f1)
        fh_3, fh_2, out_coarse, out_detail = self.decoder(f2_part)
        #########################################################
        # f1_part: b,256,1
        # f1: b,256,n
        # f2_part: b,1024,1
        # f2: b,1024,n
        # fh_3: b,256,n'
        # fh_2: b,256,1
        f1_part = f1_part.view(b, -1)
        f2_part = f2_part.view(b, -1)
        fh_2 = fh_2.view(b, -1)
        return f1_part, f1, f2_part, f2, fh_3, fh_2, out_coarse, out_detail





