
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# input size: (b, 64)
class GlobalinfolossNet256(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(256*2, 128, kernel_size=1, bias=False)
        self.c2 = nn.Conv1d(128, 64, kernel_size=1, bias=False)
        self.c3 = nn.Conv1d(64, 32, kernel_size=1, bias=False)
        self.l0 = nn.Linear(32, 1)
    
    def forward(self, x_global, c):
        # input size: (b, 256)
        # x_global = b*256   c = b*256
        xx = torch.cat((x_global, c), dim = 1)  # -> (b, 256*2)
        h = xx.unsqueeze(dim=2) # -> (b, 256*2, 1)
        h = F.relu(self.c1(h)) # -> (b, 128, 1)
        h = F.relu(self.c2(h)) # -> (b, 64, 1)
        h = F.relu(self.c3(h)) # -> (b, 32, 1)
        h = h.view(h.shape[0], -1) # (b, 32)
        return self.l0(h)  # b*1

class GlobalinfolossNet1024(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(1024*2, 128, kernel_size=1, bias=False)
        self.c2 = nn.Conv1d(128, 64, kernel_size=1, bias=False)
        self.c3 = nn.Conv1d(64, 32, kernel_size=1, bias=False)
        self.l0 = nn.Linear(32, 1)
    
    def forward(self, x_global, c):
        # input size: (b, 64)
        # x_global = b*64   c = b*64
        xx = torch.cat((x_global, c), dim = 1)  # -> (b, 128)
        h = xx.unsqueeze(dim=2) # -> (b, 128, 1)
        h = F.relu(self.c1(h)) # -> (b, 128, 1)
        h = F.relu(self.c2(h)) # -> (b, 64, 1)
        h = F.relu(self.c3(h)) # -> (b, 32, 1)
        h = h.view(h.shape[0], -1) # (b, 32)
        return self.l0(h)  # b*1

class LocalinfolossNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(256*2, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 1, kernel_size=1, bias=False)
    
    def forward(self, x_local, c):
        # x_local: b* 64* n
        # c : b* 64* n
        xx = torch.cat((x_local, c), dim=1) # -> (b, 128, num_points)
        h = F.relu(self.conv1(xx))  # (b, 128, num_points) -> (b, 64, num_points)
        h = F.relu(self.conv2(h)) #(b, 64, num_points) -> (b, 64, num_points)
        h = F.relu(self.conv3(h))  # (b, 64, num_points) -> (b, 1, num_points)
        h = h.view(h.shape[0], -1) # (b, num_points)
        return h # (b, num_points)

class PriorDiscriminator256(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(256, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))  # -> (b, 1000)
        h = F.relu(self.l1(h))  # -> (b, 200)
        return torch.sigmoid(self.l2(h))    # b*1

class PriorDiscriminator1024(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(1024, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))    

class DeepMILoss256(nn.Module):
    def __init__(self):
        super().__init__()

        self.globalinfo = GlobalinfolossNet256()
        # self.localinfo = LocalinfolossNet()
        self.priorinfo = PriorDiscriminator256()
        
   
    def forward(self, part_feature_g, gt_feature_g):
        ### generate sample pairs
        # feature_g (B, C)
        part_feature_gp = part_feature_g[torch.randperm(part_feature_g.size(0))]
        # f2_gt_prime = f2_gt[torch.randperm(f2_gt.size(0))]        
        # Ej = -F.softplus(-self.localinfo(c, x_local)).mean() # positive pairs
        # Em = F.softplus(self.localinfo(c, x_local_prime)).mean() # negetive pairs
        # LOCAL = (Em - Ej) * 0.5
        ###### global loss ###########
        Ej = -F.softplus(-self.globalinfo(gt_feature_g, part_feature_g)).mean() # positive pairs
        Em = F.softplus(self.globalinfo(gt_feature_g, part_feature_gp)).mean() # negetive pairs
        GLOBAL = (Em - Ej) * 1.0


        ###### prior #################
        prior = torch.rand_like(gt_feature_g)
        term_a = torch.log(self.priorinfo(prior)).mean()
        term_b = torch.log(1.0 - self.priorinfo(gt_feature_g)).mean()
        PRIOR = - (term_a + term_b) * 0.1
        ######### combine global and prior loss ###############
        ToT = GLOBAL + PRIOR

        # return GLOBAL, PRIOR, ToT # tensor, a value
        return ToT

class DeepMILoss1024(nn.Module):
    def __init__(self):
        super().__init__()

        self.globalinfo = GlobalinfolossNet1024()
        # self.localinfo = LocalinfolossNet()
        self.priorinfo = PriorDiscriminator1024()
        
   
    def forward(self, part_feature_g, gt_feature_g):
        ### generate sample pairs
        # feature_g (B, C)
        part_feature_gp = part_feature_g[torch.randperm(part_feature_g.size(0))]
        # f2_gt_prime = f2_gt[torch.randperm(f2_gt.size(0))]        
        # Ej = -F.softplus(-self.localinfo(c, x_local)).mean() # positive pairs
        # Em = F.softplus(self.localinfo(c, x_local_prime)).mean() # negetive pairs
        # LOCAL = (Em - Ej) * 0.5
        ###### global loss ###########
        Ej = -F.softplus(-self.globalinfo(gt_feature_g, part_feature_g)).mean() # positive pairs
        Em = F.softplus(self.globalinfo(gt_feature_g, part_feature_gp)).mean() # negetive pairs
        GLOBAL = (Em - Ej) * 1.0


        ###### prior #################
        prior = torch.rand_like(gt_feature_g)
        term_a = torch.log(self.priorinfo(prior)).mean()
        term_b = torch.log(1.0 - self.priorinfo(gt_feature_g)).mean()
        PRIOR = - (term_a + term_b) * 0.1
        ######### combine global and prior loss ###############
        ToT = GLOBAL + PRIOR

        # return GLOBAL, PRIOR, ToT # tensor, a value
        return ToT