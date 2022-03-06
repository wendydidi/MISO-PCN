import argparse
# import imp
import os
from sys import dont_write_bytecode
from numpy.lib.function_base import average, delete

import torch.utils.data as Data
import torch.optim as optim
import torch
import torch.nn as nn
# import h5py
# from yaml.events import NodeEvent
# import encoding
# from encoding.parallel import DataParallelModel, DataParallelCriterion

from models import AutoEncoder_gt, AutoEncoder_part, AutoEncoder_part_MI
# from loss import CD, ChamferDistance
import loss
from MI import DeepMILoss256, DeepMILoss1024
# from MInew import DeepMILoss256, DeepMILoss1024
import utils
from utils import PointLoss, PointLoss_test
import shapenet_part_loader
# import data

import time

from torch.utils.tensorboard import SummaryWriter



##############################################################################################
# class H5Dataset(torch.utils.data.Dataset):
#     def __init__(self, path):
#         self.file_path = path
#         self.dataset = None
#         with h5py.File(self.file_path, 'r') as file:
#             self.dataset_len = len(file["dataset"])
 
#     def __getitem__(self, index):
#         if self.dataset is None:
#             self.dataset = h5py.File(self.file_path, 'r')["dataset"]
#         return self.dataset[index]
 
#     def __len__(self):
#         return self.dataset_len

## args
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=bool, default=True, help='enables GUDA training')
parser.add_argument('--batch_size', type= int, default=32, help='Size of batch)')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--model_gt', type=str, 
                    default='/root/wangdi/completionwithmmi/Autoencoder_GT_model/lowest_loss_50.pth')
parser.add_argument('--model_partial', type=str, default=None)
parser.add_argument('--epochs', type= int, default=250)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--pnum', type=int, default=2048)
parser.add_argument('--crop_point_num', type=int, default=512)
parser.add_argument('--cropmethod', default = 'random_center', help = 'random|center|random_center')
# parser.add_argument('--log_dir', type=str, default='logs/')
parser.add_argument('--exp_name', type=str, default='exp')
parser.add_argument('--loss_type', type=str, default='MI', help='MI|L1|L2')
parser.add_argument('--withMMI', type=bool, default=True, help='with MMI Module: True or False')
args = parser.parse_args()
###############################################################################################
# used to save tensorboard
os.makedirs('./tbrecord/' + args.exp_name, exist_ok=True)
logdir = os.path.join('./tbrecord/' + args.exp_name)
writer = SummaryWriter(logdir)

os.environ["NCCL_DEBUG"] = "INFO"

## use gpus
device = torch.device("cuda" if args.cuda else "cpu")

##############################################################################################
## dataset
class_choice = [ 'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Guitar', 'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol',  'Skateboard', 'Table']

dset = shapenet_part_loader.PartDataset(root='/root/lu/project/ZA/PFNET/PFNet/dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/',
                                        classification=True, class_choice=class_choice, 
                                        npoints=args.pnum, split='train')
assert dset
train_dataloader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=int(args.num_workers))

length_traindata = len(train_dataloader)
print('the number of train data is ', length_traindata)
test_dset = shapenet_part_loader.PartDataset(root='/root/lu/project/ZA/PFNET/PFNet/dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/',
                                             classification=True, class_choice=class_choice, 
                                             npoints=args.pnum, split='test')
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=int(args.num_workers))
length_testdata = len(test_dataloader)
print('the number of test data is ', length_testdata)
###############################################################################################
def train(args): 
    ## load pretrained and freeze gt model

    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

    gt_model = AutoEncoder_gt().to(device) # load gt encoder
    if args.withMMI:
        part_model =AutoEncoder_part_MI().to(device)
    else:
        part_model = AutoEncoder_part().to(device)

    if torch.cuda.device_count() > 1:
        # gt_model = nn.DataParallel(gt_model)
        part_model = nn.DataParallel(part_model)
    if args.model_gt is not None:
        print('Loading trained GT model from {}'.format(args.model_gt))
        # gt_state_dict = torch.load(args.model_gt)
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k,v in gt_state_dict.items():
        #     name = k.replace('', 'module.')
        #     new_state_dict[name] = v
        # gt_model.load_state_dict(new_state_dict)
        gt_model.load_state_dict(torch.load(args.model_gt))
    
    if args.model_partial is not None:
        print('Loading trained model from {}'.format(args.model_partial))
        part_model.load_state_dict(torch.load(args.model_partial))
    else:
        print("\033[1;31;40m Train a new model. Let's training! \033[0m")

    # if torch.cuda.device_count()>1:
    #     gt_model = nn.DataParallel(gt_model)
    #     part_model = nn.DataParallel(part_model)
    # gt_model.to(device)
    # part_model.to(device)

    optimizer = optim.Adam(part_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    #### initial loss function #####
    criterion_PointLoss_test = PointLoss_test().to(device)
    criterion_PointLoss = PointLoss().to(device)  
    criterion_PointLoss = nn.DataParallel(criterion_PointLoss)
    criterion_PointLoss_test = nn.DataParallel(criterion_PointLoss_test)

    # l1loss = loss.l1loss()
    # l2loss = loss.l2loss()
    # MIloss256 = DeepMILoss256().to(device)
    # MIloss1024 = DeepMILoss1024().to(device)  
    ###################################################
    max_iter = int(len(dset) / args.batch_size + 0.5)
    minimum_loss = 1e4
    best_epoch = 0
    #############################################################
    # start = time.time()
    for epoch in range(1, args.epochs):
        ## training 
        part_model.train()
        cdloss = 0.0
        infoLoss = 0.0
        distloss1 = 0.0
        distloss2 = 0.0
        iter_count = 0  
        start = time.time()  
        for i, data in enumerate(train_dataloader, 1):
            # gt_pc, label = data # gt_pc (32, 2048, 3) --(B, N, C)
            # gt_pc = gt_pc.to(device)
            # gt_pc = gt_pc.permute(0, 2, 1)   # 32, 3, 2048
            # # print(gt_pc.size())
            #########################################################
            # get partial input
            partial_input, coarse_gt, dense_gt, real_center, target = utils.get_batch_data(data, device, args)
            # print('Partial input ', partial_input)
            # partial_input (b, c, n)
            # coarse_gt / dense_gt (b, 3, n)
            optimizer.zero_grad()
            #########################################################
            # get grouth truth features
            f1_gt_g, f1_gt_l, f2_gt_g, f2_gt_l, y_coarse, y_dense = gt_model(dense_gt)
            # del f1_gt_l, f2_gt_l, y_coarse, y_dense
            #########################################################
            #########################################################
            # get partial features
            if args.withMMI:
                part_f1_g, part_f1_l, part_f2_g, part_f2_l, fh_l, fh_g, out_coarse, out_dense = part_model(partial_input)
            else:
                part_f1_g, part_f1_l, part_f2_g, part_f2_l, out_coarse, out_dense = part_model(partial_input)
       
            # print('part_f1_g', part_f1_g.size()) # part_f1_g (b, c1, 1)
            # print('part_f1_l', part_f1_l.size()) # part_f1_l (b, c1, 1536)
            ### CD LOSS ####
            dense_gt = dense_gt.permute(0, 2, 1)
            coarse_gt = coarse_gt.permute(0, 2, 1)
            out_coarse = out_coarse.permute(0, 2, 1)
            out_dense = out_dense.permute(0, 2, 1)  # b, n, 3
            # print(out_dense.size())
            # CDLOSS = criterion_PointLoss(out_dense, dense_gt)
            # CDLOSS1 = criterion_PointLoss(out_coarse, coarse_gt)
            # CD_LOSS = args.alpha * CDLOSS + CDLOSS1
            CD_LOSS = args.alpha * criterion_PointLoss(out_dense, dense_gt) + criterion_PointLoss(out_coarse, coarse_gt)
            # print('CDLOSS', CDLOSS.size())
            # print('CD_LOSS', CD_LOSS)
            CDLOSS = CD_LOSS.mean()
            # print('CDLOSS', CDLOSS)
            ##### choose different loss ############
            if args.loss_type == 'MI':
                MIloss256 = DeepMILoss256().to(device)
                MIloss1024 = DeepMILoss1024().to(device)      
                # gmiloss1, pmiloss1, totmiloss1 = MIloss256(part_f1_g, f1_gt_g)
                # gmiloss2, pmiloss2, totmiloss2 = MIloss1024(part_f2_g, f2_gt_g)
                if args.withMMI:
                    totmiloss1 = MIloss256(part_f1_g, f1_gt_g)
                    totmiloss2 = MIloss1024(part_f2_g, f2_gt_g)
                    mmiloss = MIloss256(part_f1_g, fh_g)   #################
                    MI_LOSS = totmiloss1 + totmiloss2 + mmiloss
                else:
                    totmiloss1 = MIloss256(part_f1_g, f1_gt_g)
                    totmiloss2 = MIloss1024(part_f2_g, f2_gt_g)
                    MI_LOSS = totmiloss1 + totmiloss2 
                # print('MI-LOSS', MI_LOSS)
                # print('MI_LOSS', MI_LOSS.size())
                (MI_LOSS + CDLOSS).backward()  
                optimizer.step()     
                cdloss += CDLOSS.item()/length_traindata 
                infoLoss += MI_LOSS.item()/length_traindata     
                # cdloss = cdloss + CDLOSS/length_traindata
                # infoLoss = infoLoss + MI_LOSS/length_traindata

            if args.loss_type == 'L1':
                if args.withMMI:
                    MIloss256 = DeepMILoss256().to(device)
                    mmiloss = MIloss256(part_f1_g, fh_g)
                    L1loss256 = loss.l1loss(part_f1_g, f1_gt_g).to(device)
                    L1loss1024 = loss.l1loss(part_f2_g, f2_gt_g).to(device)
                    L1_LOSS = L1loss256 + L1loss1024
                    COMLOSS = mmiloss + L1loss256 + L1loss1024
                else:
                    L1loss256 = loss.l1loss(part_f1_g, f1_gt_g).to(device)
                    L1loss1024 = loss.l1loss(part_f2_g, f2_gt_g).to(device)
                    L1_LOSS = L1loss256 + L1loss1024
                    COMLOSS = L1_LOSS

                (COMLOSS + CDLOSS).backward()
                optimizer.step()
                cdloss += CDLOSS.item() / length_traindata
                distloss1 += L1_LOSS.item() / length_traindata
                # cdloss = cdloss + CDLOSS/length_traindata
                # distloss1 = distloss1 + L1_LOSS/length_traindata               

            if args.loss_type == 'L2':
                if args.withMMI:
                    MIloss256 = DeepMILoss256().to(device)
                    mmiloss = MIloss256(part_f1_g, fh_g)
                    L2loss256 = loss.l2loss(part_f1_g, f1_gt_g).to(device)
                    L2loss1024 = loss.l2loss(part_f2_g, f2_gt_g).to(device)
                    L2_LOSS = L2loss256 + L2loss1024
                    COMLOSS = mmiloss + L2loss256 + L2loss1024
                else:
                    L2loss256 = loss.l2loss(part_f1_g, f1_gt_g).to(device)
                    L2loss1024 = loss.l2loss(part_f2_g, f2_gt_g).to(device)
                    L2_LOSS = L2loss256 + L2loss1024
                    COMLOSS = L2_LOSS

                (COMLOSS + CDLOSS).backward()
                optimizer.step()
                cdloss += CDLOSS.item() / length_traindata
                distloss2 += L2_LOSS.item() / length_traindata
                # cdloss = cdloss + CDLOSS/length_traindata
                # distloss2 = distloss2 + L2_LOSS/length_traindata

            iter_count += 1
            if i%100 == 0:
                print("Training epoch {}/{}, iteration {}/{}: CDloss is {:.6f}".format(epoch, 
                        args.epochs, i, max_iter, cdloss))
        scheduler.step()   
        end = time.time()
        epoch_time = end - start
        avgcdloss = cdloss/iter_count

        if args.loss_type == 'MI':
            avginfoLoss = infoLoss/iter_count
            print("Training epoch {}/{}, Average Train CD Loss = {:.6f}, Average Train MIloss = {:.6f}, training time per epoch is {}".format(
                    epoch, args.epochs, avgcdloss, avginfoLoss, epoch_time))
            writer.add_scalar('Epoch_Time', epoch_time, epoch)
            writer.add_scalar('Average_Train_CD_Loss', avgcdloss, epoch)
            writer.add_scalar('Average_Train_MIloss', avginfoLoss, epoch)

        if args.loss_type == 'L1':
            avgdistloss1 = distloss1/iter_count
            print("Training epoch {}/{}, Average Train CD Loss = {:.4f}, Average Train L1loss = {:.4f}, training time per epoch is {}".format(
                epoch, args.epochs, avgcdloss, avgdistloss1, epoch_time))
            writer.add_scalar('Epoch_Time', epoch_time, epoch)
            writer.add_scalar('Average_Train_CD_Loss', avgcdloss, epoch)
            writer.add_scalar('Average_Train_L1_Loss', avgdistloss1, epoch)

        if args.loss_type == 'L2':
            avgdistloss2 = distloss2/iter_count
            print("Training epoch {}/{}, Average Train CD Loss = {:.4f}, Average Train L2loss = {:.4f}, training time per epoch is {}".format(
                epoch, args.epochs, avgcdloss, avgdistloss2, epoch_time))
            writer.add_scalar('Epoch_Time', epoch_time, epoch)
            writer.add_scalar('Average_Train_CD_Loss', avgcdloss, epoch)
            writer.add_scalar('Average_Train_L2_Loss', avgdistloss2, epoch)               
    #############################################################
    ## evaluation
        eval_cdloss = 0.0
        eval_cdloss1 = 0.0
        eval_cdloss2 = 0.0
        G_P = 0.0
        G_P1 = 0.0
        G_P2 = 0.0
        P_G = 0.0
        P_G1 = 0.0
        P_G2 = 0.0
        eval_infoloss = 0.0
        eval_distloss1 = 0.0
        eval_distloss2 = 0.0
        ite_count = 0
        part_model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_dataloader, 1):
                partial_input, coarse_gt, dense_gt, real_center, target = utils.get_batch_data(data, device, args)
                if args.withMMI:
                    part_f1_g, part_f1_l, part_f2_g, part_f2_l, fh_l, fh_g, out_coarse, out_dense = part_model(partial_input)
                else:
                    part_f1_g, part_f1_l, part_f2_g, part_f2_l, out_coarse, out_dense = part_model(partial_input)
                
                f1_gt_g, f1_gt_l, f2_gt_g, f2_gt_l, y_coarse, y_dense = gt_model(dense_gt)

                ### COMPUTE CD LOSS ####
                dense_gt = dense_gt.permute(0, 2, 1)
                coarse_gt = coarse_gt.permute(0, 2, 1)
                out_coarse = out_coarse.permute(0, 2, 1)
                out_dense = out_dense.permute(0, 2, 1)
                # CDLOSS, GT_Pre, Pre_GT = criterion_PointLoss_test(out_coarse, coarse_gt) + 0.1*criterion_PointLoss_test(out_dense, dense_gt)
                CDLOSS1, GT_Pre1, Pre_GT1 = criterion_PointLoss_test(out_coarse, coarse_gt)
                CDLOSS2, GT_Pre2, Pre_GT2 = criterion_PointLoss_test(out_dense, dense_gt)
                CDLOSS1 = CDLOSS1.mean()
                GT_Pre1 = GT_Pre1.mean()
                Pre_GT1 = Pre_GT1.mean()
                CDLOSS2 = CDLOSS2.mean()
                GT_Pre2 = GT_Pre2.mean()
                Pre_GT2 = Pre_GT2.mean()
                # CDLOSS1 = criterion_PointLoss_test(out_coarse, coarse_gt)

                # CDLOSS = cdloss.cpu().detach().numpy()
                # GT_Pre = GT_Pre().cpu().detach().numpy()
                # Pre_GT = Pre_GT().cpu().detach().numpy()
                # print('CDLOSS', CDLOSS.size())
                # eval_cdloss = eval_cdloss + CDLOSS/length_testdata
                # G_P = G_P + GT_Pre/length_testdata
                # P_G = P_G + Pre_GT/length_testdata
                eval_cdloss1 += CDLOSS1.item() / length_testdata
                G_P1 += GT_Pre1.item() / length_testdata
                P_G1 += Pre_GT1.item() / length_testdata

                eval_cdloss2 += CDLOSS2.item() / length_testdata
                G_P2 += GT_Pre2.item() / length_testdata
                P_G2 += Pre_GT2.item() / length_testdata

                eval_cdloss = eval_cdloss1 + args.alpha * eval_cdloss2
                G_P = G_P1 + args.alpha * G_P2
                P_G = P_G1 + args.alpha * P_G2
                if i % 20 == 0:
                    print('[%d/%d][%d/%d] CD_LOSS: %.4f GT_Pre: %.4f Pre_GT: %.4f' 
                            % (epoch, args.epochs, i, length_testdata, eval_cdloss, G_P, P_G))
            
                if args.loss_type == 'MI':
                    # gmiloss1, pmiloss1, totmiloss1 = MIloss256(part_f1_g, f1_gt_g)
                    # gmiloss2, pmiloss2, totmiloss2 = MIloss1024(part_f2_g, f2_gt_g)
                    totmiloss1 = MIloss256(part_f1_g, f1_gt_g)
                    totmiloss2 = MIloss1024(part_f2_g, f2_gt_g)
                    MI_LOSS = totmiloss1 + totmiloss2
                    eval_infoloss = eval_infoloss + MI_LOSS/length_testdata

                if args.loss_type == 'L1':
                    L1loss256 = loss.l1loss(part_f1_g, f1_gt_g)
                    L1loss1024 = loss.l1loss(part_f2_g, f2_gt_g)
                    L1_LOSS = L1loss256 + L1loss1024
                    eval_distloss1 = eval_distloss1 + L1_LOSS/length_testdata

                if args.loss_type == 'L2':
                    L2loss256 = loss.l2loss(part_f1_g, f1_gt_g)
                    L2loss1024 = loss.l2loss(part_f2_g, f2_gt_g)
                    L2_LOSS = L2loss256 + L2loss1024                      
                    eval_distloss2 = eval_distloss2 + L2_LOSS/length_testdata

                ite_count += 1

            avgcdloss = eval_cdloss / ite_count                                

            if args.loss_type == 'MI':
                avginfoLoss = eval_infoloss / ite_count
                print("Validation epoch {}/{}, CDL0SS is {:6f}, MILOSS is {:6f}".format(epoch, args.epochs, avgcdloss, avginfoLoss))
                writer.add_scalar('Average_Test_CD_Loss', avgcdloss, epoch)
                writer.add_scalar('Average_Test_MILoss', avginfoLoss, epoch)

            if args.loss_type == 'L1':
                avgdistloss1 = eval_distloss1 / ite_count
                print("Validation epoch {}/{}, CDLOSS is {:6f}, L1lOSS is {:6f}".format(epoch, args.epochs, avgcdloss, avgdistloss1))
                writer.add_scalar('Average_Test_CD_Loss', avgcdloss, epoch)
                writer.add_scalar('Average_Test_L1_Loss', avgdistloss1, epoch)

            if args.loss_type == 'L2':
                avgdistloss2 = eval_distloss2 / ite_count
                print("Validation epoch {}/{}, CDLOSS is {:6f}, L2LOSS is {:6f}".format(epoch, args.epochs, avgcdloss, avgdistloss2))
                writer.add_scalar('Average_Test_CD_Loss', avgcdloss, epoch)
                writer.add_scalar('Average_Test_L2_Loss', avgdistloss2, epoch)

            ## save the best model and epoch
            torch.save(part_model.state_dict(), logdir + '/current_model_{}.pth'.format(epoch))
            if avgcdloss < minimum_loss:
                best_epoch = epoch
                minimum_loss = avgcdloss
                torch.save(part_model.state_dict(), logdir + '/lowest_loss_{}.pth'.format(epoch))

            print("\033[34mBest model (lowest loss) in epoch {}\033[0m".format(best_epoch))

###############################################################################################
if __name__== "__main__":
    train(args)