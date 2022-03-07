# MISO-PCN
This is the code for Pytorch implementation of the paper - Maximizing Mutual Information based Similarity Operation for Point Cloud Completion Netowrk

### Dataset ###
download from https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip

../dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/02691156...

### training ###
python train_new.py --exp_name=XX --loss_type=XX --lr=XX --withMMI=True/False --model_gt=XX

python train_new.py --exp_name= --loss_type=XX --lr=XX --withMMI=True/False --model_gt=XX

## Note 
1. please change the root of groud truth autoencoder. "--model_gt"
2. please change the type of similarity operation. "--loss_type"

## Acknowlegments
https://github.com/qinglew/PCN-PyTorch.git

PCN: Point Cloud Completion Network
https://arxiv.org/pdf/1808.00671.pdf
