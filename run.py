import os, random
import numpy as np
import torch
import torch.nn as nn
import argparse

from utils import logger
from utils.hparams import HParams
from utils.utils1 import make_save_dir, get_optimizer
from loss import FocalLoss,get_cross_entorpy
from dataset import get_loader
from model import ChordConditionedMelodyTransformer as CMT
from trainer import CMTtrainer

# hyperparameter - using argparse and parameter module
parser = argparse.ArgumentParser() #创建argparse对象
parser.add_argument('--idx', type=int, help='experiment number',  default=0)
parser.add_argument('--gpu_index', '-g', type=int, default="0", help='GPU index')
parser.add_argument('--ngpu', type=int, default=0, help='0 = CPU.')
parser.add_argument('--optim_name', type=str, default='adam')
parser.add_argument('--restore_epoch', type=int, default=-1)
parser.add_argument('--load_rhythm', dest='load_rhythm', action='store_true')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()#从键盘上读取上述变量

use_cuda = torch.cuda.is_available()#cuda是否可用(是否可用GPU)
device = torch.device("cuda:%d" % args.gpu_index if use_cuda else "cpu")

hparam_file = os.path.join(os.getcwd(), "hparams1.yaml")#使用hparam调取超参数文件

config = HParams.load(hparam_file) #提取超参数文件内容到config对象中
data_config = config.data_io #将数据集路径和数据处理超参数读取到data_config中
model_config = config.model #将模型的超参数读取到model_config中
exp_config = config.experiment #将训练的超参数读取到exp_config中

# configuration 应该是调试作用？？？
asset_root = config.asset_root#读取asset_root变量
asset_path = os.path.join(asset_root, 'idx%03d' % args.idx)
make_save_dir(asset_path, config) #把config重新写进了hparams.yaml（在asset_root里面）
logger.logging_verbosity(1)#设置消息的阚值
logger.add_filehandler(os.path.join(asset_path, "log1.txt"))#设置将信息保存在log.txt中

# seed
if args.seed > 0:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

# get dataloader for training
logger.info("get loaders")
#把数据加载成dataloader类型，get_loader函数首先通过Dataset类型读取数据再用Dataloader的构造方法转换为dataloader对象
#loader()
train_loader = get_loader(data_config, mode='train')
eval_loader = get_loader(data_config, mode='eval')
test_loader = get_loader(data_config, mode='test')

# build graph, criterion and optimizer
logger.info("build graph, criterion, optimizer and trainer")
model = CMT(**model_config)#创建模型对象

if args.ngpu > 1:
    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
model.to(device)

nll_criterion = nn.NLLLoss().to(device)
pitch_criterion = FocalLoss(gamma=2).to(device)
pitch1_criterion=get_cross_entorpy()
criterion = (nll_criterion, pitch_criterion,pitch1_criterion)

if args.load_rhythm:
    rhythm_params = list()
    pitch_params = list()
    param_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    for name, param in param_model.named_parameters():
        if 'rhythm' in name:
            rhythm_params.append(param)
        else:
            pitch_params.append(param)
    rhythm_param_dict = {'params': rhythm_params, 'lr': 1e-6}
    pitch_param_dict = {'params': pitch_params}
    params = [rhythm_param_dict, pitch_param_dict]
else:
    params = model.parameters()#params为模型中可优化的参数

optimizer = get_optimizer(params, config.experiment['lr'],
                          config.optimizer, name=args.optim_name)

def get_params(s):
    params = []
    param_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    for i in s:
        for name, param in param_model.named_parameters():
            if i in name:
                params.append(param)
            # else:
            #     param.requires_grad=False
    # for name, param in param_model.named_parameters():
    #     if 'rhythm' in name and 'rhythm1' not in name:
    #         params.append(name)
    #     if 'pitch' in name and 'pitch1' not in name:
    #         params.append(name)
    # return params
params=model.parameters()
#params = model.parameters()
ac_optimizer=get_optimizer(params, config.experiment['ac_lr'],
                          config.optimizer, name=args.optim_name)
# get trainer
trainer = CMTtrainer(asset_path, model, criterion, optimizer,ac_optimizer,
                     train_loader, eval_loader, test_loader,
                     device, exp_config)

# start training - add additional train configuration
logger.info("start training")
#trainer.train(restore_epoch=args.restore_epoch,
#               load_rhythm=args.load_rhythm)
trainer._sampling(90)
