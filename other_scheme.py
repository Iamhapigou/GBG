import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
from torch import nn
from experiment.server.serveravg import FedAvg
from experiment.server.serverprox import FedProx
from experiment.server.serverkd import FedKD
from experiment.server.serverda import PFL_DA
from experiment.server.servergroup import FedGroup
from experiment.result_utils import average_data
from models.MNIST import MNIST_models
from models.FEMNIST import FEMNIST_models
from models.AGNews import AGNews_models
from models.Cifar100 import Cifar100_models

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)

class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out

def run(args):
    time_list = []

    #torch.nn.DataParallel(
    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()
        if "MNIST" == args.dataset and args.model == "MLP":
            args.model = MNIST_models.MNIST_MLP().to(args.device)
        elif "MNIST" == args.dataset and args.model == "MCLR":
            args.model = MNIST_models.MNIST_MCLR().to(args.device)
        elif "MNIST" == args.dataset and args.model == "CNN":
            args.model = MNIST_models.MNIST_CNN().to(args.device)
        elif "FEMNIST" == args.dataset and args.model == "MLP":
            args.model = FEMNIST_models.FEMNIST_MLP().to(args.device)
        elif "FEMNIST" == args.dataset and args.model == "CNN":
            args.model = FEMNIST_models.FEMNIST_CNN().to(args.device)
        elif "FEMNIST" == args.dataset and args.model == "ResNet":
            args.model = FEMNIST_models.FEMNIST_ResNet().to(args.device)
        elif "Cifar100" == args.dataset and args.model == "CNN":
            args.model = Cifar100_models.CIFAR100_CNN().to(args.device)
        elif "Cifar100" == args.dataset and args.model == "EfficientNet":
            args.model = Cifar100_models.CIFAR100_EfficientNet().to(args.device)
        elif "Cifar100" == args.dataset and args.model == "ResNet":
            args.model = Cifar100_models.CIFAR100_ResNet18().to(args.device)
        elif "AGNews" == args.dataset and args.model == "CNN":
            args.model = AGNews_models.TextCNN().to(args.device)
        elif "AGNews" == args.dataset and args.model == "LSTM":
            args.model = AGNews_models.LSTMNet().to(args.device)
        elif "AGNews" == args.dataset and args.model == "ResNet":
            args.model = AGNews_models.ResNetText().to(args.device)
        else:
            pass

    '''
    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()
        if "MNIST" == args.dataset and args.model == "MLP":
            args.model = torch.nn.DataParallel(MNIST_models.MNIST_MLP()).to(args.device)
        elif "MNIST" == args.dataset and args.model == "MCLR":
            args.model = torch.nn.DataParallel(MNIST_models.MNIST_MCLR()).to(args.device)
        elif "MNIST" == args.dataset and args.model == "CNN":
            args.model = torch.nn.DataParallel(MNIST_models.MNIST_CNN()).to(args.device)
        elif "FEMNIST" == args.dataset and args.model == "MLP":
            args.model = torch.nn.DataParallel(FEMNIST_models.FEMNIST_MLP()).to(args.device)
        elif "FEMNIST" == args.dataset and args.model == "CNN":
            args.model = torch.nn.DataParallel(FEMNIST_models.FEMNIST_CNN()).to(args.device)
        elif "FEMNIST" == args.dataset and args.model == "ResNet":
            args.model = torch.nn.DataParallel(FEMNIST_models.FEMNIST_ResNet()).to(args.device)
        elif "Cifar100" == args.dataset and args.model == "CNN":
            args.model = torch.nn.DataParallel(Cifar100_models.CIFAR100_CNN()).to(args.device)
        elif "Cifar100" == args.dataset and args.model == "EfficientNet":
            args.model = torch.nn.DataParallel(Cifar100_models.CIFAR100_EfficientNet()).to(args.device)
        elif "Cifar100" == args.dataset and args.model == "MobileNetV2":
            args.model = torch.nn.DataParallel(Cifar100_models.CIFAR100_MobileNetV2()).to(args.device)
        elif "AGNews" == args.dataset and args.model == "CNN":
            args.model = torch.nn.DataParallel(AGNews_models.TextCNN()).to(args.device)
        elif "AGNews" == args.dataset and args.model == "LSTM":
            args.model = torch.nn.DataParallel(AGNews_models.BiLSTM()).to(args.device)
        elif "AGNews" == args.dataset and args.model == "Transformer":
            args.model = torch.nn.DataParallel(AGNews_models.TinyTransformer()).to(args.device)
        else:
            pass
    '''
    #Change the corresponding model according to the Settings
    if args.algorithm == "FedAvg_ResNet":
        server = FedAvg(args, i)

    elif args.algorithm == "FedProx_ResNet":
        server = FedProx(args, i)

    elif args.algorithm == "FedKD_ResNet":
        #resnet18
        #args.head = nn.Linear(512, 100).to(args.device)
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedKD(args, i)

    elif args.algorithm == "PFL_DA_ResNet":
        #resnet18
        #args.head = nn.Linear(512, 100).to(args.device)
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = PFL_DA(args, i)

    elif args.algorithm == "FedGroup_ResNet":
        server = FedGroup(args, i)
    else:
        pass
    print(args.dataset + args.algorithm)

    server.train()

    time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

if __name__ == "__main__":
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()

        parser.add_argument('-data', "--dataset", type=str, default="AGNews")
        parser.add_argument('-m', "--model", type=str, default="ResNet")
        parser.add_argument('-algo', "--algorithm", type=str, default="FedGroup_ResNet")
        parser.add_argument('-ncl', "--num_classes", type=int, default=4)
        parser.add_argument('-lbs', "--batch_size", type=int, default=32)
        parser.add_argument('-gr', "--global_rounds", type=int, default=200)
        parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                            help="Multiple update steps in one local epoch.")

        parser.add_argument('-jr', "--join_ratio", type=float, default= 1,
                            help="Ratio of clients per round")
        parser.add_argument('-nc', "--num_clients", type=int, default=30,
                            help="Total number of clients")
        parser.add_argument('-ng', "--num_groups", type=int, default=3,
                            help="Total number of groups")
        parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,
                            help="Local learning rate")
        parser.add_argument('-alp', "--alpha", type=float, default=0.9)
        parser.add_argument('-tem', "--tem", type=int, default=3)
        parser.add_argument('-dev', '--device', type=str, default="cuda", choices=["cpu", "cuda"])

        parser.add_argument('-did', "--device_id", type=str, default="0,1")
        parser.add_argument('-mo', "--mode", type=str, default="pre_train")
        parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
        parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
        parser.add_argument('-tc', "--top_cnt", type=int, default=100,
                            help="For auto_break")
        parser.add_argument('-mu', "--mu", type=float, default=0.0)
        parser.add_argument('-t', "--times", type=int, default= 1,
                            help="Running times")
        parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                            help="Rounds gap for evaluation")
        parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
        parser.add_argument('-fs', "--few_shot", type=int, default=0)
        parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
        parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
        parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
        parser.add_argument('-go', "--goal", type=str, default="test",
                            help="The goal for this experiment")
        parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                            help="Random ratio of clients per round")
        parser.add_argument('-pv', "--prev", type=int, default=0,
                            help="Previous Running times")
        parser.add_argument('-ab', "--auto_break", type=bool, default=False)
        parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
        parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
        parser.add_argument('-fd', "--feature_dim", type=int, default=512)
        parser.add_argument('-vs', "--vocab_size", type=int, default=32000,
                            help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
        parser.add_argument('-ml', "--max_len", type=int, default=200)
        parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                            help="Rate for clients that train but drop out")
        parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0,
                            help="The rate for slow clients when training locally")
        parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.1,
                            help="The rate for slow clients when sending global model")
        parser.add_argument('-ts', "--time_select", type=bool, default=False,
                            help="Whether to group and select clients at each round according to time cost")
        parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                            help="The threthold for droping slow clients")
        parser.add_argument('-bt', "--beta", type=float, default=0.0)
        parser.add_argument('-lam', "--lamda", type=float, default=0.9,
                            help="Regularization weight")
        parser.add_argument('-K', "--K", type=int, default=5,
                            help="Number of personalized training steps for pFedMe")
        parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                            help="personalized learning rate to caculate theta aproximately using K steps")
        parser.add_argument('-Ts', "--T_start", type=float, default=1.0)
        parser.add_argument('-Te', "--T_end", type=float, default=5.0)

        args = parser.parse_args()

        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

        if args.device == "cuda" and not torch.cuda.is_available():
            print("\ncuda is not avaiable.\n")
            args.device = "cpu"

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")

        run(args)