import argparse
import torch
import ujson
import time
from SBIBD import create_blocks
from clients.clientbase import Client
from clients.clientavg import clientAVG
from server.serveravg import FedAvg
import numpy as np
from read_data import read_client_data
import os
from models.MNIST import MNIST_models
from models.FEMNIST import FEMNIST_models
from models.AGNews import AGNews_models
from models.Cifar100 import Cifar100_models
from experiment.distill import Distill

def block_training(args):
    time_list = []
    c_rest = []
    block_models = []
    clients = [_ for _ in range(args.num_clients)]
    source_path = os.path.dirname((os.path.abspath(__file__)))

    if "MNIST" == args.dataset:
        file_path_1 = os.path.join(source_path, "MNIST_blocks.json")
        file_path_2 =os.path.join(source_path, "MNIST_SBIBD.json")
    elif "FEMNIST" == args.dataset:
        file_path_1 = os.path.join(source_path, "FEMNIST_blocks.json")
        file_path_2 = os.path.join(source_path, "FEMNIST_SBIBD.json")
    if "Cifar100" == args.dataset:
        file_path_1 = os.path.join(source_path, "Cifar100_blocks.json")
        file_path_2 =os.path.join(source_path, "Cifar100_SBIBD.json")
    elif "AGNews" == args.dataset:
        file_path_1 = os.path.join(source_path, "AGNews_blocks.json")
        file_path_2 = os.path.join(source_path, "AGNews_SBIBD.json")
    else:
        pass

    if not os.path.exists(file_path_1):
        raise FileNotFoundError(f"There is no {file_path_1}")
    else:
        with open(file_path_1, "r") as f:
            blocks = ujson.load(f)

    if not os.path.exists(file_path_2):
        raise FileNotFoundError(f"There is no {file_path_2}")
    else:
        with open(file_path_2, "r") as f:
            data = ujson.load(f)
    print(data['C_SBIBD'])
    #Block training
    #Set up each client in SBIBD
    for i in range(len(blocks)):
        if "MNIST" == args.dataset and i < len(blocks)/3:
            args.model = MNIST_models.MNIST_MCLR().to(args.device)
        elif "MNIST" == args.dataset and i > len(blocks)/3:
            args.model = MNIST_models.MNIST_MLP().to(args.device)
        elif "MNIST" == args.dataset:
            args.model = MNIST_models.MNIST_MLP().to(args.device)
        elif "FEMNIST" == args.dataset and i < len(blocks)/2:
            args.model = FEMNIST_models.FEMNIST_CNN().to(args.device)
        elif "FEMNIST" == args.dataset and i >= len(blocks)/2:
            args.model = FEMNIST_models.FEMNIST_CNN().to(args.device)
        elif "Cifar100" == args.dataset and i < len(blocks)/2:
            args.model = Cifar100_models.CIFAR100_CNN().to(args.device)
        elif "Cifar100" == args.dataset and i > len(blocks)/2:
            args.model = Cifar100_models.CIFAR100_CNN().to(args.device)
        elif "AGNews" == args.dataset and i < len(blocks)/3:
            args.model = AGNews_models.ResNetText().to(args.device)
        elif "AGNews" == args.dataset and i > len(blocks)/3:
            args.model = AGNews_models.TextCNN().to(args.device)
        elif "AGNews" == args.dataset:
            args.model = AGNews_models.LSTMNet().to(args.device)

        for j in data["C_SBIBD"][i]:
            train_data = read_client_data(args.dataset, j, args, is_train=True)
            test_data = read_client_data(args.dataset,  j, args, is_train=False)
            clients[j] = clientAVG(args, id=j, train_samples=len(train_data), test_samples=len(test_data))

    #Set up the clients in C_rest
        for j in data["C_rest"][i]:
            if "MNIST" == args.dataset:
                args.model = MNIST_models.MNIST_MLP().to(args.device)
            elif "FEMNIST" == args.dataset:
                args.model = FEMNIST_models.FEMNIST_ResNet().to(args.device)
            elif "Cifar100" == args.dataset:
                args.model = Cifar100_models.CIFAR100_CNN().to(args.device)
            elif "AGNews" == args.dataset:
                args.model = AGNews_models.ResNetText().to(args.device)

            train_data = read_client_data(args.dataset, j, args, is_train=True)
            test_data = read_client_data(args.dataset, j, args, is_train=False)
            clients[j] = clientAVG(args, id=j, train_samples=len(train_data), test_samples=len(test_data))
            c_rest.append(j)

    #trainin
    server = FedAvg(args, "1")
    # first round
    print("start training")
    for j in range(len(blocks)):
        for k in range(len(blocks[j])):
            server.selected_clients = [clients[s] for s in server.select_clients(next=blocks[j][k])]
            for client in server.selected_clients:
                client.train()
            server.receive_models()
            server.aggregate_parameters()

            if k+1 < len(blocks[j]):
                server.selected_clients = [clients[s] for s in server.select_clients(next=blocks[j][k+1])]
                server.clients = server.selected_clients
            else:
                pass

            server.send_models()
        block_models.append(server.global_model)

    #reset parameters
    for i in range(len(blocks)):
        for j in data["C_SBIBD"][i]:
            clients[j].loss = Distill(alpha=0.9, tem=2)

        for j in data["C_rest"][i]:
            clients[j].loss = Distill(alpha=0.9, tem=2)

    #C_rest train model
    all_clients = [clients[s] for s in server.select_clients(next=c_rest)]
    if args.strategy == 1:
        server.selected_clients = [clients[s] for s in server.select_clients(next=c_rest)]
        for client in server.selected_clients:
            client.train_1(block_models)
        server.receive_models()
        server.aggregate_parameters()

        block_models=[]
        block_models.append(server.global_model)
    elif args.strategy == 2:
        for j in range(len(all_clients)):
            if j < len(all_clients) - 1:
                server.selected_clients = [all_clients[j]]
                server.clients = [all_clients[j + 1]]
                all_clients[j].train_1(block_models)
                server.receive_models()
                server.aggregate_parameters()
                server.send_models()

            else:
                server.selected_clients = [all_clients[j]]
                all_clients[j].train_1(block_models)
                server.receive_models()
                server.aggregate_parameters()

            block_models = []
            block_models.append(server.global_model)

    print("start ___________________  training")
    #other rounds
    for i in range(args.global_rounds-1):
    #C_SBIBD
        s_t = time.time()
        for j in range(len(blocks)):
            for k in range(len(blocks[j])):
                server.selected_clients = [clients[s] for s in server.select_clients(next=blocks[j][k])]
                for client in server.selected_clients:
                    client.train_1(block_models)
                server.receive_models()
                server.aggregate_parameters()

                if k + 1 < len(blocks[j]):
                    server.selected_clients = [clients[s] for s in server.select_clients(next=blocks[j][k + 1])]
                    server.clients = server.selected_clients
                else:
                    pass
                server.send_models()
            block_models.append(server.global_model)

    #C_rest
        if args.strategy == 1:
            server.selected_clients = [clients[s] for s in server.select_clients(next=c_rest)]
            server.clients = server.selected_clients
            for client in server.selected_clients:
                client.train_1(block_models)
            server.receive_models()
            server.aggregate_parameters()
            server.send_models()

            block_models = []
            block_models.append(server.global_model)

        elif args.strategy == 2:
            for j in range(len(all_clients)):
                if j < len(all_clients) - 1:
                    server.selected_clients = [all_clients[j]]
                    server.clients = [all_clients[j+1]]
                    all_clients[j].train_1(block_models)
                    server.receive_models()
                    server.aggregate_parameters()
                    server.send_models()

                else:
                    server.selected_clients = [all_clients[j]]
                    all_clients[j].train_1(block_models)
                    server.receive_models()
                    server.aggregate_parameters()

                block_models = []
                block_models.append(server.global_model)

        time_list.append(time.time() - s_t)
        print(f"\n-------------Round number: {i+1}-------------")
        print("\nEvaluate global model")
        print('-' * 25, 'time cost', '-' * 25, time_list[-1])

        server.clients = all_clients
        server.evaluate()

    print("\nBest accuracy.")
    print(max(server.rs_test_acc))
    print("\nAverage time cost per round.")
    print(sum(time_list.Budget[1:]) / len(time_list.Budget[1:]))

    server.save_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', "--dataset", type=str, default="Cifar100")
    parser.add_argument('-ncl', "--num_classes", type=int, default=100)
    parser.add_argument('-algo', "--algorithm", type=str, default="Cifar100_GBG_CNN_1")
    parser.add_argument('-tem', "--tem", type=int, default=3)
    parser.add_argument('-lbs', "--batch_size", type=int, default=32)
    parser.add_argument('-alp', "--alpha", type=float, default=0.9)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-st', "--strategy", type=int, default=1)

    parser.add_argument('-m', "--model", type=str, default="_")
    parser.add_argument('-nc', "--num_clients", type=int, default=30,
                        help="Total number of clients")
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=200)

    parser.add_argument('-dev', '--device', type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-mo', "--mode", type=str, default="pre_train")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100,
                        help="For auto_break")
    parser.add_argument('-t', "--times", type=int, default=1,
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
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
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
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")

    block_training(args)
