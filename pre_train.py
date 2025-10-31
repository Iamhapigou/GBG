import argparse
import torch
import ujson
from clients.clientbase import Client
from clients.clientavg import clientAVG
import numpy as np
from pre_mlp import pre_MLP, cifar100_MLP, AGNewsMLP
from Group import Group
from SBIBD import SBIBD
from read_data import read_client_data
from multiprocessing import Pool
import os

def pre_train(args):
    clients = []
    C_rest = []
    C_SBIBD = []
    updates = []
    a = []
    k = []
    idx_clients = np.arange(args.num_clients)
    num_group = 3
    data = {"C_rest": C_rest, "C_SBIBD": C_SBIBD, "a": a, "k": k}

    if "MNIST" == args.dataset:
        args.model = pre_MLP(input_dim=1*28*28, hidden_dim=10, output_dim=args.num_classes).to(args.device)
        args.local_epochs = 30
    elif "FEMNIST" == args.dataset:
        args.model = pre_MLP(input_dim=1*28*28, hidden_dim=50, output_dim=args.num_classes).to(args.device)
        args.local_epochs = 30
    elif "Cifar100" == args.dataset:
        args.model = cifar100_MLP(input_dim=3 * 32 * 32, hidden_dim=100, output_dim=args.num_classes).to(args.device)
        args.local_epochs = 30
    elif "AGNews" == args.dataset:
        args.model = AGNewsMLP(vocab_size=32000, embed_dim=128, hidden_dim=256, num_classes=args.num_classes).to(args.device)
        args.local_epochs = 30
    else:
        pass

    #create the clients
    for i in idx_clients:
        train_data = read_client_data(args.dataset, i, args, is_train=True)
        test_data = read_client_data(args.dataset, i, args, is_train=False)
        client = clientAVG(args, id=i, train_samples=len(train_data), test_samples=len(test_data))
        clients.append(client)

    #torch.cuda.set_device(args.device_id)
    #torch.cuda.set_per_process_memory_fraction(0.25, device=0)
    #with Pool(processes=2) as pool:
    #   pool.map(clientAVG().train(), clients)

    # pretrain
    for client in clients:
        client.train()
        grads = []
        for name, param in client.model.named_parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))

        updates.append(torch.cat(grads))

    #Group
    group = Group(torch.stack(updates, dim=1), idx_clients, num_group)
    result = group.Group()
    result = [[int(x) for x in group] for group in result]
    print(f"result: {result}")

    save_path = os.path.dirname(os.path.abspath(__file__))

    if "MNIST" == args.dataset:
        with open(os.path.join(save_path, "MNIST_result.json"), "w") as f:
            ujson.dump(result, f)
    elif "FEMNIST" == args.dataset:
        with open(os.path.join(save_path, "FEMNIST_result.json"), "w") as f:
            ujson.dump(result, f)
    elif "Cifar100" == args.dataset:
        with open(os.path.join(save_path, "Cifar100_result.json"), "w") as f:
            ujson.dump(result, f)
    elif "AGNews" == args.dataset:
        with open(os.path.join(save_path, "AGNews_result.json"), "w") as f:
            ujson.dump(result, f)
    else:
        pass

    #SBIBD
    for i in range(num_group):
        result_1, result_2, result_3, result_4 = SBIBD(result[i]).create_SBIBD()
        C_rest.append(result_1)
        C_SBIBD.append(result_2.tolist())
        a.append(result_3)
        k.append(result_4)

    print(f"\nC_rest:{C_rest} \nC_SBIBD:{C_SBIBD}")

    #save the results
    if "MNIST" == args.dataset:
        with open(os.path.join(save_path, "MNIST_SBIBD.json"), "w") as f:
            ujson.dump(data, f)
    elif "FEMNIST" == args.dataset:
        with open(os.path.join(save_path, "FEMNIST_SBIBD.json"), "w") as f:
            ujson.dump(data, f)
    elif "Cifar100" == args.dataset:
        with open(os.path.join(save_path, "Cifar100_SBIBD.json"), "w") as f:
            ujson.dump(data, f)
    elif "AGNews" == args.dataset:
        with open(os.path.join(save_path, "AGNews_SBIBD.json"), "w") as f:
            ujson.dump(data, f)
    else:
        pass

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument('-dev', '--device', type=str, default="cuda", choices=["cpu","cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-alp', "--alpha", type=float, default=0.9)
    parser.add_argument('-tem', "--tem", type=int, default=3)
    parser.add_argument('-data', "--dataset", type=str, default="AGNews")
    parser.add_argument('-mo', "--mode", type=str, default="pre_train")
    parser.add_argument('-ncl', "--num_classes", type=int, default=4)
    parser.add_argument('-m', "--model", type=str, default="CNN")
    parser.add_argument('-lbs', "--batch_size", type=int, default=16)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-gr', "--global_rounds", type=int, default=200)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100,
                        help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-nc', "--num_clients", type=int, default=30,
                        help="Total number of clients")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=',getattr(args, arg))
    print("=" * 50)


    pre_train(args)
