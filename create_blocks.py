import argparse
import torch
import ujson
from SBIBD import create_blocks
import os

def create(args):
    blocks = []
    source_path = os.path.dirname((os.path.abspath(__file__)))

    if "MNIST" == args.dataset:
        file_path = os.path.join(source_path, "MNIST_SBIBD.json")
    elif "FEMNIST" == args.dataset:
        file_path = os.path.join(source_path, "FEMNIST_SBIBD.json")
    elif "Cifar100" == args.dataset:
        file_path = os.path.join(source_path, "Cifar100_SBIBD.json")
    elif "AGNews" == args.dataset:
        file_path = os.path.join(source_path, "AGNews_SBIBD.json")
    else:
        pass

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"There is no {file_path}")
    else:
        with open(file_path, "r") as f:
            data = ujson.load(f)

    #create blocks for all SBIBD
    print(data["C_SBIBD"])
    for i in range(len(data["C_SBIBD"])):
        if data["C_SBIBD"][i] != []:
            blocks.append(create_blocks(data["C_SBIBD"][i], data["a"][i], data["k"][i]))
        else:
            blocks.append([])

    print(f"The blocks are \n{blocks}")

    #save the file
    save_path = os.path.dirname(os.path.abspath(__file__))

    if "MNIST" == args.dataset:
        with open(os.path.join(save_path, "MNIST_blocks.json"), "w") as f:
            ujson.dump(blocks, f)
    elif "FEMNIST" == args.dataset:
        with open(os.path.join(save_path, "FEMNIST_blocks.json"), "w") as f:
            ujson.dump(blocks, f)
    elif "Cifar100" == args.dataset:
        with open(os.path.join(save_path, "Cifar100_blocks.json"), "w") as f:
            ujson.dump(blocks, f)
    elif "AGNews" == args.dataset:
        with open(os.path.join(save_path, "AGNews_blocks.json"), "w") as f:
            ujson.dump(blocks, f)
    else:
        pass

    return blocks

if __name__ == "__main__":
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()

        parser.add_argument('-data', "--dataset", type=str, default="AGNews")
        parser.add_argument('-nc', "--num_clients", type=int, default=30,
                            help="Total number of clients")
        parser.add_argument('-alp', "--alpha", type=float, default=0.9)
        parser.add_argument('-tem', "--tem", type=int, default=3)
        parser.add_argument('-dev', '--device', type=str, default="cuda", choices=["cpu", "cuda"])
        parser.add_argument('-did', "--device_id", type=str, default="0")
        parser.add_argument('-mo', "--mode", type=str, default="pre_train")
        parser.add_argument('-ncl', "--num_classes", type=int, default=10)
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
        parser.add_argument('-vs', "--vocab_size", type=int, default=80,
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

        create(args)

