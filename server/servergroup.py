import copy
import time
from experiment.clients.clientgroup import clientGroup
from experiment.server.serverbase import Server
from threading import Thread
import os
import ujson

class FedGroup(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.num_groups = args.num_groups
        self.path = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
        self.group_model = []
        self.group_weight = []
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientGroup)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        if "MNIST" == self.dataset:
            file_path = os.path.join(self.path, "MNIST_result.json")
        elif "FEMNIST" == self.dataset:
            file_path = os.path.join(self.path, "FEMNIST_result.json")
        elif "AGNews" == self.dataset:
            file_path = os.path.join(self.path, "AGNews_result.json")
        elif "Cifar100" == self.dataset:
            file_path = os.path.join(self.path, "Cifar100_result.json")
        else:
            pass

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"There is no {file_path}")
        else:
            with open(file_path, "r") as f:
                group = ujson.load(f)

        self.current_num_join_clients = 1

        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients(next=[])
            for client in self.selected_clients:
                client.train()

            for j in range(self.num_groups):
                self.selected_clients = [self.clients[s] for s in group[j]]

                self.receive_models()
                self.aggregate_parameters()

                self.group_model.append(copy.deepcopy(self.global_model))
                self.group_weight.append(len(self.selected_clients)/self.num_clients)

            for param in self.global_model.parameters():
                param.data.zero_()

            for w, group_model in zip(self.group_weight, self.group_model):
                self.add_parameters(w, group_model)

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            self.send_models()
            self.group_model = []
            self.group_weight = []
            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientGroup)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()