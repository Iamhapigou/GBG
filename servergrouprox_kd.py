import copy
import time
import torch
import os
import numpy as np
import ujson
import torch.nn as nn
import torch.nn.functional as F
from experiment.clients.clientgrouprox import clientGrouprox
from experiment.server.serverbase import Server
from threading import Thread
from experiment.distill import Distill
from experiment.read_data import PublicDataset
from torch.utils.data import Dataset, DataLoader
from models.MNIST import MNIST_models
from models.FEMNIST import FEMNIST_models
from models.AGNews import AGNews_models
from models.Cifar100 import Cifar100_models

class FedGrouprox_KD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.num_groups = args.num_groups
        self.path = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
        self.group_model = []
        self.group_weight = []
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientGrouprox)

        self.kd_alpha = args.alpha
        self.kd_temp = args.tem
        self.kd_epochs = 1
        self.kd_lr = 0.01
        self.loss = nn.CrossEntropyLoss()

        if args.dataset == "MNIST":
            self.global_model = MNIST_models.MNIST_MLP().to(args.device)
        elif args.dataset == "FEMNIST":
            self.global_model = FEMNIST_models.FEMNIST_ResNet().to(args.device)
        elif args.dataset == "Cifar100" :
            self.global_model = Cifar100_models.CIFAR100_CNN().to(args.device)
        elif args.dataset == "AGNews":
            self.global_model = AGNews_models.TextCNN().to(args.device)
        else:
            pass

        self.public_train_loader = DataLoader(PublicDataset(self.dataset, True, 0), batch_size=self.batch_size,
                                            shuffle=False,  num_workers=4,  pin_memory=True, persistent_workers=True)
        self.public_test_loader = DataLoader(PublicDataset(self.dataset, False, 0), batch_size=self.batch_size,
                                            shuffle=False,  num_workers=4,  pin_memory=True, persistent_workers=True)
        self.kd_loss = Distill(alpha=self.kd_alpha, tem=self.kd_temp)
        self.server_opt = torch.optim.SGD(self.global_model.parameters(), lr=self.kd_lr)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def aggregate_uploaded_models_to_new_model(self):
        assert len(self.uploaded_models) > 0

        agg = copy.deepcopy(self.uploaded_models[0])
        for p in agg.parameters():
            p.data.zero_()

        for w, m in zip(self.uploaded_weights, self.uploaded_models):
            for ap, cp in zip(agg.parameters(), m.parameters()):
                ap.data.add_(cp.data, alpha=w)  # 等价于 ap += w * cp

        return agg

    @torch.no_grad()
    def _eval_on_loader(self, loader, compute_loss=True, compute_auc=True):
        import numpy as np
        from sklearn.metrics import roc_auc_score  # type: ignore

        self.global_model.eval()

        total_loss = 0.0
        total_correct = 0
        total_num = 0

        all_probs = []
        all_labels = []

        for x, y in loader:
            if type(x) == type([]):
                x = x[0].to(self.device)
            else:
                x = x.to(self.device)

            # CE / AUC 都更稳：y 统一用 long
            y = y.to(self.device).long()

            logits = self.global_model(x)
            bs = y.size(0)

            pred = logits.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total_num += bs

            if compute_loss:
                loss = self.loss(logits, y)
                total_loss += loss.item() * bs

            if compute_auc:
                probs = F.softmax(logits, dim=1).detach().cpu()
                all_probs.append(probs)
                all_labels.append(y.detach().cpu())

        avg_loss = total_loss / max(total_num, 1) if compute_loss else 0.0
        acc = total_correct / max(total_num, 1)

        auc = float("nan")
        if compute_auc and total_num > 0:
            probs = torch.cat(all_probs, dim=0).numpy()  # (N, C)
            labels = torch.cat(all_labels, dim=0).numpy()  # (N,)

            num_classes = probs.shape[1]
            uniq = np.unique(labels)

            # 二分类：需要同时有正负类，否则不可定义
            if num_classes == 2:
                if len(uniq) < 2:
                    auc = float("nan")
                else:
                    auc = float(roc_auc_score(labels, probs[:, 1]))
            else:
                # 多分类 OvR：逐类算，跳过不可算的类（该类全0或全1）
                aucs = []
                for k in range(num_classes):
                    yk = (labels == k).astype(np.int32)  # one-vs-rest
                    if yk.min() == yk.max():  # 全 0 或全 1 -> AUC 不定义
                        continue
                    try:
                        aucs.append(roc_auc_score(yk, probs[:, k]))
                    except Exception:
                        continue

                auc = float(np.mean(aucs)) if len(aucs) > 0 else float("nan")

        return avg_loss, acc, auc, total_num

    def evaluate(self, acc=None, loss=None, auc=None):
        train_loss, _, _, _ = self._eval_on_loader(
            self.public_train_loader, compute_loss=True, compute_auc=False
        )

        test_loss, test_acc, test_auc, _ = self._eval_on_loader(
            self.public_test_loader, compute_loss=False, compute_auc=True
        )

        if acc is None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss is None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        if auc is None:
            self.rs_test_auc.append(test_auc)
        else:
            auc.append(test_auc)

        print("Averaged Train Loss (public): {:.4f}".format(train_loss))
        print("Averaged Test Accuracy (public): {:.4f}".format(test_acc))
        print("Averaged Test AUC (public): {:.4f}".format(test_auc))

    @torch.no_grad()
    def _ensemble_logits_batch(self, x_batch):
        """
        Teacher = ensemble of group models (already aggregated within each group).
        Weighted average by group_weight (default: |group|/num_clients).
        """
        tea = None
        wsum = 0.0
        for w, gm in zip(self.group_weight, self.group_model):
            gm.eval()
            out = gm(x_batch)
            tea = out * w if tea is None else tea + out * w
            wsum += w
        return tea / max(wsum, 1e-12)

    def server_train(self):
        self.global_model.train()

        for _ in range(self.kd_epochs):
            for x, y in self.public_train_loader:
                if type(x) == type([]):
                    x = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device).long()

                # student forward
                stu_log = self.global_model(x)

                # teacher logits (ensemble)
                with torch.no_grad():
                    tea_log = self._ensemble_logits_batch(x)

                loss = self.kd_loss(stu_log, tea_log, y)

                self.server_opt.zero_grad()
                loss.backward()
                self.server_opt.step()

    def train(self):
        if "MNIST" == self.dataset:
            file_path = os.path.join(self.path, "sem_MNIST_result.json")
        elif "FEMNIST" == self.dataset:
            file_path = os.path.join(self.path, "sem_FEMNIST_result.json")
        elif "AGNews" == self.dataset:
            file_path = os.path.join(self.path, "sem_AGNews_result.json")
        elif "Cifar100" == self.dataset:
            file_path = os.path.join(self.path, "sem_Cifar100_result.json")
        else:
            pass

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"There is no {file_path}")
        else:
            with open(file_path, "r") as f:
                group = ujson.load(f)

        for i in range(self.global_rounds + 1):
            s_t = time.time()

            self.group_model = []
            self.group_weight = []

            # --------- group-wise local train + group aggregation ---------
            for j in range(self.num_groups):
                self.selected_clients = [self.clients[s] for s in group[j]]

                for c in self.selected_clients:
                    c.train()

                self.uploaded_ids = [c.id for c in self.selected_clients]
                self.uploaded_models = [c.model for c in self.selected_clients]

                ws = [getattr(c, "train_samples", 1) for c in self.selected_clients]
                tot = sum(ws)
                if tot <= 0:
                    ws = [1 for _ in ws]
                    tot = len(ws)
                self.uploaded_weights = [w / tot for w in ws]

                gmodel = self.aggregate_uploaded_models_to_new_model()
                self.group_model.append(gmodel)

                self.group_weight.append(float(tot))

            self.server_train()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

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
            self.set_new_clients(clientGrouprox)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

