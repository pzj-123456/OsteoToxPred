import json
import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from dataset.dataset_test import MolTestDatasetWrapper

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score



def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('config_finetune.yaml', os.path.join(model_checkpoints_folder, 'config_finetune.yaml'))


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    # 标准化数据
    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    # 反标准化
    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class FineTune(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time + '_' + config['task_name'] + '_' + config['dataset']['target']
        log_dir = os.path.join('scaffold/finetune', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset
        if config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif config['dataset']['task'] == 'regression':
            if self.config["task_name"] in ['qm7', 'qm8', 'qm9']:
                self.criterion = nn.L1Loss()
            else:
                self.criterion = nn.MSELoss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data, n_iter):
        # get the prediction
        __, pred,__ = model(data)  # [N,C]  前向传播

        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, data.y.flatten())
        elif self.config['dataset']['task'] == 'regression':
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(data.y))
            else:
                loss = self.criterion(pred, data.y)

        return loss

    def train(self):

        # 打印所使用的指纹及其维度
        print("使用的指纹及其维度：")
        ['ecfp', 'maccs', 'ap', 'ext', 'torsion', 'avalon']
        for fp in self.config['dataset']['fingerprint_list']:
            if fp == 'ecfp':
                print(f"  ECFP: {self.config['dataset']['ecfp_bits']} 位")
            elif fp == 'maccs':
                print(f"  MACCS: {self.config['dataset']['maccs_bits']} 位")
            elif fp == 'ap':
                print(f"  Atom Pair(ap): {self.config['dataset']['ap_bits']} 位")
            elif fp == 'ext':
                print(f"  Extended Fingerprint(ext): {self.config['dataset']['ext_bits']} 位")
            elif fp == 'torsion':
                print(f"  Topological Torsion(torsion): {self.config['dataset']['torsion_bits']} 位")
            elif fp == 'avalon':
                print(f"  Avalon: {self.config['dataset']['avalon_bits']} 位")


        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        self.normalizer = None
        # if self.config["task_name"] in ['qm7', 'qm9']:
        #     # qm7 /9  的大范围数值需要标准化
        #     labels = []
        #     for batch in train_loader:  # 直接遍历 Batch 对象，无需解包
        #         labels.append(batch.y)
        #     labels = torch.cat(labels)
        #     self.normalizer = Normalizer(labels)
        #     print(self.normalizer.mean, self.normalizer.std, labels.shape)
        #
        # if self.config['model_type'] == 'gin':
        #     from models.ginet_finetune import GINet
        #     model = GINet(self.config['dataset']['task'], **self.config["model"]).to(self.device)
        #     # model = self._load_pre_trained_weights(model)
        # elif self.config['model_type'] == 'gcn':
        #     # from models.gcn_finetune import GCN
        #     model = GCN(self.config['dataset']['task'], **self.config["model"]).to(self.device)
        #     model = self._load_pre_trained_weights(model)

        # layer_list = []
        # for name, param in model.named_parameters():
        #     if 'pred_head' in name:
        #         print(name, param.requires_grad)
        #         layer_list.append(name)

        # 其中params是pred_head的部分，而base_params是其他部分。
        # 这可能是因为想要对不同的层使用不同的学习率，比如在迁移学习中常见的做法，基础层用较小的学习率，而顶层用较大的。
        # params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        # base_params = list(
        #     map(lambda x: x[1], list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))
        #
        # optimizer = torch.optim.Adam(
        #     [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
        #     self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        # )

        from models.ginet_finetune import GINet
        fps = self.config['dataset']['fingerprint_list']
        model = GINet(self.config['dataset']['task'],fingerprint_list=fps, **self.config["model"]).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config['init_lr'],
            weight_decay=eval(self.config['weight_decay'])
        )
        print(model)


        #   init_base_lr: 0.0001 （微调） <  init_lr: 0.002  （快速训练）

        # if apex_support and self.config['fp16_precision']:
        #     model, optimizer = amp.initialize(
        #         model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
        #     )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = 0

        for epoch_counter in range(self.config['epochs']):
            # —— 在每个 epoch 开始时打印当前 lr
            for i, param_group in enumerate(optimizer.param_groups):
                print(f"Epoch {epoch_counter} — lr[{i}] = {param_group['lr']}")

            # =---- 新增：初始化本 epoch 的累积变量 ----
            epoch_loss = 0.0
            num_samples = 0
            train_preds = []
            train_labels = []
            # -------------------------------------------=

            for bn, data in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(self.device)
                # 前向传播 +损失计算
                __, pred,__ = model(data)
                loss = self._step(model, data, n_iter)

                # 累计 loss =-------------------------------
                batch_size = data.y.size(0)
                epoch_loss += loss.item() * batch_size
                num_samples += batch_size

                # 将 logits 转概率（分类任务）
                if self.config['dataset']['task'] == 'classification':
                    probs = F.softmax(pred, dim=-1)[:, 1].detach().cpu().numpy()
                    labels = data.y.flatten().cpu().numpy()
                    train_preds.extend(probs.tolist())
                    train_labels.extend(labels.tolist())
              # ---------------------------------------------=

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print(epoch_counter, bn, loss.item())

                loss.backward()

                optimizer.step()
                n_iter += 1

            # ---- 新增：计算并打印本 epoch 的平均 loss、AUC、ACC ----
            avg_loss = epoch_loss / num_samples
            if self.config['dataset']['task'] == 'classification' and num_samples > 0:
                from sklearn.metrics import roc_auc_score, accuracy_score
                train_auc = roc_auc_score(train_labels, train_preds)
                # tn, fp, fn, tp = confusion_matrix(train_labels, train_preds).ravel()
                # train_acc = (tp + tn) / (tp + fp + tn + fn)
                # 用 0.5 阈值计算 acc
                # train_acc = accuracy_score(train_labels, [1 if p > 0.5 else 0 for p in train_preds])
                # train_acc = accuracy_score(train_labels, train_preds)
                # train_acc = accuracy_score(train_labels, np.argmax(train_preds, axis=1))

                 # 可写入 TensorBoard
                # self.writer.add_scalar('train_auc', train_auc, epoch_counter)
                # self.writer.add_scalar('train_acc', train_acc, epoch_counter)

                # pred_probs：softmax第二列或sigmoid输出，train_labels真实标签
                train_auc = roc_auc_score(train_labels, train_preds)
                train_preds = [1 if p > 0.5 else 0 for p in train_preds]  # 二分类sigmoid/softmax输出转类别
                train_acc = accuracy_score(train_labels, train_preds)
                train_f1 = f1_score(train_labels, train_preds)
                train_precision = precision_score(train_labels, train_preds)
                train_recall = recall_score(train_labels, train_preds)
                train_mcc = matthews_corrcoef(train_labels, train_preds)
                print(
                    f"==> Epoch {epoch_counter} 训练集: avg_loss = {avg_loss:.4f}, AUC = {train_auc:.4f}, ACC = {train_acc:.4f}")

                # avg_loss：每轮平均loss

                self.writer.add_scalar('train/loss', avg_loss, epoch_counter)
                self.writer.add_scalar('train/acc', train_acc, epoch_counter)
                self.writer.add_scalar('train/auc', train_auc, epoch_counter)
                self.writer.add_scalar('train/f1', train_f1, epoch_counter)
                self.writer.add_scalar('train/precision', train_precision, epoch_counter)
                self.writer.add_scalar('train/recall', train_recall, epoch_counter)
                self.writer.add_scalar('train/mcc', train_mcc, epoch_counter)

            else:
                print(f"==> Epoch {epoch_counter} 训练集: avg_loss = {avg_loss:.4f}")
            # -------------------------------------------


            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification':
                    valid_loss, valid_cls, valid_acc, valid_f1, valid_precision, valid_recall, valid_mcc\
                        = self._validate(model, valid_loader)
                    if valid_cls > best_valid_cls:
                        # save the model weights
                        best_valid_cls = valid_cls
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                # elif self.config['dataset']['task'] == 'regression':
                #     valid_loss, valid_rgr,valid_acc = self._validate(model, valid_loader)
                #     if valid_rgr < best_valid_rgr:
                #         # save the model weights
                #         best_valid_rgr = valid_rgr
                #         torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                # valid_loss：每轮平均loss

                self.writer.add_scalar('valid/loss', valid_loss, epoch_counter)
                self.writer.add_scalar('valid/acc', valid_acc, epoch_counter)
                self.writer.add_scalar('valid/auc', valid_cls, epoch_counter)
                self.writer.add_scalar('valid/f1', valid_f1, epoch_counter)
                self.writer.add_scalar('valid/precision', valid_precision, epoch_counter)
                self.writer.add_scalar('valid/recall', valid_recall, epoch_counter)
                self.writer.add_scalar('valid/mcc', valid_mcc, epoch_counter)
                valid_n_iter += 1

        self._test(model, test_loader)

    def _load_pre_trained_weights(self, model):
        try:
            #  预训练的模型 通过手动进行添加啊..................................................................
            checkpoints_folder = os.path.join('./ckpt/May22_9-25-30', 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            # print("-----------------------------------------------")
            # for name, param in model.named_parameters():
            #     print(name, param.requires_grad, param.shape)
            # print("-----------------------------------------------")
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                __, pred,__ = model(data)
                loss = self._step(model, data, bn)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            valid_loss /= num_data

        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                mae = mean_absolute_error(labels, predictions)
                print('Validation loss:', valid_loss, 'MAE:', mae)
                return valid_loss, mae
            else:
                rmse = mean_squared_error(labels, predictions, squared=False)
                print('Validation loss:', valid_loss, 'RMSE:', rmse)
                return valid_loss, rmse

        elif self.config['dataset']['task'] == 'classification':
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:, 1])
            # tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
            # # acc = (tp + tn) / (tp + fp + tn + fn)
            # acc = accuracy_score(labels, np.argmax(predictions, axis=1))


            valid_probs = predictions[:, 1]  # 正类概率
            # valid_preds = np.argmax(predictions, axis=1)  # 预测类别
            # valid_preds = [1 if p > 0.5 else 0 for p in predictions]
            valid_preds = (predictions[:, 1] > 0.5).astype(int)

            valid_acc = accuracy_score(labels, valid_preds)
            valid_f1 = f1_score(labels, valid_preds)
            valid_precision = precision_score(labels, valid_preds)
            valid_recall = recall_score(labels, valid_preds)
            valid_mcc = matthews_corrcoef(labels, valid_preds)
            valid_auc = roc_auc_score(labels, valid_probs)
            print('Validation loss:', valid_loss, 'ROC AUC:', valid_auc,'Accuracy:', valid_acc)
            return valid_loss, valid_auc, valid_acc, valid_f1, valid_precision, valid_recall, valid_mcc
        return None

    def _test(self, model, test_loader):
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        # ---- 新增：初始化存储 ----
        all_smiles = []
        all_labels = []
        all_preds = []
        all_attns = []
        # -------------------------

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                __, pred, node_attn  = model(data)
                loss = self._step(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)


                # ---- 新增：收集信息 ----
                # 1) SMILES 列表（length=batch_size）
                smiles_batch = data.z
                # 2) 真实标签
                label_batch = data.y.flatten().cpu().tolist()
                # 按图拆分注意力
                node_attn = node_attn.cpu().detach().numpy()
                batch_idx = data.batch.cpu().numpy()
                pred_vals = F.softmax(pred, dim=-1)[:, 1].cpu().tolist()
                for i, smi in enumerate(smiles_batch):
                    mask = (batch_idx == i)
                    attn_per_graph = node_attn[mask].tolist()
                    all_smiles.append(smi)
                    all_labels.append(label_batch[i])
                    all_preds.append(pred_vals[i])
                    all_attns.append(attn_per_graph)
                # -------------------------

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data

        # ---- 新增：保存到文件 ----
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('scaffold/attn', exist_ok=True)
        # a) 保存预测结果 CSV
        df = pd.DataFrame({
            'smiles': all_smiles,
            'label': all_labels,
            'pred': all_preds
        })
        df.to_csv(f'scaffold/attn/attn_pred_{now}.csv', index=False)
        # b) 保存注意力 JSON
        attn_data = [
            {'smiles': s, 'node_attention': a}
            for s, a in zip(all_smiles, all_attns)
        ]
        with open(f'scaffold/attn/attn_values_{now}.json', 'w') as f:
            json.dump(attn_data, f, indent=2)
        print(f"Saved attention results to scaffold/attn/attn_pred_{now}.csv and .json")
        # -------------------------



        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                self.mae = mean_absolute_error(labels, predictions)
                print('Test loss:', test_loss, 'Test MAE:', self.mae)
            else:
                self.rmse = mean_squared_error(labels, predictions, squared=False)
                print('Test loss:', test_loss, 'Test RMSE:', self.rmse)

        elif self.config['dataset']['task'] == 'classification':
            predictions = np.array(predictions)
            labels = np.array(labels)

            # 保存一下结果
            df = pd.DataFrame({
                'Label': labels,
                'Prediction': predictions[:, 1]
            })

            self.roc_auc = roc_auc_score(labels, predictions[:, 1])
            print('Test loss:', test_loss, 'Test ROC AUC:', self.roc_auc)
            # df.to_csv(
            #     'scaffold/roc/{}_{}_finetune.csv'.format(config['fine_tune_from'], config['task_name']),
            #     mode='a', index=False,
            # )
            now = datetime.now().strftime("%Y%m%d_%H%M%S")  # 形如 "20250622_203045"
            # 2）构造带时间戳和 AUC 的文件名
            out_dir = 'scaffold/roc'
            os.makedirs(out_dir, exist_ok=True)
            filename = f"{config['dataset']['fingerprint_list']}_{config['task_name']}_{now}_auc{self.roc_auc:.4f}_finetune.csv"
            filepath = os.path.join(out_dir, filename)
            # 3）保存 DataFrame
            # 如果你希望每次都是新文件，用 mode='w'；若要追加则改回 mode='a'
            df.to_csv(filepath, mode='w', index=False)
            print(f"Saved ROC results to {filepath}")


            # Calculate confusion matrix and other metrics
            pred_classes = np.argmax(predictions, axis=1)  # Get predicted class labels
            from sklearn.metrics import  confusion_matrix

            tn, fp, fn, tp = confusion_matrix(labels, pred_classes).ravel()

            # Sensitivity (Recall, Se)
            sensitivity = tp / (tp + fn)  # True Positive Rate
            # print('Confusion matrix (Se，TPR):', sensitivity)
            # Specificity (Sp)
            specificity = tn / (tn + fp)  # True Negative Rate
            # print('Confusion matrix (Sp,TNR):', specificity)
            # Accuracy (Acc)
            test_acc = (tp + tn) / (tp + fp + tn + fn)
            # print('Confusion matrix (Acc):', accuracy)

            # 马修斯相关系数 (MCC)
            test_mcc = matthews_corrcoef(labels, pred_classes)
            # print('马修斯相关系数 (MCC):', mcc)

            # 精确度 (P)
            test_precision = precision_score(labels, pred_classes)
            # print('精确度 (P):', precision)

            # F1 分数
            test_f1 = f1_score(labels, pred_classes)
            # print('F1 分数 (F1):', f1)

            # 平衡准确率 (BA)
            balanced_accuracy = (sensitivity + specificity) / 2
            # print('平衡准确率 (BA):', balanced_accuracy)
            test_auc = self.roc_auc

            test_recall = recall_score(labels, pred_classes)

            # 打印所有指标
            print(f'灵敏度 (TPR): {sensitivity}')
            print(f'特异性 (TNR): {specificity}')
            print(f'MCC: {test_mcc}')
            print(f'精确度 (P): {test_precision}')
            print(f'F1 分数 (F1): {test_f1}')
            print(f'平衡准确率 (BA): {balanced_accuracy}')
            print(f'准确率 (ACC): {test_acc}')
            print(f'ROC AUC: {self.roc_auc}')

            # self.writer.add_scalar('test/loss', test_loss, 0)
            # self.writer.add_scalar('test/acc', test_acc, 0)
            # self.writer.add_scalar('test/auc', test_auc, 0)
            # self.writer.add_scalar('test/f1', test_f1, 0)
            # self.writer.add_scalar('test/precision', test_precision, 0)
            # self.writer.add_scalar('test/recall', test_recall, 0)
            # self.writer.add_scalar('test/mcc', test_mcc, 0)
        print(f'========== End of testing for {self.config["task_name"]} ==========')


def main(config):
    results = []
    for _ in range(5):
        dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])
        fine_tune = FineTune(dataset, config)
        fine_tune.train()

        # Store the result based on the task type
        if config['dataset']['task'] == 'classification':
            results.append(fine_tune.roc_auc)
        if config['dataset']['task'] == 'regression':
            if config['task_name'] in ['qm7', 'qm8', 'qm9']:
                results.append(fine_tune.mae)
            else:
                results.append(fine_tune.rmse)

    dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])
    fine_tune = FineTune(dataset, config)
    fine_tune.train()

    # Store the result based on the task type
    if config['dataset']['task'] == 'classification':
        results.append(fine_tune.roc_auc)
    if config['dataset']['task'] == 'regression':
        if config['task_name'] in ['qm7', 'qm8', 'qm9']:
            results.append(fine_tune.mae)
        else:
            results.append(fine_tune.rmse)

    # Convert the list to a numpy array for easy computation of mean and std dev
    results = np.array(results)
    print(results)
    print(np.mean(results), np.std(results))
    # Return mean and standard deviation of results
    return np.mean(results), np.std(results)


if __name__ == "__main__":
    config = yaml.load(open("config_finetune.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader)

    if config['task_name'] == 'BBBP':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/downstream_data/BBBP.csv'
        target_list = ["p_np"]


    elif config["task_name"] == 'bonetox':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/downstream_data/bonetox-new.csv'
        target_list = ['label']

    elif config["task_name"] == 'bonetox-A':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/downstream_data/bonetox-A.csv'
        target_list = ['label']
    elif config["task_name"] == 'bonetox-B':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/downstream_data/bonetox-B.csv'
        target_list = ['label']
    elif config["task_name"] == 'bonetox-C1':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = 'data/downstream_data/bonetox-C1.csv'
        target_list = ['label']
    else:
        raise ValueError('Undefined downstream task!')

    print(config)
    print()
    print(config["task_name"])

    results_list = []
    for target in target_list:
        config['dataset']['target'] = target
        mean, std = main(config)
        results_list.append([target, mean, std])

    os.makedirs('experiments', exist_ok=True)
    df = pd.DataFrame(results_list)
    df.to_csv(
        'scaffold/experiments/{}_{}_finetune.csv'.format(config['fine_tune_from'], config['task_name']),
        mode='a', index=False, header=False
    )
