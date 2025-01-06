# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# --------------------------------------------------------

import math
import sys
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy
from typing import Iterable, Optional
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, multilabel_confusion_matrix
from pycm import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, cohen_kappa_score, roc_curve


def classification_metrics(y_true, y_pred, y_score, nb_classes, use_youden_index=False):
    # print(y_true.shape, y_pred.shape, y_score.shape, nb_classes)
    if nb_classes == 2 and use_youden_index:
        fpr, tpr, thresholds = roc_curve(y_true, y_score[:, 1])
        best_threshold_index = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_threshold_index]
        y_pred = (y_score[:, 1] >= best_threshold).astype(int)

    if nb_classes == 2:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        sen = recall_score(y_true, y_pred)
        spe = recall_score(y_true, y_pred, pos_label=0)
        pre = precision_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred, weights='linear')
        auc_roc = roc_auc_score(y_true, y_score[:, 1])
        auc_pr = average_precision_score(y_true, y_score[:, 1])
    else:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        sen = recall_score(y_true, y_pred, average='macro')
        spe = np.mean(specificity(y_true, y_pred))
        pre = precision_score(y_true, y_pred, average='macro')
        kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        auc_roc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
        auc_pr = average_precision_score(y_true, y_score, average='macro')
        # auc_pr = 0

    return dict(auc_roc=auc_roc, auc_pr=auc_pr, acc=acc, f1=f1, sen=sen, spe=spe, pre=pre, kappa=kappa, y_pred=y_pred)


def specificity(y_true: np.array, y_pred: np.array, classes: set = None):
    if classes is None:
        classes = set(np.concatenate((np.unique(y_true), np.unique(y_pred))))
    specs = []
    for cls in classes:
        y_true_cls = np.array((y_true == cls), np.int32)
        y_pred_cls = np.array((y_pred == cls), np.int32)
        specs.append(recall_score(y_true_cls, y_pred_cls, pos_label=0))
    return specs




def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def evaluate(data_loader, model, device, task, epoch, mode, num_class, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    if not os.path.exists(args.method +   task):
        os.makedirs(args.method +   task)

    prediction_decode_list = []
    prediction_list = []
    true_label_decode_list = []
    true_label_onehot_list = []
    
    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        true_label=F.one_hot(target.to(torch.int64), num_classes=num_class)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
            prediction_softmax = nn.Softmax(dim=1)(output)
            _,prediction_decode = torch.max(prediction_softmax, 1)
            _,true_label_decode = torch.max(true_label, 1)

            prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
            true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
            prediction_list.extend(prediction_softmax.cpu().detach().numpy())


        acc1,_ = accuracy(output, target, topk=(1,2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)
    prediction_list = np.array(prediction_list)
    results = classification_metrics(y_true=true_label_decode_list, y_pred=prediction_decode_list, y_score=prediction_list, nb_classes=num_class)
    metric_logger.synchronize_between_processes()
    
    print('Sklearn Metrics - Acc: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} F1-score: {:.4f} sensitivity: {:.4f} specificity: {:.4f} precision: {:.4f} kappa: {:.4f}'.format\
                            (results['acc'], results['auc_roc'], results['auc_pr'], results['f1'], results['sen'], results['spe'], results['pre'], results['kappa'])) 

    results_path = args.method +  task+'_metrics_{}.csv'.format(mode)
    if not os.path.exists(results_path):
        with open(results_path,mode='a',newline='',encoding='utf8') as cfa:
            wf = csv.writer(cfa)
            data2=[['method', 'seed','frac', 'task','acc', 'auc_roc', 'auc_pr', 'f1', 'sen', 'spe', 'pre', 'kappa', 'loss']]
            for i in data2:
                wf.writerow(i)        
    with open(results_path,mode='a',newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data2=[[args.method, args.seed,args.frac, args.task,results['acc'], results['auc_roc'], results['auc_pr'], results['f1'], results['sen'], results['spe'], results['pre'], results['kappa'], metric_logger.loss]]
        for i in data2:
            wf.writerow(i)
            
    
    if mode=='test':
        cm = ConfusionMatrix(actual_vector=true_label_decode_list, predict_vector=prediction_decode_list)
        cm.plot(cmap=plt.cm.Blues,number_label=True,normalized=True,plot_lib="matplotlib")
        plt.savefig(args.method +  task+'confusion_matrix_test.jpg',dpi=600,bbox_inches ='tight')
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},results['auc_roc']

