"""
@desc:
    - The original implementation by https://github.com/Jiangbo-Shi/ViLa-MIL
"""
import os

import numpy as np
import torch

import ml_collections
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.metrics import auc as calc_auc

from utils import get_split_loader
from utils import get_optim
from utils import print_network
from utils import calculate_error

from datasets import save_splits
from models import MGPATH
from datasets.dataset_handler import DatasetHandler

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    """
    @desc:
        - Reimplementing the EarlyStopping class to track the validation AUC
            instead of the validation loss.
        - Monitor the F1 score as well. If the AUC is increased, but the F1 score
            is decreased, then the model is not improving. Thus, the model should
            not be saved.
    """
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_val_auc = 0
        self.best_f1 = 0
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(
        self,
        epoch,
        val_loss,
        val_auc,
        val_f1,
        model,
        ckpt_name = 'checkpoint.pt'
    ) -> None:

        score = -val_loss

        if (val_auc > self.best_val_auc and val_f1 >= self.best_f1) or (val_f1 >= self.best_f1):
            self.save_checkpoint(val_loss, val_auc, val_f1, model, ckpt_name)

    def save_checkpoint(
        self,
        val_loss,
        val_auc,
        val_f1,
        model,
        ckpt_name
    ):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            message = f'Validation AUC increase {self.best_val_auc:.6f} --> {val_auc:.6f}.'
            message += f' Validation F1 score increase {self.best_f1:.6f} --> {val_f1:.6f}.'
            message += ' Saving model ...'
            print(message)
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss
        self.best_val_auc = val_auc
        self.best_f1 = val_f1

def train(datasets, cur, args):
    """
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    
    loss_fn = torch.nn.CrossEntropyLoss().cuda()

    print('Done!')

    print('\nInit Model...', end=' ')

    config = ml_collections.ConfigDict()
    config.input_size = 512 if args.use_plip_backbone else 1024
    config.hidden_size = 192
    config.text_prompt = args.text_prompt
    config.freeze_text_encoder = args.freeze_text_encoder
    config.ratio_graph = args.ratio_graph
    config.typeGNN = args.typeGNN
    config.use_plip_backbone = args.use_plip_backbone
    config.use_gigapath_backbone = args.use_gigapath_backbone

    model_dict = {'config': config, 'num_classes':args.n_classes}
    model = MGPATH(**model_dict)

    model = model.to(torch.device('cuda:0'))
    print('Done!')

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')

    low_aug_patches_dir = args.aug_data_folder_s
    high_aug_patches_dir = args.aug_data_folder_l
    low_aug_graph_dir = args.aug_data_graph_dir_s
    high_aug_graph_dir = args.aug_data_graph_dir_l

    train_split = DatasetHandler(
        dataset=train_split,
        low_aug_patches_dir=low_aug_patches_dir,
        high_aug_patches_dir=high_aug_patches_dir,
        low_aug_graph_dir=low_aug_graph_dir,
        high_aug_graph_dir=high_aug_graph_dir
    )
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=100, stop_epoch=160, verbose=True)
    else:
        early_stopping = None
    print('Done!')

    print_network(model)

    for epoch in range(args.max_epochs):
        train_loop(args, epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
        stop = validate(cur, epoch, model, val_loader, args.n_classes,
            early_stopping, writer, loss_fn, args.results_dir)
        if stop:
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _, val_f1 = summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}, F1: {:.4f}'.format(val_error, val_auc, val_f1))

    results_dict, test_error, test_auc, acc_logger, test_f1 = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}, F1: {:.4f}'.format(test_error, test_auc, test_f1))

    each_class_acc = []
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        each_class_acc.append(acc)
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error, each_class_acc, test_f1

def compute_grad_norms(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def train_loop(args, epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    for batch_idx, (data_s, _, nodes_s, edges_s, data_l, _, nodes_l, edges_l, label) in enumerate(loader):
        data_s, data_l, label = (
            data_s.to(device), data_l.to(device), label.to(device)
        )
        nodes_s, edges_s, nodes_l, edges_l = (
            nodes_s.to(device), edges_s.to(device), nodes_l.to(device), edges_l.to(device)
        )
        _, Y_hat, loss = model(data_s, nodes_s, edges_s, data_l, nodes_l, edges_l, label)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        acc_logger.log(Y_hat, label)
        loss_value = loss.item()
        train_loss += loss_value
        error = calculate_error(Y_hat, label)
        train_error += error

    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

def validate(
    cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None
):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    all_pred = []
    all_label = []
    with torch.no_grad():
        for batch_idx, (data_s, _, nodes_s, edges_s, data_l, _, nodes_l, edges_l, label) in enumerate(loader):
            data_s, data_l, label = (
                data_s.to(device, non_blocking=True),
                data_l.to(device, non_blocking=True),
                label.to(device, non_blocking=True)
            )
            nodes_s, edges_s, nodes_l, edges_l = (
                nodes_s.to(device, non_blocking=True), edges_s.to(device, non_blocking=True),
                nodes_l.to(device, non_blocking=True), edges_l.to(device, non_blocking=True)
            )
            Y_prob, Y_hat, loss = model(data_s, nodes_s, edges_s, data_l, nodes_l, edges_l, label)

            acc_logger.log(Y_hat, label)
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            all_pred.append(Y_hat.cpu())
            all_label.append(label.cpu())

    val_error /= len(loader)
    val_loss /= len(loader)
    all_label = [float(label.cpu().numpy()) for label in all_label]
    all_pred = [float(pred.cpu().numpy()) for pred in all_pred]
    val_f1 = f1_score(all_label, all_pred, average='macro')

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])

    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, f1: {: .4f}'.format(val_loss, val_error, auc, val_f1))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_error, auc, val_f1, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(
    model, loader, n_classes
):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_error = 0.
    all_pred = []
    all_label = []

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_s, _, nodes_s, edges_s, data_l, _, nodes_l, edges_l, label) in enumerate(loader):
        data_s, data_l, label = (
            data_s.to(device), data_l.to(device), label.to(device)
        )
        nodes_s, edges_s, nodes_l, edges_l = (
            nodes_s.to(device), edges_s.to(device), nodes_l.to(device), edges_l.to(device)
        )
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            Y_prob, Y_hat, loss = model(data_s, nodes_s, edges_s, data_l, nodes_l, edges_l, label)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

        all_pred.append(Y_hat.cpu())
        all_label.append(label.cpu())

    test_error /= len(loader)
    all_label = [float(label.cpu().numpy()) for label in all_label]
    all_pred = [float(pred.cpu().numpy()) for pred in all_pred]
    test_f1 = f1_score(all_label, all_pred, average='macro')

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, acc_logger, test_f1
