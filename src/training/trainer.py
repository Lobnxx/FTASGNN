"""Training and evaluation utilities."""

import time
import torch
import torch.nn.functional as F
from dgl.dataloading import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from ..config import Config
import dgl
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

class Trainer:
    """Trainer class for model training and evaluation."""
    
    def __init__(self, model, data, sampler, optimizer):
        self.model = model
        self.g = data  # data is now g
        self.sampler = sampler
        self.optimizer = optimizer       
        self.model_name = Config.MODEL_NAMES[Config.MODELS_SELECT]
        # Initialize training state
        self.epoch_f1s = []
        self.epoch_accs = []
        self.epoch_losses = []
        self.batches_f1s = []
        self.batches_accs = []
        self.batches_losses = []
        
        # Early stopping related
        self.best_f1 = 0
        self.patience = Config.PATIENCE
        self.patience_counter = 0
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=Config.PATIENCE // 2
        )
        
        self.print_data_info()

    def print_data_info(self):
        """Prints dataset information."""
        data_info:str = "\n"
        data_info += f"input_dim: {self.g.ndata['feat'].shape}\t"
        data_info += f"label_dim: {len(torch.unique(self.g.ndata['label']))}\n"
        data_info += f"train_size: {self.g.ndata['train_mask'].sum().item()}\t"
        data_info += f"train_malicious: {self.g.ndata['label'][self.g.ndata['train_mask']].sum()}\n"
        data_info += f"test_size: {self.g.ndata['test_mask'].sum().item()}\t"
        data_info += f"test_malicious: {self.g.ndata['label'][self.g.ndata['test_mask']].sum()}\n"
        data_info += f"num_nodes: {self.g.num_nodes()}\t"
        data_info += f"num_edges: {self.g.num_edges()}\t"
        data_info += f"avg_degree: {self.g.in_degrees().float().mean().item():.2f}\t"
        data_info += f"max_degree: {self.g.in_degrees().max().item()}\t"
        data_info += f"min_degree: {self.g.in_degrees().min().item()}\n"
        data_info += f"device: {self.g.ndata['feat'].device}  "
        start_flag = '\n'+"-"*(9)+"Data Info"+"-"*(9) +'\n'
        end_flag = '\n'+"-"*(len(start_flag)-2)
        logging.info(start_flag+data_info+end_flag)


    def save_checkpoint(self, epoch, is_best=False, train=Config.train):
        """Saves a checkpoint."""
        if not train:
            return
        dataset_name = Config.get_dataset_name()
        dataset_name += f"_adv_{Config.MN}"
        model_name = Config.MODEL_NAMES[Config.MODELS_SELECT]
        
        # Construct save directory: MODEL_SAVE_DIR/dataset_name/
        save_dir = os.path.join(Config.MODEL_SAVE_DIR, f"{dataset_name}/{model_name}")
        os.makedirs(save_dir, exist_ok=True)

        # Construct filename: model_name_epochXXX.pt
        filename = f"epoch{epoch:03d}.pt"
        filepath = os.path.join(save_dir, filename)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_f1': self.best_f1,
            'epoch_f1s': self.epoch_f1s,
            'epoch_accs': self.epoch_accs,
            'epoch_losses': self.epoch_losses,
            'batches_f1s': self.batches_f1s,
            'batches_accs': self.batches_accs,
            'batches_losses': self.batches_losses,
            'config': {k: getattr(Config, k) for k in dir(Config) if not k.startswith('__') and not callable(getattr(Config, k)) and not k.startswith('get_')} # Save partial Config
        }
        
        # Save current checkpoint
        torch.save(checkpoint, filepath)
        # logging.info(f"Model checkpoint saved to {filepath}")
        
        # If it's the best model, save an additional copy to 'best.pt'
        if is_best:
            best_filename = "best.pt"
            best_filepath = os.path.join(save_dir, best_filename)
            torch.save(checkpoint, best_filepath)
            # logging.info(f"Best model saved to {best_filepath}")

    def load_model_checkpoint(self, checkpoint_path: str):
        """
        Loads a model checkpoint.
        :param checkpoint_path: Path to the checkpoint file.
                                Expected format is MODEL_SAVE_DIR/dataset_name/model_name_epochXXX.pt or MODEL_SAVE_DIR/dataset_name/model_name_best.pt
        :return: The loaded epoch.
        """
        # If the provided path is relative, try to find it under MODEL_SAVE_DIR/dataset_name
        if not os.path.isabs(checkpoint_path):
            dataset_name = Config.get_dataset_name()
            expected_path = os.path.join(Config.MODEL_SAVE_DIR, dataset_name, checkpoint_path)
            if os.path.exists(expected_path):
                checkpoint_path = expected_path
            else:
                # If not found in the subdirectory, try to find it in the MODEL_SAVE_DIR root directory (for compatibility with old paths)
                expected_path_root = os.path.join(Config.MODEL_SAVE_DIR, checkpoint_path)
                if os.path.exists(expected_path_root):
                    checkpoint_path = expected_path_root
                else:
                    raise FileNotFoundError(f"Checkpoint file not found at '{checkpoint_path}' or '{expected_path}' or '{expected_path_root}'")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_f1 = checkpoint['best_f1']
        self.epoch_f1s = checkpoint['epoch_f1s']
        self.epoch_accs = checkpoint['epoch_accs']
        self.epoch_losses = checkpoint['epoch_losses']
        self.batches_f1s = checkpoint['batches_f1s']
        self.batches_accs = checkpoint['batches_accs']
        self.batches_losses = checkpoint['batches_losses']
        logging.info(f"Model loaded from {checkpoint_path} (Epoch: {checkpoint['epoch']}, Best F1: {self.best_f1:.4f})")
        return checkpoint['epoch']

    def train(self):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0
        total_nodes = 0
        total_cor = 0

        # Create training data loader
        train_nids = torch.nonzero(self.g.ndata['train_mask'], as_tuple=True)[0]
        dataloader = DataLoader(
            self.g, train_nids, self.sampler,
            batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0
        )

        all_preds = []
        all_labels = []
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        accs = []
        losses = []
        for n, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            
            if "session_time_stamps" in self.g.ndata:
                node_type = blocks[0].ntypes[0] if blocks[0].ntypes else '_N'
                blocks[0].ndata["session_time_stamps"] = {node_type: self.g.ndata["session_time_stamps"][input_nodes]}
            if "packet_time_stamps" in self.g.ndata:
                node_type = blocks[0].ntypes[0] if blocks[0].ntypes else '_N'
                blocks[0].ndata["packet_time_stamps"] = {node_type: self.g.ndata["packet_time_stamps"][input_nodes]}
            self.optimizer.zero_grad()

            # Get features and labels
            x = self.g.ndata['feat'][input_nodes]
            y = self.g.ndata['label'][output_nodes]
            # Forward pass
            # print(self.model_name, " ", Config.MODEL_NAMES_WITHOUT_GCN)
            if self.model_name not in Config.MODEL_NAMES_WITHOUT_GCN:
                out = self.model(blocks, x)
            else:
                # For CNN1D, use only output node features
                x = self.g.ndata['feat'][output_nodes]
                out = self.model(x)
            pred = out.argmax(dim=1)

            ###
            if Config.BINARY_CLASS:
                TP += ((pred == 1) & (y == 1)).sum().item()
                FP += ((pred == 1) & (y == 0)).sum().item() # Predicted as 1, actual as 0
                TN += ((pred == 0) & (y == 0)).sum().item()
                FN += ((pred == 0) & (y == 1)).sum().item() # Predicted as 0, actual as 1
                P = TP/(TP + FP) if TP + FP!=0 else 0
                R = TP/(TP + FN) if TP + FN != 0 else 0
                f1 = 2*P*R/(P+R) if P+R !=0 else 0
        
            ###
            # Calculate loss
            loss = F.nll_loss(F.log_softmax(out, dim=1), y)
            loss_each = F.nll_loss(F.log_softmax(out, dim=1), y, reduction="none")
            # Backward pass
            loss.backward()
            
            self.optimizer.step()           
            
            # Update sampler's reward (only for graph models)
            if Config.MODELS_SELECT != 2 and hasattr(self.sampler, 'get_reward'):
                if blocks is not None and dgl.EID in blocks[-1].edata:
                    edge_ids = blocks[-1].edata[dgl.EID]
                    if not Config.BINARY_CLASS:
                        labels = y.cpu().numpy()
                        preds = pred.cpu().numpy()
                        f1 = f1_score(labels, preds, average='weighted', zero_division=0)
                    reward = self.sampler.get_reward(blocks, self.g, loss_each, 0.0) # Placeholder f1
                    self.sampler.update_action_values_batch(self.g, edge_ids, reward)
               
            # Accumulate statistics           
            total_cor += (pred == y).sum().item()
            batch_nodes = len(output_nodes)
            total_loss += loss.item() * batch_nodes
            total_nodes += batch_nodes
            accs.append(round((pred == y).sum().item()/ batch_nodes, 4))
            losses.append(round(loss.item(), 4))
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
        # Calculate average loss and multi-class metrics
        avg_loss = round(total_loss / total_nodes, 4)
        avg_acc = round(total_cor/total_nodes, 4)
        if not Config.BINARY_CLASS:
        # Calculate multi-class P, R, F1 using weighted average
            P = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            R = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        P = round(P, 4)
        R = round(R, 4)
        f1 = round(f1, 4)
        return avg_acc, avg_loss, accs, P, R, f1, losses

    def test(self):
        """Evaluate the model."""
        start_time = time.time()
        self.model.eval()
        total_correct = 0
        total_normal_correct = 0
        tmc = 0
        total_correct_nodes = 0
        tmn = 0
        total_nodes = 0
        total_loss = 0
        all_preds = []
        all_labels = []
        ###
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        ###
        # Create test data loader
        test_nids = torch.nonzero(self.g.ndata['test_mask'], as_tuple=True)[0]
        dataloader = DataLoader(
            self.g, test_nids, self.sampler,
            batch_size=1024, shuffle=False, drop_last=False, num_workers=0
        )
        accs = []
        with torch.no_grad():
            for input_nodes, output_nodes, blocks in dataloader:
                # Get features and labels
                x = self.g.ndata['feat'][input_nodes]
                y = self.g.ndata['label'][output_nodes]
                if "session_time_stamps" in self.g.ndata:
                    node_type = blocks[0].ntypes[0] if blocks[0].ntypes else '_N'
                    blocks[0].ndata["session_time_stamps"] = {node_type: self.g.ndata["session_time_stamps"][input_nodes]}
                if "packet_time_stamps" in self.g.ndata:
                    node_type = blocks[0].ntypes[0] if blocks[0].ntypes else '_N'
                    blocks[0].ndata["packet_time_stamps"] = {node_type: self.g.ndata["packet_time_stamps"][input_nodes]}
                # Forward pass
                if self.model_name not in Config.MODEL_NAMES_WITHOUT_GCN:
                    out = self.model(blocks, x)
                else:
                    out = self.model(self.g.ndata['feat'][output_nodes])
                
                # Calculate loss
                loss = F.nll_loss(F.log_softmax(out, dim=1), y)
                # Calculate statistics
                pred = out.argmax(dim=1)
                mask = (pred == y)
                if not Config.BINARY_CLASS:
                    tmc += (y[mask] > (len(Config.id_to_label)/2 - 1)).sum().item()
                    tmn += (y > (len(Config.id_to_label)/2 - 1)).sum().item()
                    total_normal_correct += (y[mask] < len(Config.id_to_label)/2).sum().item()
                    total_correct_nodes += (y < len(Config.id_to_label)/2).sum().item()
                total_correct += mask.sum().item()
                total_nodes += len(output_nodes)
                total_loss += loss.item() * total_nodes
                accs.append(round(total_correct / total_nodes * 100, 4)) 
                ###
                if Config.BINARY_CLASS:
                    # 1 malicious traffic, 0 benign traffic
                    TP += ((pred == 1)&(y==1)).sum().item()
                    FP += ((pred == 1) & (y == 0)).sum().item() # Predicted as 1, actual as 0
                    TN += ((pred == 0) & (y == 0)).sum().item()
                    FN += ((pred == 0) & (y == 1)).sum().item() # Predicted as 0, actual as 1
                ###
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # Calculate metrics
        acc = round(total_correct / total_nodes, 4)
        if Config.BINARY_CLASS:
            P = TP/(TP + FP) if TP + FP!=0 else 0
            R = TP/(TP + FN) if TP + FN != 0 else 0
            f1 = 2*P*R/(P+R) if P+R !=0 else 0
            FPR = FP/(FP + TN)
        else:
            P = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            R = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        if not Config.BINARY_CLASS:
            all_labels = np.asarray(all_labels)
            all_preds = np.asarray(all_preds)
            normal_mask = all_labels < (len(Config.id_to_label)/2)
            m_mask = all_labels > (len(Config.id_to_label)/2 - 1)
            n_labels = all_labels[normal_mask]
            m_labels = all_labels[m_mask]
            n_preds = all_preds[normal_mask]
            m_preds = all_preds[m_mask]
            n_acc = round(total_normal_correct/total_correct_nodes, 4)
            m_acc = round(tmc/tmn, 4)
            n_P = precision_score(n_labels, n_preds, average='weighted', zero_division=0)
            m_P = precision_score(m_labels, m_preds, average='weighted', zero_division=0)
            n_R = recall_score(n_labels, n_preds, average='weighted', zero_division=0)
            m_R = recall_score(m_labels, m_preds, average='weighted', zero_division=0)
            n_f1 = f1_score(n_labels, n_preds, average='weighted', zero_division=0)
            m_f1 = f1_score(m_labels, m_preds, average='weighted', zero_division=0)
            n_P = round(n_P, 4);n_R=round(n_R, 4)
            m_P = round(m_P, 4);m_R=round(m_R, 4)
            logging.info(f"n: acc {n_acc}, f1 {n_f1}, p {n_P}, r {n_R}")
            logging.info(f"m: acc {m_acc}, f1 {m_f1}, p {m_P}, r {m_R}")
        P = round(P, 4)
        R = round(R, 4)
        f1 = round(f1, 4)
        FPR = round(FPR, 4)
        avg_loss = round(total_loss/total_nodes, 4)
        return acc, avg_loss, f1, accs, P, R, all_labels, all_preds, FPR

    def train_epochs(self, train=True):
        """Train the model for multiple epochs."""
        # Create checkpoint directory
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        if train:
            logging.info("Starting model training...")
        else:
            logging.info("Skipping training.")
        for epoch in range(Config.EPOCHS):
            if train:
                train_acc, train_loss, train_accs, train_P, train_R, train_f1, train_losses = self.train()
                if epoch == 0:
                    logging.info(f"train_accs: {train_accs}")
                    logging.info(f"train_losses: {train_losses}")            
            # Test current model
            with torch.no_grad():
                test_acc, test_loss, f1, test_accs, test_P, test_R, all_labels, all_preds, FPR = self.test()
            
            # Plot confusion matrix
            # self.plot_confusion_matrix(all_labels, all_preds, epoch)

            # Update learning rate
            self.scheduler.step(f1)
            
            # Check for early stopping
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logging.info(f"\nEarly stopping at epoch {epoch}")
                    break
            
            # Save checkpoint
            self.save_checkpoint(epoch)
            
            logging.info(f"Epoch {epoch:03d} Loss: {test_loss}, Acc: {test_acc}, F1-score: {f1:.4f}, P: {test_P}, R: {test_R}, FPR: {FPR}")
        

    def plot_confusion_matrix(self, all_labels, all_preds, epoch):
        """Plots and saves the confusion matrix."""
        # Solve the problem of Chinese garbled characters
        plt.rcParams['font.sans-serif'] = ['SimHei']  # Specify default font
        plt.rcParams['axes.unicode_minus'] = False  # Solve the problem of negative sign '-' displaying as squares when saving images

        cm = confusion_matrix(all_labels, all_preds)
        
        # Get all unique labels and sort them
        unique_labels = np.unique(all_labels)
        class_names = [f'Category {label}' for label in unique_labels] # Label names can be modified according to actual circumstances

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        
        # Save confusion matrix plot
        plot_dir = os.path.join(Config.CHECKPOINT_DIR, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f'confusion_matrix_epoch_{epoch}.png'))
        plt.close()
        logging.info(f"Confusion matrix saved to {os.path.join(plot_dir, f'confusion_matrix_epoch_{epoch}.png')}")
