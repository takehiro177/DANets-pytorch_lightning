
import numpy as np

from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
from torchvision.transforms import v2

from utils.config_loader import load_config
CFG = load_config('../config/experiment/danets_baseline.yaml')

# example of a custom metric for partial AUC using torchmetrics
class CustomMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):        
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        # Convert lists of tensors to single tensors
        preds = torch.cat(self.preds, dim=0) if self.preds else torch.Tensor()
        target = torch.cat(self.target, dim=0) if self.target else torch.Tensor()
        
        # Calculate the custom metric
        custom_metric = 0.0
        
        return torch.tensor(custom_metric)

# custom loss and metric implementation using PyTorch Lightning, optional to use TTA as reference
class LightningModel(pl.LightningModule):
    def __init__(self, model, num_train_length=None, TTA=False):
        super().__init__()

        self.learning_rate = CFG['lr']
        self.weight_decay = CFG['wd']
        self.max_epochs = CFG['max_epochs']

        # The inherited PyTorch module
        self.model = model

        # setting training strategy
        #self.label_smoothing = CFG['label_smoothing']
        self.scheduler = CFG['scheduler']

        # The custom loss function
        self.criteria = "YOUR_CUSTOM_LOSS_FUNCTION_HERE"

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=['model'])

        # Set up attributes for computing the accuracy
        self.train_score = CustomMetric()
        self.valid_score = CustomMetric()
        #self.test_score = CustomMetric()
        
    # Defining the forward method is only necessary 
    # if you want to use a Trainer's .predict() method (optional)
    def forward(self, x_meta):
        return self.model(x_meta)
        
    # A common forward step to compute the loss and labels
    # this is used for training, validation, and testing below
    def _shared_step(self, batch, train=True):
        true_labels, feats = batch

        outputs = self(feats)
        logits = torch.sigmoid(outputs)

        if train:
            # Apply label smoothing
            #y_smooth = true_labels.unsqueeze(1) * (1 - self.label_smoothing) + (0.5 * self.label_smoothing)
            loss = self.criteria(outputs, true_labels.unsqueeze(1).float())

            return loss, true_labels, logits
        else:
            return true_labels, logits


    def training_step(self, batch, batch_idx):
        loss, true_labels, preds = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # To account for Dropout behavior during evaluation
        self.model.eval()

        with torch.no_grad():
            _, true_labels, preds = self._shared_step(batch)
        self.train_score(preds, true_labels)
        self.log("train_score", self.train_score, on_epoch=True, on_step=False, prog_bar=True)

        self.model.train()

        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, logits = self._shared_step(batch)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.valid_score(logits, true_labels)
        self.log("valid_score", self.valid_score, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def predict_step(self, batch, batch_idx):
        if not self.TTA:
            loss, true_labels, preds = self._shared_step(batch, False)
            return preds.detach().cpu().numpy()
        else:  # TTA option
            logits = torch.zeros((batch['org'].shape[0], CFG['num_classes']))
            for tta_name in batch:
                if CFG['num_classes'] > 1:
                    logits += F.softmax(self(batch[tta_name]), dim=1) / len(batch)
                    return torch.argmax(logits, dim=1).detach().cpu().numpy()
                else:
                    logits += torch.sigmoid(self(batch[tta_name])) / len(batch)
                    return (logits > 0.5).int().detach().cpu().numpy()         

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                            max_lr=self.learning_rate * 100,
                                                            steps_per_epoch=self.num_train_steps,
                                                            epochs=self.max_epochs,
                                                            )
        elif self.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                             T_0=self.num_train_steps * 2,
                                                                             T_mult=2, eta_min=self.min_lr,
                                                                             last_epoch=-1
                                                                             )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=self.min_lr)
        
        return [optimizer], [scheduler]