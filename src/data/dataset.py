import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, KBinsDiscretizer, \
    TargetEncoder, StandardScaler, QuantileTransformer, PowerTransformer, \
          MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, FunctionTransformer
from sklearn.decomposition import PCA,FactorAnalysis
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from utils.config_loader import load_config
CFG = load_config('../config/experiment/danets_baseline.yaml')


# load features from pandas dataframe for batch processing
class CustomDataset(Dataset):
    def __init__(self, df, meta_feats):
        self.df = df
        self.targets = df['target'].to_list()
        self.feats_cols = ["YOUR_FEATURES"]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        sample = self.df.iloc[idx]

        label = sample['target']
        feats = torch.from_numpy(sample[self.feats_cols].values.astype(np.float32)).to(torch.float32)

        return label, feats

# split training and validation data by pre-defined fold in dataframe and set loader for training and validation
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, df: pd.DataFrame, fold: int, feats: list, batch_size: int = CFG['batch_size']):
        super().__init__()
        self.df = df
        self.fold = fold
        self.batch_size = batch_size
        self.feats = feats
        
    def prepare_data(self):
        None

    def setup(self, stage=None):

        self.train = self.df.loc[self.df['fold'] != self.fold, :].reset_index(drop=True)

        x = self.train[self.feats].values
        scaler = StandardScaler()
        tr_arr_norm = scaler.fit_transform(x)
        self.train[self.feats] = tr_arr_norm
        joblib.dump(scaler, f'Path/standardscaler_fold{self.fold}.pkl')

        self.val = self.df.loc[self.df['fold'] == self.fold, :].reset_index(drop=True)

        x = self.val[self.feats].values
        val_arr_norm = scaler.transform(x)
        self.val[self.feats] = val_arr_norm
        
        self.test = None

        self.train_dataset = CustomDataset(self.train, self.feats)
        self.valid_dataset = CustomDataset(self.val, self.feats)

    def train_dataloader(self):

        train_loader = DataLoader(dataset=self.train_dataset, 
                                  batch_size=self.batch_size, 
                                  drop_last=True,
                                  shuffle=True,
                                  num_workers=CFG['num_workers'],
                                  persistent_workers=True,
                                  pin_memory=True)
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(dataset=self.valid_dataset, 
                                  batch_size=int(self.batch_size * 4), 
                                  drop_last=False,
                                  shuffle=False,
                                  num_workers=CFG['num_workers'],
                                  persistent_workers=True,
                                  pin_memory=True)
        return valid_loader

    
    def predict_dataloader(self):
        valid_loader = DataLoader(dataset=self.valid_dataset, 
                                  batch_size=int(self.batch_size * 4), 
                                  drop_last=False,
                                  shuffle=False,
                                  num_workers=CFG['num_workers'],
                                  persistent_workers=True)
        return valid_loader
    
    
    def test_dataloader(self):
        raise NotImplementedError