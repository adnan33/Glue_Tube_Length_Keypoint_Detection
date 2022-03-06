# this will work as a wrapper for lighting  over native pytorch model, datamodule and dataloader

import torch
import pytorch_lightning as pl
from typing import  Dict, Optional
from torch.utils.data import  DataLoader
from dataset import get_train_val_dataset

class plModel(pl.LightningModule):
    """ Pytorch lightning wrapper module for the model class. 
    """
    def __init__(self, network, criterion,learning_rate = 1e-4, **kwargs):
      super().__init__()
      self.network = network
      self.criterion = criterion
      self.learning_rate = learning_rate
      self.save_hyperparameters("network","criterion","learning_rate")  
      
      
    def forward(self,x):
      return self.network(x)

    def configure_optimizers(self):
      optimizer=torch.optim.Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=0.0001)
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", 
                                                             factor =0.5, patience= 3,)
      
      ret_dict = {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'monitor': 'val_loss',
        }
    }
      return ret_dict

    def training_step(self,batch,batch_idx):
      x,y = batch
      y_hat = self(x)
      loss = self.criterion(y_hat,y)
      self.log("train_loss",loss,on_step=False,on_epoch=True,prog_bar=True,logger=True)
      return loss
      
    def validation_step(self,batch,batch_idx):
      x,y = batch
      y_hat = self(x)
      loss = self.criterion(y_hat,y)
      self.log("val_loss",loss,prog_bar=True,logger=True)
      return loss

    def test_step(self,batch,batch_idx):
      x,y = batch
      y_hat = self(x)
      loss = self.criterion(y_hat,y)
      self.log("test_loss",loss,prog_bar=True,logger=True)
      return loss
      


class plDataModule(pl.LightningDataModule):
  
    """ Pytorch lightning wrapper module for the dataset class. 
    """

    def __init__(self,base_dataset,test_dataset, transforms_dict: Dict, val_split: float = 0.1, batch_size: int = 16):
        super().__init__()
        self.base_dataset = base_dataset
        self.test_dataset = test_dataset
        self.transforms_dict = transforms_dict
        self.val_split = val_split
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):

        self.datasets = get_train_val_dataset(self.base_dataset,
                                          transforms = self.transforms_dict,
                                          val_split = self.val_split)

        self.datasets["test"] = self.test_dataset
  

    def train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size = self.batch_size, shuffle=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.datasets["val"],batch_size = self.batch_size, shuffle=False, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.datasets["test"],batch_size = self.batch_size, shuffle=False, pin_memory=True)