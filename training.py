
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (DeviceStatsMonitor, ModelCheckpoint, 
                                          LearningRateMonitor)
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning import Trainer, loggers as pl_loggers

from configs import *
from utils import *
from models import *
from dataset import *
from pl_modules import *
from augmentations import *

## setting seeds for reproducable results.
pl.seed_everything(hash("setting random seeds") % 2**32 - 1)

## Inserting keypoint in single tube images.
insert_dummy_keypoints(TRAIN_PATH)
insert_dummy_keypoints(TEST_PATH)

#defining loss function
criterion = nn.MSELoss()

## defining the callbacks 
tb_logger = pl_loggers.TensorBoardLogger(logdir, name="glue_tube_length_model")

checkpoint_callback = ModelCheckpoint(dirpath=ckptdir, 
                                      every_n_epochs=10,
                                      save_top_k=2, 
                                      monitor = "val_loss",
                                      filename='{epoch}-{val_loss:.2f}')

progressbar_callback = TQDMProgressBar(refresh_rate= 3)
lr_monitor_callback = LearningRateMonitor()
callbacks = [lr_monitor_callback,progressbar_callback,checkpoint_callback]

## Configuring the trainer for pytorch lightning
if torch.cuda.is_available():
    trainer = Trainer(max_epochs = EPOCHS,
                  num_sanity_val_steps=1,
                  log_every_n_steps = 6,
                  weights_summary='top',
                  gpus = -1,
                  precision = 16,
                  logger=tb_logger,
                  callbacks = callbacks,
                  accumulate_grad_batches=4,
                  auto_lr_find = True,
                  reload_dataloaders_every_n_epochs = True
                  )
else:
    trainer = Trainer(max_epochs = EPOCHS,
                  num_sanity_val_steps=1,
                  log_every_n_steps = 6,
                  weights_summary='top',
                  logger=tb_logger,
                  callbacks = callbacks,
                  auto_lr_find = True,
                  reload_dataloaders_every_n_epochs = True
                  )
    

network = SimpleKeypointModel() # creating the model object
pl_model = plModel(network, criterion, learning_rate = LR) # creating the PL model wrapper object

base_dataset = TubeLengthDataset(TRAIN_PATH) # creating the base dataset object
test_dataset = TubeLengthDataset(TEST_PATH,transform = test_transforms) #creating the test dataset object
#creating the pl datamodule
pl_datamodule = plDataModule(base_dataset,test_dataset,transforms_dict,val_split = VAL_SPLIT, batch_size = BATCH_SIZE)

#fit the model
trainer.fit(model = pl_model,datamodule= pl_datamodule)

#evaluate the model 
val_score = trainer.test(model = pl_model,datamodule= pl_datamodule)
print(val_score)

#save only the weight separately
torch.save(pl_model.network.state_dict(),"./model/tube_length_model.pt")
