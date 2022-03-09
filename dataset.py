import os
import json
from typing import Dict
import numpy as np
from PIL import Image
from torch.utils.data import Dataset,Subset
from sklearn.model_selection import train_test_split
from augmentations import ToTensor
from utils import flatten_keypoints

def get_train_val_dataset(dataset, val_split:float=0.1,transforms:Dict = None)-> Dict:
  """Creating train and validation dataset from a base dataset with different transfromation pipeline.

  Args:
      dataset (_type_): TubeLengthDataset object
      val_split (float, optional): validation split Defaults to 0.1.
      transforms (dict, optional): transformation pipeline dict. Defaults to None.

  Returns:
      Dict: datasets wrapped in dictionary.
  """
  train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
  datasets = {}
  datasets['train'] = Subset(dataset, train_idx)
  datasets['val'] = Subset(dataset, val_idx)
  if (transforms != None):
    datasets['train'] = DatasetFromSubset(datasets['train'] , transforms["train"])
    datasets['val'] = DatasetFromSubset(datasets['val'] , transforms["val"])
  else:
    print("please apply resize and totensor transform!!!")
  return datasets
  
class TubeLengthDataset(Dataset):
    
  '''Dataset class for the glue tube keypoint model training.'''
  
  def __init__(self, data_dir:str, transform = None):

    self.image_dir = os.path.join(data_dir,"images")
    self.keypoints_dir = os.path.join(data_dir,"annotations")
    self.transform = transform
    self.images = os.listdir(self.image_dir)
    self._repr_indent = 4

  def __len__(self):
        return len(self.images)
  
  def __getitem__(self, index):

        image_path = os.path.join(self.image_dir, self.images[index])
        keypoint_path = os.path.join(self.keypoints_dir, self.images[index].replace(".jpg", ".json"))

        image = np.array(Image.open(image_path).convert("RGB"))
        with open(keypoint_path) as jsonfile: 
          keypoint = flatten_keypoints(json.load(jsonfile)["keypoints"])
        if self.transform is not None:
            transformed_data = self.transform(image=image, keypoints=keypoint)
            transformed_data = ToTensor(transformed_data)
            image,keypoint = transformed_data["image"], transformed_data["keypoints"]

        return image, keypoint
      
  ## ImageFolder like __repr__ function
  def __repr__(self):
      head = "Dataset " + self.__class__.__name__
      body = ["Number of datapoints: {}".format(self.__len__())]
      body.append("Image location: {}".format(self.image_dir))
      body.append("Keypoint location: {}".format(self.keypoints_dir))
      if hasattr(self, "transform") and self.transform is not None:
          body += ["Transform: "+str(self.transform)]
      lines = [head] + [" " * self._repr_indent + line for line in body]
      return '\n'.join(lines)

class DatasetFromSubset(Dataset):
    '''Custom class for creating subsets for being able to apply
        diffrent transformation to training and validation dataset.
    '''
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.image_dir = subset.dataset.image_dir
        self.keypoints_dir = subset.dataset.keypoints_dir 
        self._repr_indent = subset.dataset._repr_indent

    def __getitem__(self, index):
        image, keypoint = self.subset[index]
        if self.transform:
            transformed_data = self.transform(image=image, keypoints=keypoint)
            transformed_data = ToTensor(transformed_data)
            image,keypoint = transformed_data["image"], transformed_data["keypoints"]
        return image, keypoint

    def __len__(self):
        return len(self.subset)

    ## ImageFolder like __repr__ function
    def __repr__(self):
      head = "Dataset " + self.subset.dataset.__class__.__name__
      body = ["Number of datapoints: {}".format(self.__len__())]
      body.append("Image location: {}".format(self.image_dir))
      body.append("Mask location: {}".format(self.keypoints_dir ))

      if hasattr(self, "transform") and self.transform is not None:
          body += ["Transform: "+self.transform.name]
      lines = [head] + [" " * self._repr_indent + line for line in body]
      return '\n'.join(lines)
