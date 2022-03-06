import os
import cv2
import glob
import json
import numpy as np
from typing import List,Dict
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt



KEYPOINT_COLOR = (0, 255, 0) # Green

def vis_keypoints(image:np.array, keypoints:List, color:tuple=KEYPOINT_COLOR, diameter:int=15,title:str = "")->None:
  """ Util function to plot the keypoints on images.

  Args:
      image (np.array): image as numpy array
      keypoints (List): keypoints converted to a list of tuples
      color (tuple, optional): Keypoint color. Defaults to KEYPOINT_COLOR.
      diameter (int, optional): Keypoint diameter in pixels. Defaults to 15.
      title (str, optional): title of the plotted image. Defaults to "".
  """
  image = image.copy()

  for (x, y) in keypoints:
      cv2.circle(image, (int(x), int(y)), diameter, color, -1)
      
  plt.figure(figsize=(8, 8))
  plt.axis('off')
  plt.imshow(image)
  plt.title(title)
  plt.show()

def get_tube_length(keypoints:List)->List:
  """ Function to calculate the tube length from the image.

  Args:
      keypoints (List): keypoints converted to a list of tuples

  Returns:
      List: Lengths of the glue tubes present in the image.
  """
  lengths = []
  for i in range(0, len(keypoints), 2):
    lengths.append(np.linalg.norm(np.array(keypoints[i])- np.array(keypoints[i+1])))
  return lengths

def insert_dummy_keypoints(keypoint_dir:str)-> None:
  """ Function to insert dummy keypoints in the images that has only one keypoints. This is done to create consistant
      data format for the model 

  Args:
      keypoint_dir (str): root directory of the intented dataset.
  """
  for path in glob.glob(os.path.join(keypoint_dir,"annotations","*")):
    with open(path,"r+") as f:
      data = json.load(f)
    if len(data["keypoints"])<2:
      data["keypoints"].append([[0, 0,1], [0, 0,1]])
    with open(path,"w+") as f:
      json.dump(data,f)



def flatten_keypoints(keypoints:List)-> List:
  """Function to convert the keypoints from nested list to a list of tuples for using with albumentations library and
      visualizing them.

  Args:
      keypoints (List): keypoints read from the json file

  Returns:
      list[tuple]: keypoints converted to a list of tuples
  """
  #Flatten the keypoints and convert them into tuples for vizualize function and transform function
  return [tuple(keypoint[:-1]) for keypoint in keypoints for keypoint in keypoint]

    
def get_loaders(datasets,batch_size:int)-> Dict:
    """Util function to wrap all datasets into a dictionary for passing to the docker 

    Args:
        datasets (_type_): TubeLengthDataset object
        batch_size (int): batch_size for the dataloader

    Returns:
        Dict : Dictionary containing the dataloaders.
    """
    dataloaders = {}
    dataloaders["train"] = DataLoader(datasets["train"],batch_size, shuffle=True)
    dataloaders["val"] = DataLoader(datasets["val"],batch_size, shuffle=False)
    return dataloaders



