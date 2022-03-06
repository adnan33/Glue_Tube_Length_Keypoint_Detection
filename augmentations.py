import torch
import numpy as np
from typing import Dict,List,Tuple
import albumentations as A
from configs import IMAGE_SHAPE

def ToTensor(sample:Dict)-> Dict:
    """ Convert the augmented output of the transformation pipeline to tensor.

    Args:
        sample (Dict): augmentation output from albumentations pipeline.

    Returns:
        Dict: dictionary object containing image and keypoint converted to tensor.
    """
    image, keypoints = sample['image'], sample['keypoints']
      
    # if image has no grayscale color channel, add one
    if(len(image.shape) == 2):
        # add that third color dim
        image = image.reshape(image.shape[0], image.shape[1], 1)
        
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    if len(keypoints)==2:
      keypoints.extend([(0, 0), (0, 0)])
    return {'image': torch.from_numpy(image),
            'keypoints': torch.from_numpy(np.array(keypoints,dtype=np.float32).flatten()/IMAGE_SHAPE[0])} #scaling the keypoint values to 0-1 range

## Separating the transfroms in to multiple functions to be able to apply them in various combination.

def pre_transforms(image_shape:Tuple=IMAGE_SHAPE)->List:
    
    """ Function to add pre transforms to the data samples.
        Here only resize is applied.

    Args:
        image_shape (Tuple, optional): output image shape. Defaults to IMAGE_SHAPE.

    Returns:
        List: transformations in list to be composed.
    """
    return [A.Resize(image_shape[0],image_shape[1], p=1)]


def hard_transforms()->List:
    """Function to add transforms to the training data samples.

    Returns:
        List: transformations in list to be composed.
    """
    result = [
     A.VerticalFlip(p=0.7,),
     A.HorizontalFlip(p=0.7),
     A.OneOf([
            A.HueSaturationValue(p=0.5), 
            A.RGBShift(p=0.7)
        ], p=1),                          
        A.RandomBrightnessContrast(p=0.5),
      
    ]

    return result

def post_transforms():
    """ Function to normalize the data samples.

    Returns:
        List: transformations in list to be composed.
    """
    return [
            A.Normalize(mean = 0, std = 1,max_pixel_value=255.0),
           ]

def compose(transforms_to_compose:List):
    """Function to create augmentation pipeline.

    Args:
        transforms_to_compose (List): list of diffrent transformantions to be applied.

    Returns:
        Albumentation pipeline
    """
    # combine all augmentations into single pipeline
    result = A.Compose([
      item for sublist in transforms_to_compose for item in sublist
    ],keypoint_params=A.KeypointParams(format='xy',remove_invisible="False"))
    return result



train_transforms = compose([
    pre_transforms(),
    hard_transforms(), 
    post_transforms()
])

test_transforms = compose([pre_transforms(),post_transforms()])
viz_transform = compose([hard_transforms()])
train_transforms.name = "train_transforms"
test_transforms.name = "test_transforms"
transforms_dict = {"train":train_transforms,"val":test_transforms , "test":test_transforms}