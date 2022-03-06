import cv2
import json
import torch
from utils import flatten_keypoints , vis_keypoints, get_tube_length
from augmentations import ToTensor, test_transforms
from models import *
from configs import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# creating the model object
test_model = SimpleKeypointModel()
#loading the model weight
test_model.load_state_dict(torch.load("./model/tube_length_model.pt",map_location=DEVICE))
test_model.eval();

# loading the image
filename = "IMG_4915_JPG_jpg.rf.565de31cc108a914afec6fc6a2b5ee13.jpg"
image = cv2.imread(os.path.join(TEST_PATH,"images",filename))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#Loading keypoints
annotation_path = os.path.join(TEST_PATH,"annotations",filename.replace('.jpg','.json'))
with open(annotation_path) as file:
  content = json.load(file)
  keypoints = flatten_keypoints(content["keypoints"])

#converting data to tensor after resize and normalization
test_data = ToTensor(test_transforms(image = image, keypoints = keypoints))

#resize image
image = cv2.resize(image,(512,512))

ground_truth_kp = test_data["keypoints"]*IMAGE_SHAPE[0]
ground_truth_kp = list(map(tuple,ground_truth_kp.cpu().detach().numpy().reshape(4,2)))

#inference
pred_kp = test_model(test_data["image"].reshape(1,3, 512, 512))*IMAGE_SHAPE[0]
pred_kp = list(map(tuple,pred_kp.cpu().detach().numpy().reshape(4,2)))


print(f"ground truth keypoints: {pred_kp}")
vis_keypoints(image,ground_truth_kp,title="ground truth keypoints")
print(f"predicted keypoints: {pred_kp}")
vis_keypoints(image,pred_kp,title="predicted keypoints")

print(f"ground truth tube length(s): {get_tube_length(ground_truth_kp)}")
print(f"predicted tube length(s): {get_tube_length(pred_kp)}")