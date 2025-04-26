import torch
import torchvision.transforms as transforms
import torchxrayvision as xrv
from PIL import Image
from skimage.io import imread
import os


class PreprocessingImage:
  binary_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
  multi_label_transform = transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])

  @staticmethod
  def binary_classification_preprocessing(image_path):
    image = Image.open(image_path).convert("RGB")
    image = PreprocessingImage.binary_transform(image)
    image = image[None,...]
    return image

  @staticmethod
  def multi_label_preprocessing(image_path):
    image = imread(image_path)
    image = xrv.datasets.normalize(image, 255) # convert 8-bit image to [-1024, 1024] range
    image = image.mean(2)[None, ...] # Make single color channel
    image = PreprocessingImage.multi_label_transform(image)
    image = torch.from_numpy(image)
    image = image[None,...]
    return image
  

class LungModel:
  binary_model = None
  multi_label_model = None

  diseases_names = [
    'Atelectasis',
    'Cardiomegaly',
    'Effusion',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Pleural_Thickening'
  ]

  @staticmethod
  def load_models(binary_model_path='lung_ai\\Deep_Learing\\best_bc_resnet50.pt', multi_label_model_path='lung_ai\\Deep_Learing\\best_mlc_resnet50.pt', device='cpu'):
    if os.path.exists(binary_model_path) and os.path.exists(multi_label_model_path):
      LungModel.binary_model = torch.load(binary_model_path, map_location=torch.device(device))
      LungModel.multi_label_model = torch.load(multi_label_model_path, map_location=torch.device(device))
      print("Models loaded")
    else:
      print("The Models Does't Exist")

  @staticmethod
  def binary_predict(image_path, device='cpu'):
    image = PreprocessingImage.binary_classification_preprocessing(image_path)
    image = image.to(device)
    LungModel.binary_model.eval()
    output = LungModel.binary_model(image) # Image shape is: [1, 3, 224, 224]
    return (torch.sigmoid(output[0][0]).cpu() >= 0.5).item()

  @staticmethod
  def multi_label_predict(image_path, device='cpu'):
    image = PreprocessingImage.multi_label_preprocessing(image_path)
    image = image.to(device)
    LungModel.multi_label_model.eval()
    output = LungModel.multi_label_model(image)
    dict_output = dict(zip(LungModel.multi_label_model.pathologies,output[0].detach().numpy()))
    return {key: value for key, value in dict_output.items() if key in LungModel.diseases_names}

  @staticmethod
  def predict(image_path, device='cpu'):
    binary_model_output = LungModel.binary_predict(image_path, device)
    if binary_model_output:
      return {'No Finding' : 1}
    else:
      multi_label_model_output = LungModel.multi_label_predict(image_path, device)
      return multi_label_model_output


LungModel.load_models()

if __name__ == '__main__':
  print('run main ...')
  
  print(LungModel.predict("C:/Users/AbdulBari/Desktop/مشروع التخرج - الرئة -/Clinic-Mangment/lung_ai/Deep_Learing/Atelectasis.png"))