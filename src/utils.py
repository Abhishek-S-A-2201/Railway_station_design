import torch
import torch_utils
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import pandas as pd
import numpy as np

# device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"

def openings_prediction(space, room_size, capacity, user_per_min):

  # Load the machine learning model
  pipeline = joblib.load("src/models/openings_pipeline.pkl")
  model = torch.load("src/models/openings.pth")

  inputs = pd.DataFrame({
      'space': space, 
      'room_size': room_size, 
      'capacity': capacity, 
      'user_per_min': user_per_min
  }, index=[0])
    
  # Make predictions
  model.eval()
  with torch.inference_mode():
    input_prepped = torch.Tensor(pipeline.transform(inputs)).to(device)
    predictions = model(input_prepped).to(device)
    val = [0]*6
    for prediction in predictions[:10]:
      val[0] = torch.argmax(prediction[5:]).type(torch.int).item() + 1
      for i in range(val[0]):
        val[i+1] = np.round(prediction[i].cpu().numpy(), 2)

  # Return the predictions
  return val

def footprint_prediction(image):
  
  # Apply some image processing to the image
  processed_image = image.resize((256, 256))

    # Load the machine learning model
  model = torch_utils.footprint_generator
  model.load_state_dict(torch.load("/models/footprint.pth"))

  # Make predictions
  model.eval()
  with torch.inference_mode():
    prediction = model(processed_image)

  # Return the predictions
  return prediction

def circulation_prediction(image):
  
  # Apply some image processing to the image
  processed_image = image.resize((256, 256))

    # Load the machine learning model
  model = torch_utils.circulation_detector
  model.load_state_dict(torch.load("/models/footprint.pth"))

  # Make predictions
  model.eval()
  with torch.inference_mode():
    prediction = model(processed_image)

  # Return the processed image
  return prediction