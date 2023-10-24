import torch
import torch_utils

def openings_prediction(input_1, input_2):

  # Load the machine learning model
  model = torch_utils.openings_predictor
  model.load_state_dict(torch.load("/models/openings.pth"))

  # Make predictions
  model.eval()
  with torch.inference_mode():
    prediction = model([[input_1, input_2]])

  # Return the predictions
  return prediction

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