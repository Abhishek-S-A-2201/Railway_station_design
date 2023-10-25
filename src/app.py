import streamlit as st
from PIL import Image
from utils import *
from torch_utils import *
import numpy as np


# Create a Streamlit app
st.title('🚂 Railway Station Design 🚂')

# Display the function selection menu
function_selection = st.selectbox('Select a function:', ['Openings Prediction', 'Footprint Generation', 'Circulation Detection'])

# Display the UI for the selected function
if function_selection == 'Openings Prediction':

  space = st.selectbox(
           "What are you working on?",
           ("Area for standing", "concourse", "assembly hall", "check in que", "Office", "Conferences", "dining room", "restaurant", "Hold room", "wait/circulate", ""),
           index=None,
           placeholder="Select space...",
    )

  # Display the input fields
  room_size = st.number_input('Room Size:')
  capacity = st.number_input('Capacity:')
  user_per_min = st.number_input('Users per minute:')

  # If the user clicks the "Predict" button, make a prediction
  if st.button('Predict'):
    predictions = openings_prediction(space, room_size, capacity, user_per_min)

    for prediction in predictions:
      # Display the predictions
      st.write('Predictions:')
      st.write(f'No of openings: {torch.round(prediction[0]).type(torch.int).item()}')
      for i in range(torch.round(prediction[0]).type(torch.int).item()):
        st.write(f'Width {i+1}: {prediction[i+1].item():.2f}')

# elif function_selection == 'Footprint Generation':
#   # Display the image upload field
#   uploaded_file = st.file_uploader('Upload an image:')

#   # If the user uploads an image, process it
#   if uploaded_file is not None:
#     image = Image.open(uploaded_file)

#     # Process the image
#     processed_image = footprint_prediction(image)

#     # Display the processed image
#     st.image(processed_image, caption='Footprint image')

#     # Add a button to download the processed image
#     st.download_button('Download footprint image', processed_image, mimetype='image/png')

# elif function_selection == 'Circulation Detection':
#   # Display the image upload field
#   uploaded_file = st.file_uploader('Upload an image:')

#   # If the user uploads an image, process it
#   if uploaded_file is not None:
#     image = Image.open(uploaded_file)

#     # Process the image
#     processed_image = circulation_prediction(image)

#     # Display the processed image
#     st.image(processed_image, caption='Circulation image')

#     # Add a button to download the processed image
#     st.download_button('Download circulation image', processed_image, mimetype='image/png')

else:
  st.write('Please select a function.')