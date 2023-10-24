import streamlit as st
from PIL import Image
from utils import *


# Create a Streamlit app
st.title('🚂 Railway Station Design 🚂')

# Display the function selection menu
function_selection = st.selectbox('Select a function:', ['Openings Prediction', 'Footprint Generation', 'Circulation Detection'])

# Display the UI for the selected function
if function_selection == 'Openings Prediction':
  # Display the input fields
  input_1 = st.number_input('Input 1:')
  input_2 = st.number_input('Input 2:')

  # If the user clicks the "Predict" button, make a prediction
  if st.button('Predict'):
    predictions = openings_prediction(input_1, input_2)

    # Display the predictions
    st.write('Predictions:')
    for prediction in predictions:
      st.write(prediction)

elif function_selection == 'Footprint Generation':
  # Display the image upload field
  uploaded_file = st.file_uploader('Upload an image:')

  # If the user uploads an image, process it
  if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Process the image
    processed_image = footprint_prediction(image)

    # Display the processed image
    st.image(processed_image, caption='Footprint image')

    # Add a button to download the processed image
    st.download_button('Download footprint image', processed_image, mimetype='image/png')

elif function_selection == 'Circulation Detection':
  # Display the image upload field
  uploaded_file = st.file_uploader('Upload an image:')

  # If the user uploads an image, process it
  if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Process the image
    processed_image = circulation_prediction(image)

    # Display the processed image
    st.image(processed_image, caption='Circulation image')

    # Add a button to download the processed image
    st.download_button('Download circulation image', processed_image, mimetype='image/png')

else:
  st.write('Please select a function.')