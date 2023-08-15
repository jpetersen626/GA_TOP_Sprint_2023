import streamlit as st
# from tensorflow.keras.models import load_model

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential

from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import smart_resize

# model structure
effnet = EfficientNetB0(include_top = False, weights=None)
effnet.trainable = False
model_t = Sequential()
model_t.add(effnet)
model_t.add(GlobalAveragePooling2D())
model_t.add(Dense(1, activation='sigmoid'))

model_t.load_weights('weights_b0.h5')

uploaded_file = st.file_uploader("Choose an image...", type="jpeg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)


if uploaded_file is not None:
    # Assuming you resize the image and normalize the pixel values
    # image = image.resize((224, 224))
    # image = np.expand_dims(image, axis=0)
    image = smart_resize(image, (224, 224))
    image = np.expand_dims(image, axis = 0)
    prediction = model_t.predict(image)
    prob = prediction[0][0]
    prob = round(prob, 2)
    st.write(f'The probability of Positive is {prob}')
    if prediction[0][0] >= 0.5:
        st.write("It's Positive")
    else:
        st.write("It's Negative")
