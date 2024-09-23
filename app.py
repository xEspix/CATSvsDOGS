import streamlit as st
from PIL import Image
import numpy as np
import pickle

model=pickle.load(open('CATSDOGSCNN.pkl', 'rb'))

st.title("CATS VS DOGS CLASSIFIER")

st.write("Upload an image of a cat or dog, and this app will display the pixel values in the three color channels (RGB).")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:

    image = Image.open(uploaded_image).resize((256,256))
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    image=image.convert('RGB')
    img_array = np.array(image)
    img_array=img_array/255
    
    test_input=img_array.reshape(1, 256, 256, 3)

    y_pred=model.predict(test_input)
    st.write(y_pred)
    if(y_pred[0][0]<0.5):
        st.write("PREDICTED : CAT")
    else:
        st.write("PREDICTED : DOG")