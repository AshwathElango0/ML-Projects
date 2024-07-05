import streamlit as st
import tensorflow as tf
import numpy as np
import cv2


#Loading model trained on and downloaded from Google Colab trained
model_layer = tf.keras.layers.TFSMLayer(r"C:\Users\achus\Downloads\my_model", call_endpoint='serving_default')

#Due to use of newer TensorFlow version, model is loaded in as a layer and a working model is built using this layer
inputs = tf.keras.layers.Input(shape=(None, None, 3))               
x = model_layer(inputs)
model = tf.keras.Model(inputs=inputs, outputs=model_layer.output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Define the class names of the CIFAR-10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#Add information to webpage
st.title("CIFAR-10 Image Classification")
st.write("Upload an image and the model will predict the class.")

#Adding a file uploader to allow the user to provide input images
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    #Storing the image as a file-like object before converting it into an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  #Converting color channels to OpenCV format (BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)          #Converting to standard RGB format
    #Displaying the image using Streamlit, for user to see
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    
    #Classifying the image using the ML model, adding a spinner for aesthetics in case of delay
    with st.spinner('Please wait...'):
        image = tf.expand_dims(image, axis=0)       #Adding batch dimension to the image to meet model input shape requirements
        predictions = model(image)
        predicted_class = class_names[np.argmax(predictions)]       #Getting the predicted class label
    
    #Displaying the result for user to see
    st.success(f"Predicted class: {predicted_class}")
    st.balloons()
