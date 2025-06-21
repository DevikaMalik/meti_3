import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import random
import matplotlib.pyplot as plt

st.title("Handwritten digit Generator (MNIST)")

model = load_model('digit_generator_model.h5')

(_, _) , (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0

digit = st.selectbox("Select a digit (0 - 9):", list(range(10)))

generate = st.button("Gnerate 5 images")

if generate :
    st.subheader(f"Generated images for digit {digit}:")
    digit_images = x_test[y_test == digit]
    selected_images = random.sample(list(digit_images), 5)
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(selected_images[i], width = 80, clamp = True, channels = "L")


