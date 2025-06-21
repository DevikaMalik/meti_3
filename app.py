# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.datasets import mnist
# import random
# import matplotlib.pyplot as plt

# st.title("Handwritten digit Generator (MNIST)")

# model = load_model('digit_generator_model.h5')

# (_, _) , (x_test, y_test) = mnist.load_data()
# x_test = x_test / 255.0

# digit = st.selectbox("Select a digit (0 - 9):", list(range(10)))

# generate = st.button("Gnerate 5 images")

# if generate :
#     st.subheader(f"Generated images for digit {digit}:")
#     digit_images = x_test[y_test == digit]
#     selected_images = random.sample(list(digit_images), 5)
#     cols = st.columns(5)
#     for i in range(5):
#         with cols[i]:
#             st.image(selected_images[i], width = 80, clamp = True, channels = "L")

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

st.title("Handwritten Digit Generator (MNIST-like)")
st.write("This app generates new, random handwritten digit images using a pre-trained GAN generator model.")

# Load the generator model
# IMPORTANT: Ensure 'digit_generator_model.h5' is actually a GENERATOR model
#            that takes a latent vector (noise) as input and outputs an image.
#            If it's a classifier or something else, this will not work.
try:
    generator_model = load_model('digit_generator_model.h5')
    # Get the expected latent dimension from the model's input shape
    # Assuming the input is a 2D tensor (batch_size, latent_dim)
    latent_dim = generator_model.input_shape[1]
except Exception as e:
    st.error(f"Error loading the generator model: {e}")
    st.info("Please make sure 'digit_generator_model.h5' is a valid Keras model for generating images.")
    st.stop() # Stop the app if model loading fails

num_images_to_generate = st.slider("Number of images to generate:", 1, 10, 5)

generate_button = st.button("Generate New Images")

if generate_button:
    st.subheader(f"Generated {num_images_to_generate} new images:")

    # Generate random latent vectors (noise)
    noise = tf.random.normal([num_images_to_generate, latent_dim])

    # Use the generator model to create images
    generated_images = generator_model.predict(noise)

    # Assuming the generator outputs images in a range like [-1, 1] or [0, 1]
    # If it's [-1, 1], scale to [0, 1] for display
    if np.min(generated_images) < 0:
        generated_images = (generated_images + 1) / 2.0

    # Reshape to (batch, height, width) if it's (batch, height, width, channels)
    if generated_images.shape[-1] == 1: # Grayscale image
        generated_images = generated_images.squeeze(-1) # Remove the channel dimension

    cols = st.columns(num_images_to_generate)
    for i in range(num_images_to_generate):
        with cols[i]:
            # Matplotlib is good for displaying raw numpy arrays as images
            fig, ax = plt.subplots(figsize=(1, 1)) # Small figure size
            ax.imshow(generated_images[i], cmap='gray')
            ax.axis('off') # Hide axes
            st.pyplot(fig)
            plt.close(fig) # Close the figure to free memory


