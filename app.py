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

# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt

# st.title("Handwritten Digit Generator (MNIST-like)")
# st.write("This app generates new, random handwritten digit images using a pre-trained GAN generator model.")

# # Load the generator model
# # IMPORTANT: Ensure 'digit_generator_model.h5' is actually a GENERATOR model
# #            that takes a latent vector (noise) as input and outputs an image.
# #            If it's a classifier or something else, this will not work.
# try:
#     generator_model = load_model('digit_generator_model.h5')
#     # Get the expected latent dimension from the model's input shape
#     # Assuming the input is a 2D tensor (batch_size, latent_dim)
#     latent_dim = generator_model.input_shape[1]
# except Exception as e:
#     st.error(f"Error loading the generator model: {e}")
#     st.info("Please make sure 'digit_generator_model.h5' is a valid Keras model for generating images.")
#     st.stop() # Stop the app if model loading fails

# num_images_to_generate = st.slider("Number of images to generate:", 1, 10, 5)

# generate_button = st.button("Generate New Images")

# if generate_button:
#     st.subheader(f"Generated {num_images_to_generate} new images:")

#     # Generate random latent vectors (noise)
#     noise = tf.random.normal([num_images_to_generate, latent_dim])

#     # Use the generator model to create images
#     generated_images = generator_model.predict(noise)

#     # Assuming the generator outputs images in a range like [-1, 1] or [0, 1]
#     # If it's [-1, 1], scale to [0, 1] for display
#     if np.min(generated_images) < 0:
#         generated_images = (generated_images + 1) / 2.0

#     # Reshape to (batch, height, width) if it's (batch, height, width, channels)
#     if generated_images.shape[-1] == 1: # Grayscale image
#         generated_images = generated_images.squeeze(-1) # Remove the channel dimension

#     cols = st.columns(num_images_to_generate)
#     for i in range(num_images_to_generate):
#         with cols[i]:
#             # Matplotlib is good for displaying raw numpy arrays as images
#             fig, ax = plt.subplots(figsize=(1, 1)) # Small figure size
#             ax.imshow(generated_images[i], cmap='gray')
#             ax.axis('off') # Hide axes
#             st.pyplot(fig)
#             plt.close(fig) # Close the figure to free memory

# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
# import os

# # Set Streamlit page configuration for better aesthetics
# st.set_page_config(
#     page_title="Handwritten Digit Generator",
#     page_icon="✨",
#     layout="centered"
# )

# # --- Streamlit UI Setup ---
# st.title("✨ Handwritten Digit Generator (GAN)")
# st.markdown("""
#     This application uses a Generative Adversarial Network (GAN) to create **new**, synthetic handwritten digit images.
    
#     **Instructions:**
#     1. First, make sure you have run your `train_gan_generator.py` script to train the GAN and save the `digit_generator_model.h5` file.
#        In a Colab environment, you would run that script in a separate cell, and the model file will be saved
#        in the current working directory or a specified subfolder (like 'models').
#     2. Ensure the `digit_generator_model.h5` file is accessible to this `app.py` script.
#     3. Use the slider below to select how many images you want to generate.
#     4. Click the 'Generate New Images' button!
# """)

# # --- Model Loading ---
# # Define the path where the generator model is expected to be found.
# # If running in Colab, this path should be relative to where your .ipynb file is,
# # or where you've uploaded/saved the 'models' directory.
# model_path = os.path.join('models', 'digit_generator_model.h5')

# @st.cache_resource # Cache the model to avoid reloading on every rerun of the app
# def load_generator_model(path):
#     """
#     Loads the pre-trained Keras generator model from the specified path.
#     Includes error handling for robust loading.
#     """
#     if not os.path.exists(path):
#         st.error(f"Model file not found: {path}. Please ensure 'train_gan_generator.py' was run and the model is saved correctly.")
#         st.info("In Google Colab, you might need to upload the 'models' directory or ensure it's created and the model is saved to the right place.")
#         return None
#     try:
#         model = load_model(path)
#         # Basic check to ensure the loaded model has an input shape,
#         # which is characteristic of a functional Keras model.
#         if model.input_shape is None:
#              st.error("The loaded model does not appear to have a defined input shape. It might not be a valid Keras model for generation.")
#              return None
#         return model
#     except Exception as e:
#         st.error(f"Error loading the generator model from {path}. Details: {e}")
#         st.info("This could be due to a corrupted file, an incompatible TensorFlow version, or the model not being saved correctly.")
#         return None

# # Define the latent dimension (input size for the generator).
# # This MUST match the NOISE_DIM used in your `train_gan_generator.py` script.
# NOISE_DIM = 100 

# # Attempt to load the generator model
# generator_model = load_generator_model(model_path)

# # If the model failed to load, stop the Streamlit app here.
# if generator_model is None:
#     st.warning("Generator model not loaded. Please ensure `train_gan_generator.py` has successfully trained and saved the model to the `models` directory.")
#     st.stop() 

# # Verify the latent dimension of the loaded model against the expected NOISE_DIM.
# # This helps catch inconsistencies between training and inference code.
# if generator_model.input_shape[1] != NOISE_DIM:
#     st.error(f"Mismatch in expected noise dimension! The loaded model expects {generator_model.input_shape[1]} but the app uses {NOISE_DIM}. Please ensure they match in both scripts.")
#     st.stop()


# # --- Image Generation UI Elements ---
# num_images_to_generate = st.slider("Number of images to generate:", 1, 10, 5)

# generate_button = st.button("Generate New Images")

# if generate_button:
#     st.subheader(f"Generated {num_images_to_generate} new images:")

#     # Generate random latent vectors (noise) as input for the generator.
#     # The shape is (batch_size, noise_dimension).
#     noise = tf.random.normal([num_images_to_generate, NOISE_DIM])

#     # Use the generator model to create images from the generated noise.
#     # `predict` is used for inference.
#     generated_images = generator_model.predict(noise)

#     # --- Post-processing for Display ---
#     # The GAN generator, using 'tanh' activation in its output layer,
#     # produces pixel values in the range [-1, 1].
#     # For displaying with Matplotlib or Streamlit's image function,
#     # we need to scale them back to the conventional [0, 1] range.
#     generated_images = (generated_images * 0.5) + 0.5 # Scale from [-1, 1] to [0, 1]

#     # MNIST images are grayscale with a single channel (e.g., shape (28, 28, 1)).
#     # Matplotlib's `imshow` and Streamlit's `st.image` work well with
#     # 2D grayscale arrays (shape (height, width)).
#     # If the last dimension is 1, we can remove it using `squeeze`.
#     if generated_images.shape[-1] == 1:
#         generated_images = generated_images.squeeze(-1) # Remove the last dimension if it's 1

#     # Display generated images in a responsive column layout.
#     # `st.columns` creates an iterator for context managers.
#     cols = st.columns(num_images_to_generate)
#     for i in range(num_images_to_generate):
#         with cols[i]: # Enter the context of each column
#             # Create a Matplotlib figure for each image. This allows fine-grained control
#             # over the image plot and prevents display issues in Streamlit.
#             fig, ax = plt.subplots(figsize=(2, 2)) # Set a small figure size for compact display
#             ax.imshow(generated_images[i], cmap='gray') # Display the image with a grayscale colormap
#             ax.axis('off') # Turn off the axes for a cleaner image display (no ticks, labels)
#             st.pyplot(fig) # Render the Matplotlib figure in Streamlit
#             plt.close(fig) # Crucially, close the Matplotlib figure to free up memory and
#                             # prevent it from being rendered again on subsequent Streamlit reruns.


import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="✨",
    layout="centered"
)

# --- Streamlit UI Setup ---
st.title("✨ Handwritten Digit Generator (GAN)")
st.markdown("""
    This application uses a Generative Adversarial Network (GAN) to create **new**, synthetic handwritten digit images.
    
    **Instructions:**
    1. First, make sure you have run your `train_gan_generator.py` script to train the GAN and save the `digit_generator_model.h5` file.
       This model file should be in the **same directory** as this `app.py` script.
    2. Use the slider below to select how many images you want to generate.
    3. Click the 'Generate New Images' button!
""")

# --- Model Loading ---
# Define the path where the generator model is expected to be found.
# Now, it's expected to be directly in the same directory as app.py.
model_path = 'digit_generator_model.h5'

@st.cache_resource # Cache the model to avoid reloading on every rerun of the app
def load_generator_model(path):
    """
    Loads the pre-trained Keras generator model from the specified path.
    Includes error handling for robust loading.
    """
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}. Please ensure 'train_gan_generator.py' was run and the model is saved correctly in the same directory as app.py.")
        return None
    try:
        model = load_model(path)
        # Basic check to ensure the loaded model has an input shape,
        # which is characteristic of a functional Keras model.
        if model.input_shape is None:
             st.error("The loaded model does not appear to have a defined input shape. It might not be a valid Keras model for generation.")
             return None
        return model
    except Exception as e:
        st.error(f"Error loading the generator model from {path}. Details: {e}")
        st.info("This could be due to a corrupted file, an incompatible TensorFlow version, or the model not being saved correctly.")
        return None

# Define the latent dimension (input size for the generator).
# This MUST match the NOISE_DIM used in your `train_gan_generator.py` script.
NOISE_DIM = 100 

# Attempt to load the generator model
generator_model = load_generator_model(model_path)

# If the model failed to load, stop the Streamlit app here.
if generator_model is None:
    st.warning("Generator model not loaded. Please ensure `train_gan_generator.py` has successfully trained and saved the model in the same directory as app.py.")
    st.stop() 

# Verify the latent dimension of the loaded model against the expected NOISE_DIM.
# This helps catch inconsistencies between training and inference code.
if generator_model.input_shape[1] != NOISE_DIM:
    st.error(f"Mismatch in expected noise dimension! The loaded model expects {generator_model.input_shape[1]} but the app uses {NOISE_DIM}. Please ensure they match in both scripts.")
    st.stop()


# --- Image Generation UI Elements ---
num_images_to_generate = st.slider("Number of images to generate:", 1, 10, 5)

generate_button = st.button("Generate New Images")

if generate_button:
    st.subheader(f"Generated {num_images_to_generate} new images:")

    # Generate random latent vectors (noise) as input for the generator.
    # The shape is (batch_size, noise_dimension).
    noise = tf.random.normal([num_images_to_generate, NOISE_DIM])

    # Use the generator model to create images from the generated noise.
    # `predict` is used for inference.
    generated_images = generator_model.predict(noise)

    # --- Post-processing for Display ---
    # The GAN generator, using 'tanh' activation in its output layer,
    # produces pixel values in the range [-1, 1].
    # For displaying with Matplotlib or Streamlit's image function,
    # we need to scale them back to the conventional [0, 1] range.
    generated_images = (generated_images * 0.5) + 0.5 # Scale from [-1, 1] to [0, 1]

    # MNIST images are grayscale with a single channel (e.g., shape (28, 28, 1)).
    # Matplotlib's `imshow` and Streamlit's `st.image` work well with
    # 2D grayscale arrays (shape (height, width)).
    # If the last dimension is 1, we can remove it using `squeeze`.
    if generated_images.shape[-1] == 1:
        generated_images = generated_images.squeeze(-1) # Remove the last dimension if it's 1

    # Display generated images in a responsive column layout.
    # `st.columns` creates an iterator for context managers.
    cols = st.columns(num_images_to_generate)
    for i in range(num_images_to_generate):
        with cols[i]: # Enter the context of each column
            # Create a Matplotlib figure for each image. This allows fine-grained control
            # over the image plot and prevents display issues in Streamlit.
            fig, ax = plt.subplots(figsize=(2, 2)) # Set a small figure size for compact display
            ax.imshow(generated_images[i], cmap='gray') # Display the image with a grayscale colormap
            ax.axis('off') # Turn off the axes for a cleaner image display (no ticks, labels)
            st.pyplot(fig) # Render the Matplotlib figure in Streamlit
            plt.close(fig) # Crucially, close the Matplotlib figure to free up memory and
                            # prevent it from being rendered again on subsequent Streamlit reruns.

