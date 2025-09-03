import streamlit as st
from PIL import Image
import numpy as np
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

st.set_page_config(page_title="COVID-19 X-ray Detector", layout="centered")

st.title("Early COVID-19 detection from Chest X-ray (Demo)")
st.write(
    "Upload a trained Keras model (.h5) and a chest X-ray image (jpg/png). The app will preprocess the image (64x64, rescale) and run the model to show a prediction."
)

# Sidebar: model upload
st.sidebar.markdown("## Model / settings")
model_file = st.sidebar.file_uploader("Upload Keras model (.h5) — full model or weights", type=["h5", "keras"])
use_builtin_arch = st.sidebar.checkbox("Rebuild notebook architecture and load weights if full model fails", value=True)
invert_labels = st.sidebar.checkbox("Invert labels mapping (swap covid/normal)", value=False)

# Image uploader
st.markdown("---")
uploaded_file = st.file_uploader("Upload chest X-ray image", type=["png", "jpg", "jpeg"])

# Helper: build same architecture as notebook
def build_notebook_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

loaded_model = None
model_status = "No model loaded. Upload a .h5 model file in the sidebar or place `model.h5` in the repo root."

# Try auto-loading a local model.h5 if present
import os
local_model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
if os.path.exists(local_model_path):
    try:
        with st.spinner(f"Loading local model from {local_model_path}..."):
            loaded_model = load_model(local_model_path)
            model_status = f"Loaded local model from {local_model_path}"
    except Exception as e_local:
        # ignore; user can upload via sidebar
        st.sidebar.warning(f"Found local model.h5 but failed to load: {e_local}")

if model_file is not None:
    try:
        # try to load full model
        with st.spinner("Loading model — attempting full model load..."):
            model_bytes = model_file.read()
            # Need to wrap bytes into a temporary file for load_model
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
            tmp.write(model_bytes)
            tmp.flush()
            tmp.close()
            loaded_model = load_model(tmp.name)
            model_status = "Full model loaded successfully."
    except Exception as e_full:
        st.sidebar.error(f"Full model load failed: {e_full}")
        if use_builtin_arch:
            try:
                with st.spinner("Rebuilding architecture and loading weights..."):
                    base_model = build_notebook_model()
                    # write bytes to temp file and load weights
                    import tempfile
                    tmpw = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
                    tmpw.write(model_bytes)
                    tmpw.flush()
                    tmpw.close()
                    base_model.load_weights(tmpw.name)
                    loaded_model = base_model
                    model_status = "Model architecture built and weights loaded."
            except Exception as e_weights:
                st.sidebar.error(f"Loading weights also failed: {e_weights}")
                model_status = "Failed to load model or weights from uploaded file."
        else:
            model_status = "Failed to load full model and rebuild option disabled."

st.sidebar.write(model_status)

# Prediction flow
if uploaded_file is not None:
    # show image
    try:
        image = Image.open(uploaded_file).convert('RGB')
    except Exception:
        st.error("Failed to read the uploaded image. Try a different file.")
        st.stop()

    st.image(image, caption='Uploaded image', use_column_width=True)

    if loaded_model is None:
        st.warning("No model loaded. Please upload a Keras .h5 model in the sidebar to run predictions.")
    else:
        # preprocess
        img = image.resize((64,64))
        x = keras_image.img_to_array(img)
        x = x.astype('float32') / 255.0
        x = np.expand_dims(x, axis=0)

        with st.spinner("Running prediction..."):
            preds = loaded_model.predict(x)

        # preds shape: (1,1) for this model
        score = float(preds[0][0])
        # Notebook mapping: result==1 -> 'normal', else 'covid'
        # We'll use threshold 0.5 and allow inversion
        if score >= 0.5:
            label = 'normal'
        else:
            label = 'covid'

        if invert_labels:
            label = 'normal' if label == 'covid' else 'covid'

        st.markdown("## Prediction")
        st.write(f"Label: **{label}**")
        st.write(f"Score (sigmoid output): **{score:.4f}** — higher means closer to 'normal' under the notebook mapping")

        # simple explanation
        st.info("This is a demo. Model performance depends on how it was trained. Do NOT use this for clinical diagnosis.")

else:
    st.info("Upload an X-ray image and a Keras model (.h5) to get a prediction.")

# Footer
st.markdown("---")
st.caption("App created from the repository's notebook. If your model's label mapping is different, try the sidebar 'Invert labels mapping' option.")
