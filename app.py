import pathlib
import platform
from fastai.vision.all import *
import gradio as gr

# Handle path compatibility on Windows
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# Load the model
learn = load_learner("model.pkl")


# Prediction function
def classify_image(img_path):
    try:
        img = PILImage.create(img_path)  # This is all you need
        pred, idx, probs = learn.predict(img)
        return dict(zip(learn.dls.vocab, map(float, probs)))
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {}


# Gradio UI
gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="filepath", label="Upload a pet image"),
    outputs=gr.Label(num_top_classes=3, label="Top 3 Predictions"),
    title="🐶🐱 Pet Breed Classifier",
    description="Upload a dog or cat image to identify its breed using FastAI.",
).launch()
