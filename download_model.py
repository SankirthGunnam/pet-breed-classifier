# download the model trained on Oxford Pets dataset
from fastai.vision.all import *

learn = load_learner(
    "https://github.com/fastai/fastai/releases/download/2021.09.12/export.pkl"
)
learn.export("model.pkl")  # Save locally
