import keras
import numpy as np
from PIL import Image

def load_model(path: str) -> keras.Sequential:
    return keras.models.load_model(path)

def predict_digit(model: keras.Sequential, data_point: list) -> str:
    data_point = np.array(data_point).reshape((1,784))
    prediction = model.predict(data_point)
    digit = np.argmax(prediction)
    return str(digit)

def format_image(image: Image) -> list:
    image = image.resize((28, 28))
    image = image.convert('L')  # Convert to grayscale
    data_point = [pixel / 255.0 for pixel in list(image.getdata())]
    return data_point