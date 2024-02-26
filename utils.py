from PIL import Image, ImageChops, ImageEnhance
import numpy as np

def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'

    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize((128, 128))).flatten() / 255.0

def prepare_image_for_prediction(image_path):
    # Load and prepare the image
    image = prepare_image(image_path)
    image = image.reshape(-1, 128, 128, 3)
    return image

def predict_single_image(image_path, model):
    # Prepare the image for prediction
    image = prepare_image_for_prediction(image_path)
    # Make predictions
    y_pred = model.predict(image)
    class_names = ['fake', 'real']
    y_pred_class = np.argmax(y_pred, axis=1)[0]
    confidence = np.amax(y_pred) * 100
    return class_names[y_pred_class], confidence
