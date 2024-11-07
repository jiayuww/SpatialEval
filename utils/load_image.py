import requests
from PIL import Image, PngImagePlugin, JpegImagePlugin
from io import BytesIO

def load_image(image_file):
    if isinstance(image_file, str) and (image_file.startswith('http://') or image_file.startswith('https://')):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    elif isinstance(image_file, (JpegImagePlugin.JpegImageFile, PngImagePlugin.PngImageFile)):
        image = image_file.convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image