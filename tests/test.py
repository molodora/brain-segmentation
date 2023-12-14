import requests
import io
from pathlib import Path

import torch
from PIL import Image

# from app import transform_image, get_predictions


if __name__ == "__main__":
    """
    print("<=====================Loading model==========================>")
    model = torch.jit.load('unet-0-scripted.pt')
    model.eval()
    print(model)
    print()


    print("<===================transform_image test======================>")
    with open("img.tif", 'rb') as f:
        image_bytes = f.read()
        tensor = transform_image(image_bytes=image_bytes)
        print(tensor, tensor.max(), tensor.min())
    print()
    

    print("<==================get_predictions test========================>")
    with open("img.tif", 'rb') as f:
        image_bytes = f.read()
        preds = get_predictions(image_bytes=image_bytes)
        print(preds, preds.max(), preds.min())
    print()

    """
    print("<====================Request test=======================>")
    response = requests.post("http://localhost:8000/predict",
                            files = {"file": open(Path(__file__).parent / "img.tif", "rb")})
    with open(Path(__file__).parent /'test.png', 'wb') as f:
        f.write(response.content)
    