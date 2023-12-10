import io
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile
from fastapi.responses import Response


model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(__file__).parent / "unet-scripted.pt"
    global model
    model = torch.jit.load(model_path)
    model.eval()
    yield
    del model


app = FastAPI(lifespan=lifespan)


def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ])

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)


def get_predictions(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = torch.sigmoid(model(tensor))
    outputs = (outputs > 0.5).float()
    return outputs[0]


@app.get("/")
def route():
    return {"health_check": "OK"}


@app.post("/predict", response_class=Response)
async def predict(file: UploadFile):
    bytes = await file.read()
    prediction = get_predictions(bytes)

    image = transforms.ToPILImage()(prediction)
    with io.BytesIO() as buf:
        image.save(buf, format='PNG')
        image_bytes = buf.getvalue()

    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    return Response(image_bytes, headers=headers, media_type='image/png')