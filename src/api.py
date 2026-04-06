from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_WEIGHTS_PATH = BASE_DIR / "models" / "melhor_modelo_resnet18_finetune_stratified_seed42.pth"
MODEL_METADATA_PATH = BASE_DIR / "models" / "melhor_modelo_resnet18_finetune_stratified_seed42.json"


class TopPrediction(BaseModel):
    classe: str
    probabilidade: float


class PredictionResponse(BaseModel):
    classe_predita: str
    confianca: float
    top_k: list[TopPrediction]


class FERModelService:
    def __init__(self, weights_path: Path, metadata_path: Path) -> None:
        if not weights_path.exists():
            raise FileNotFoundError(f"Pesos do modelo nao encontrados em: {weights_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata do modelo nao encontrada em: {metadata_path}")

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.class_names: list[str] = metadata["class_names"]
        input_size = metadata.get("input_size", [224, 224])
        normalize_mean = metadata.get("normalize_mean", [0.485, 0.456, 0.406])
        normalize_std = metadata.get("normalize_std", [0.229, 0.224, 0.225])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(num_classes=len(self.class_names))
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(tuple(input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize_mean, std=normalize_std),
            ]
        )

    @staticmethod
    def _build_model(num_classes: int) -> nn.Module:
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes),
        )
        return model

    def predict(self, image_bytes: bytes, top_k: int = 3) -> PredictionResponse:
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError("Arquivo enviado nao e uma imagem valida.") from exc

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        top_k = max(1, min(top_k, len(self.class_names)))
        top_values, top_indices = torch.topk(probs, k=top_k)

        predicted_idx = int(top_indices[0].item())
        predicted_class = self.class_names[predicted_idx]
        confidence = float(top_values[0].item())

        top_predictions = [
            TopPrediction(
                classe=self.class_names[int(class_idx.item())],
                probabilidade=float(prob.item()),
            )
            for prob, class_idx in zip(top_values, top_indices)
        ]

        return PredictionResponse(
            classe_predita=predicted_class,
            confianca=confidence,
            top_k=top_predictions,
        )


app = FastAPI(title="FER2013 Emotion API", version="1.0.0")
model_service: FERModelService | None = None


@app.on_event("startup")
def load_model() -> None:
    global model_service
    model_service = FERModelService(MODEL_WEIGHTS_PATH, MODEL_METADATA_PATH)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": str(model_service.device) if model_service else "unloaded",
        "model_loaded": model_service is not None,
    }


@app.get("/classes")
def classes() -> dict:
    if model_service is None:
        raise HTTPException(status_code=503, detail="Modelo ainda nao carregado.")
    return {"classes": model_service.class_names}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), top_k: int = 3) -> PredictionResponse:
    if model_service is None:
        raise HTTPException(status_code=503, detail="Modelo ainda nao carregado.")

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie um arquivo de imagem.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")

    try:
        return model_service.predict(image_bytes=image_bytes, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
