import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
import streamlit as st
from models import MeanTeacherModel, FixMatchModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    return transform(image).unsqueeze(0)


@st.cache_resource
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    resnet50_encoder = models.resnet50(weights=None)
    resnet50_encoder = nn.Sequential(*list(resnet50_encoder.children())[:-1])
    pretrained_encoder = resnet50_encoder
    num_classes = 4

    if "mean" in checkpoint_path.lower():
        # MeanTeacher
        model = MeanTeacherModel(pretrained_encoder, num_classes)
        model.load_state_dict(checkpoint['student_model_state_dict'])
    else:
        model = FixMatchModel(pretrained_encoder, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    return model
