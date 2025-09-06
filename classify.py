import streamlit as st
from PIL import Image, ImageStat
import torch
from model import preprocess_image, load_model
import pandas as pd
from io import BytesIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("Brain Tumour Classification")

model_mapping = {
    "FixMatch Model": "best_fixmatch_model.pth",
    "MeanTeacher Model": "best_mean_teacher_model.pth",
}

display_names = list(model_mapping.keys())

selected_display_name = st.selectbox(
    "Choose a model:",
    options=display_names
)

selected_model_path = model_mapping[selected_display_name]

loaded_model = load_model(selected_model_path)

uploaded_files = st.file_uploader(
    "Upload Brain MRI",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    results = []

    for i, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file).convert('RGB')

        input_tensor = preprocess_image(image).to(device)
        with torch.no_grad():
            outputs = loaded_model(input_tensor).cpu()
            pred_class = outputs.argmax(dim=1).item()

        class_names = ["glioma", "meningioma", "no tumour", "pituitary"]
        pred_label = class_names[pred_class]

        image_bytes = BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)

        results.append({
            "filename": uploaded_file.name,
            "predicted_tumour_type": pred_label,
            "image_bytes": image_bytes.getvalue()
        })

        st.empty().text(f"Processed {i + 1} of {len(uploaded_files)} images")

    if results:
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        st.write(f"**{len(results)}** / **{len(uploaded_files)}** predicted")

        for r in results:
            cols = st.columns([1, 2])
            with cols[0]:
                st.image(Image.open(BytesIO(r["image_bytes"])), width=200)
            with cols[1]:
                st.write(f"**{r['filename']}**")
                st.write(f"Predicted Tumour Type: **{r['predicted_tumour_type']}**")

        df = pd.DataFrame([{k: v for k, v in r.items() if k != 'image_bytes'} for r in results])
        csv = df.to_csv(index=False).encode()
        st.download_button("Download predictions CSV", data=csv,
                           file_name="brain_tumour_predictions.csv", mime="text/csv")
    else:
        st.info("No valid MRI images were uploaded.")
