import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

import birdid

EXPERIMENT_NAME = "unfreeze3_earlystop"
TOP_K = 3
DEFAULT_THRESHOLD = 0.70

st.set_page_config(page_title="Bird ID", page_icon="🐦")


@st.cache_resource
def get_model():
    return birdid.load_model(EXPERIMENT_NAME)


@st.cache_data
def get_species():
    return birdid.load_species()


@st.cache_data
def get_class_names():
    return birdid.class_names()


model = get_model()
species_by_folder = get_species()
class_names = get_class_names()

st.title("🐦 Bird ID")
st.write("Identification d'oiseaux européens et statut de conservation.")

with st.sidebar:
    st.header("Réglages")
    threshold = st.slider(
        "Seuil de confiance",
        min_value=0.0,
        max_value=0.95,
        value=DEFAULT_THRESHOLD,
        step=0.05,
    )
    st.caption(
        "En dessous du seuil, l'app n'affirme pas d'identification. "
        "À 0,70, la précision passe de 78 % à 91 % sur le jeu de validation."
    )

uploaded_file = st.file_uploader("Choisis une photo d'oiseau", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Photo envoyée", width=400)

    tensor = birdid.inference_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    top_probs, top_indices = probabilities.topk(TOP_K)

    best_probability = top_probs[0].item()
    best_folder = class_names[top_indices[0].item()]
    best_species = species_by_folder[best_folder]

    if best_probability >= threshold:
        st.success(
            f"**{best_species['common_name_fr']}** — *{best_species['scientific_name']}*  \n"
            f"Confiance : {best_probability:.1%}"
        )
    else:
        st.warning(
            f"Confiance insuffisante ({best_probability:.1%}) pour affirmer une identification. "
            "L'oiseau est peut-être trop loin ou trop peu visible. "
            "Essaie une photo plus rapprochée ou mieux cadrée."
        )

    st.subheader(f"{TOP_K} espèces les plus probables")
    for rank, (probability, index) in enumerate(zip(top_probs.tolist(), top_indices.tolist()), start=1):
        folder = class_names[index]
        species = species_by_folder[folder]
        st.write(f"**{rank}. {species['common_name_fr']}** — *{species['scientific_name']}*")
        st.progress(probability, text=f"{probability:.1%}")