import matplotlib.cm as cm
import numpy as np
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


@st.cache_data
def get_iucn(scientific_name):
    return birdid.iucn_for(scientific_name)


def overlay_gradcam(image, cam, alpha=0.5):
    """Superpose une carte de chaleur sur une image PIL."""
    size = (birdid.IMAGE_SIZE, birdid.IMAGE_SIZE)
    base = np.array(image.resize(size)).astype(float) / 255

    heat = np.array(Image.fromarray(cam).resize(size, Image.BILINEAR))
    heat = cm.jet(heat)[:, :, :3]

    blended = (1 - alpha) * base + alpha * heat
    return Image.fromarray((blended * 255).astype("uint8"))


model = get_model()
species_by_folder = get_species()
class_names = get_class_names()

with st.sidebar:
    st.header("À propos")
    st.write(
        "Cette application identifie une espèce d'oiseau à partir d'une photo, "
        "puis affiche son statut de conservation UICN."
    )
    st.write(
        "Le modèle a été entraîné sur des photos issues d'iNaturalist. "
        "Il reconnaît 10 espèces européennes et atteint 78,5 % de précision "
        "sur des images qu'il n'a jamais vues."
    )

    st.subheader("Espèces reconnues")
    for folder in class_names:
        species = species_by_folder[folder]
        status = get_iucn(species["scientific_name"])
        if status is None:
            st.write(f"- {species['common_name_fr']}")
            continue
        st.markdown(
            f"<span style='background-color:{status['color']}; color:white; "
            f"padding:0.1rem 0.4rem; border-radius:0.3rem; font-size:0.75rem; "
            f"font-weight:600;'>{status['category']}</span> "
            f"{species['common_name_fr']} "
            f"<span style='opacity:0.6; font-style:italic;'>{species['scientific_name']}</span>",
            unsafe_allow_html=True,
        )

    st.caption(
        "Statuts : IUCN Red List, version 2026-1, périmètre mondial. "
        "9 de ces 10 espèces ont une population en déclin."
    )

st.title("🐦 Bird ID")
st.write("Identification d'oiseaux européens et statut de conservation.")

uploaded_file = st.file_uploader("Choisis une photo d'oiseau", type=["jpg", "jpeg", "png"])

with st.expander("Paramètres avancés"):
    threshold = st.slider(
        "Seuil de confiance",
        min_value=0.0,
        max_value=0.95,
        value=DEFAULT_THRESHOLD,
        step=0.05,
    )
    st.caption(
        "En dessous de ce seuil, l'application n'affirme pas d'identification. "
        "À 0,70, la précision passe de 78 % à 91 % sur le jeu de validation, "
        "au prix d'une abstention sur environ une photo sur trois."
    )

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    tensor = birdid.inference_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    top_probs, top_indices = probabilities.topk(TOP_K)

    best_probability = top_probs[0].item()
    best_folder = class_names[top_indices[0].item()]
    best_species = species_by_folder[best_folder]

    cam = birdid.gradcam(model, tensor, top_indices[0].item())
    overlay = overlay_gradcam(image, cam)

    col_left, col_right = st.columns(2)
    with col_left:
        st.image(image, caption="Photo envoyée", use_container_width=True)
    with col_right:
        st.image(overlay, caption="Zones ayant pesé dans la décision", use_container_width=True)

    st.caption(
        "Les zones chaudes (rouge, jaune) ont le plus contribué à la prédiction, "
        "les zones froides (bleu) très peu. Si la chaleur n'est pas sur l'oiseau, "
        "le modèle s'appuie sur le décor : la prédiction est alors peu fiable."
    )

    if best_probability >= threshold:
        st.success(
            f"**{best_species['common_name_fr']}** — *{best_species['scientific_name']}*  \n"
            f"Confiance : {best_probability:.1%}"
        )

        status = get_iucn(best_species["scientific_name"])
        if status is not None:
            trend_labels = {
                "increasing": "en augmentation",
                "stable": "stable",
                "decreasing": "en déclin",
            }
            trend = trend_labels.get(status["trend"], status["trend"])

            st.subheader("Statut de conservation")
            st.markdown(
                f"<div style='background-color:{status['color']}; color:white; "
                f"padding:0.6rem 1rem; border-radius:0.4rem; font-weight:600; "
                f"display:inline-block;'>{status['category']} — {status['label']}</div>",
                unsafe_allow_html=True,
            )
            st.write("")
            st.write(status["description"])
            st.caption(
                f"Population {trend} · Évaluation {status['assessed']} · "
                f"Source : IUCN Red List (version 2026-1, périmètre mondial)"
            )
    else:
        st.warning(
            f"Confiance insuffisante ({best_probability:.1%}) pour affirmer une identification. "
            "L'oiseau est peut-être trop loin ou trop peu visible. "
            "Essaie une photo plus rapprochée ou mieux cadrée."
        )

    st.info(
        "Le modèle se trompe dans environ 1 cas sur 10 au-dessus du seuil de confiance, "
        "parfois avec une confiance élevée. Les silhouettes en vol et les oiseaux "
        "photographiés de loin sont les cas les plus difficiles. "
        "Vérifiez auprès d'une source ornithologique avant toute conclusion, "
        "en particulier sur le statut de conservation."
    )
        
    st.subheader(f"{TOP_K} espèces les plus probables")
    for rank, (probability, index) in enumerate(zip(top_probs.tolist(), top_indices.tolist()), start=1):
        folder = class_names[index]
        species = species_by_folder[folder]
        st.write(f"**{rank}. {species['common_name_fr']}** — *{species['scientific_name']}*")
        st.progress(probability, text=f"{probability:.1%}")