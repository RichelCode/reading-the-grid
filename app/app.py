"""Streamlit app for the Reading the Grid solar-cell fault classifier.

Loads the fine-tuned ResNet18, accepts an uploaded electroluminescence image or a bundled
example, and reports the predicted class, confidence, and a Grad-CAM attention overlay.

Run with:  streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# Locate the repo root by searching upward for src/model.py, then expose src/ for import.
_here = Path(__file__).resolve()
REPO_ROOT = next(c for c in _here.parents if (c / "src" / "model.py").exists())
sys.path.insert(0, str(REPO_ROOT / "src"))

from model import build_model, get_device  # noqa: E402
from dataset import get_transforms  # noqa: E402
from pytorch_grad_cam import GradCAM  # noqa: E402
from pytorch_grad_cam.utils.image import show_cam_on_image  # noqa: E402
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # noqa: E402

CLASS_NAMES = ["healthy", "faulty"]
CHECKPOINT = REPO_ROOT / "models" / "best_model_finetuned.pth"
EXAMPLES_DIR = _here.parent / "examples"
# Below this softmax confidence the prediction is reported as a borderline call.
CONFIDENCE_THRESHOLD = 0.65

st.set_page_config(page_title="Reading the Grid", layout="centered")

# --- Visual identity ---------------------------------------------------------------
# Dark instrument-panel theme: charcoal surfaces, off-white text, semantic green/amber
# verdict accents, electric-blue brand accent. Inter for UI, JetBrains Mono for readouts.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
        --bg: #0E1116;
        --surface: #161B22;
        --border: #30363D;
        --text: #E6EDF3;
        --muted: #8B949E;
        --healthy: #2EA043;
        --healthy-bright: #3FB950;
        --faulty: #D29922;
        --faulty-bright: #E3B341;
        --brand: #388BFD;
    }

    .stApp { background: var(--bg); color: var(--text); }
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    .block-container { max-width: 760px; padding-top: 2.2rem; padding-bottom: 4rem; }

    /* Hide default Streamlit chrome for a cleaner product surface. */
    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
        visibility: hidden;
    }
    header { background: transparent !important; }

    /* Header */
    .kicker {
        font-family: 'JetBrains Mono', ui-monospace, monospace;
        font-size: 0.72rem; letter-spacing: 0.16em; text-transform: uppercase;
        color: var(--brand); margin-bottom: 0.5rem;
    }
    .brand { font-size: 2rem; font-weight: 700; line-height: 1.1; margin: 0; }
    .tagline { color: var(--muted); font-size: 1.02rem; margin-top: 0.4rem; }
    .rule { height: 1px; background: var(--border); margin: 1.6rem 0; border: 0; }

    /* Section labels */
    .section-label {
        font-family: 'JetBrains Mono', ui-monospace, monospace;
        font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;
        color: var(--muted); margin: 0.2rem 0 0.5rem;
    }

    /* Framed images */
    [data-testid="stImage"] img {
        border-radius: 8px; border: 1px solid var(--border);
    }

    /* Example buttons */
    .stButton > button {
        width: 100%; background: var(--surface); color: var(--text);
        border: 1px solid var(--border); border-radius: 8px;
        font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 0.78rem;
        padding: 0.3rem 0.2rem;
    }
    .stButton > button:hover { border-color: var(--brand); color: var(--brand); }

    /* File uploader surface */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--surface); border: 1px dashed var(--border); border-radius: 10px;
    }

    /* Verdict banner */
    .verdict {
        background: var(--surface); border: 1px solid var(--border);
        border-left: 4px solid var(--border); border-radius: 10px;
        padding: 16px 20px; margin: 0.4rem 0;
    }
    .verdict-healthy { border-left-color: var(--healthy); background: rgba(46,160,67,0.08); }
    .verdict-faulty { border-left-color: var(--faulty); background: rgba(210,153,34,0.09); }
    .verdict-row {
        display: flex; align-items: baseline; justify-content: space-between; gap: 12px;
    }
    .verdict-label { font-size: 1.5rem; font-weight: 700; }
    .verdict-healthy .verdict-label { color: var(--healthy-bright); }
    .verdict-faulty .verdict-label { color: var(--faulty-bright); }
    .verdict-readout {
        font-family: 'JetBrains Mono', ui-monospace, monospace;
        font-size: 2rem; font-weight: 500;
    }
    .verdict-sub {
        font-family: 'JetBrains Mono', ui-monospace, monospace;
        font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
        color: var(--muted); margin-top: 2px;
    }
    .verdict-note { margin-top: 10px; font-size: 0.9rem; color: var(--faulty-bright); }

    .heatmap-caption { color: var(--muted); font-size: 0.86rem; margin-top: 0.5rem; }
    .placeholder { color: var(--muted); font-size: 0.95rem; padding: 0.5rem 0 1rem; }

    a, a:visited { color: var(--brand); }

    @media (max-width: 480px) {
        .brand { font-size: 1.6rem; }
        .verdict-readout { font-size: 1.5rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model() -> tuple[torch.nn.Module, torch.device]:
    """Load the fine-tuned model once and reuse it across reruns."""
    device = get_device()
    model = build_model(num_classes=2, freeze_backbone=True)
    # Unfreeze layer4 to match the fine-tuned checkpoint and to let gradients reach the
    # Grad-CAM target layer.
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.to(device)
    model.eval()
    return model, device


# Normalized transform for the model; resize-only transform for the display/overlay image.
_, eval_transform = get_transforms()
display_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def analyze(image: Image.Image, model: torch.nn.Module, device: torch.device):
    """Return (predicted_index, confidence, display_rgb, overlay) for one image."""
    input_tensor = eval_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)[0]
    pred = int(probs.argmax())
    confidence = float(probs[pred])

    # Grad-CAM for the predicted class, targeting the last conv block. The context manager
    # releases the hooks after use so repeated runs do not accumulate them.
    with GradCAM(model=model, target_layers=[model.layer4[-1]]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=[ClassifierOutputTarget(pred)])[0]

    display_rgb = display_transform(image).permute(1, 2, 0).numpy().astype(np.float32)
    overlay = show_cam_on_image(display_rgb, grayscale_cam, use_rgb=True)
    return pred, confidence, display_rgb, overlay


def render_verdict(pred: int, confidence: float) -> None:
    """Render the status banner in the semantic color for the prediction."""
    semantic = "healthy" if pred == 0 else "faulty"
    label = CLASS_NAMES[pred].capitalize()
    note = ""
    if confidence < CONFIDENCE_THRESHOLD:
        note = '<div class="verdict-note">Borderline call — low confidence.</div>'
    st.markdown(
        f"""
        <div class="verdict verdict-{semantic}">
          <div class="verdict-row">
            <span class="verdict-label">{label}</span>
            <span class="verdict-readout">{confidence * 100:.1f}%</span>
          </div>
          <div class="verdict-sub">model confidence</div>
          {note}
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Header ------------------------------------------------------------------------
st.markdown('<div class="kicker">Electroluminescence fault inspection</div>',
            unsafe_allow_html=True)
st.markdown('<div class="brand">Reading the Grid</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="tagline">Find faults in solar cells from electroluminescence imaging, '
    'and see where the model looked.</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="rule">', unsafe_allow_html=True)

model, device = load_model()

if "example" not in st.session_state:
    st.session_state.example = None

# --- Input: upload or pick an example ----------------------------------------------
st.markdown('<div class="section-label">Upload an image</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop a PNG or JPG electroluminescence cell image",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

st.markdown('<div class="section-label">Or try an example</div>', unsafe_allow_html=True)
example_files = [EXAMPLES_DIR / f"{name}_{i}.png"
                 for name in ("healthy", "faulty") for i in (1, 2, 3)]
columns = st.columns(6)
for column, example_path in zip(columns, example_files):
    with column:
        if example_path.exists():
            st.image(str(example_path), width="stretch")
            if st.button(example_path.stem.replace("_", " ").title(),
                         key=example_path.name):
                st.session_state.example = str(example_path)

# Uploaded files take priority; otherwise fall back to the selected example.
image = None
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
elif st.session_state.example:
    image = Image.open(st.session_state.example).convert("RGB")

st.markdown('<hr class="rule">', unsafe_allow_html=True)

# --- Result ------------------------------------------------------------------------
if image is None:
    st.markdown(
        '<div class="placeholder">Upload a cell image or select an example to run the '
        'classifier.</div>',
        unsafe_allow_html=True,
    )
else:
    pred, confidence, display_rgb, overlay = analyze(image, model, device)
    render_verdict(pred, confidence)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
        st.image(display_rgb, width="stretch", clamp=True)
    with right:
        st.markdown('<div class="section-label">Model attention</div>',
                    unsafe_allow_html=True)
        st.image(overlay, width="stretch")

    st.markdown(
        '<div class="heatmap-caption">Warm areas show where the model focused — its '
        'attention, not a guaranteed defect location.</div>',
        unsafe_allow_html=True,
    )

# --- Method and limitations --------------------------------------------------------
with st.expander("How this works & limitations"):
    st.markdown(
        """
The classifier is a ResNet18 pretrained on ImageNet, transfer-learned on the ELPV
electroluminescence dataset, then fine-tuned by unfreezing its final convolutional block.
It outputs a healthy/faulty label and a softmax confidence.

The overlay is produced with Grad-CAM on the last convolutional layer. Grad-CAM is
min-max normalized per image, so every map has a full-intensity peak by construction: it
shows where the evidence for the predicted class is concentrated, not the severity or
presence of a defect, and it does not go quiet on healthy cells. On faulty cells the peak
is sometimes on or near a visible defect and sometimes not clearly aligned; uniformly dead
or disconnected cells tend to draw activation toward the edges, since there is no localized
lesion to attend to.

Read this as an inspection-assist aid that directs attention, not as a calibrated defect
localizer. Predictions below 65% confidence are flagged as borderline.

**Dataset.** ELPV — electroluminescence images of crystalline silicon solar cells
(Buerhop-Lutz et al.; Deitsch et al.). Used here for non-commercial research and
demonstration only.
        """
    )
