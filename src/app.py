import streamlit as st
import requests
import os
import tempfile
from pathlib import Path

st.set_page_config(
    page_title="Inpaint",
    page_icon="✦",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400&family=DM+Sans:wght@300;400;500&display=swap');

* { box-sizing: border-box; }

html, body, .stApp {
    background: #f8f7f4 !important;
    color: #1a1a1a !important;
    font-family: 'DM Sans', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 4rem 2rem; max-width: 680px; }

h1 {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    font-size: 1.6rem;
    letter-spacing: -0.02em;
    color: #1a1a1a;
    margin-bottom: 0.2rem;
}
.subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #aaa;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 3rem;
}
.divider {
    border: none;
    border-top: 1px solid #e4e2de;
    margin: 2rem 0;
}
.label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #aaa;
    margin-bottom: 0.6rem;
}

div[data-testid="stFileUploader"] {
    border: 1px solid #e4e2de !important;
    border-radius: 8px !important;
    background: #fff !important;
    padding: 0.25rem !important;
}
div[data-testid="stFileUploader"]:hover { border-color: #1a1a1a !important; }
div[data-testid="stFileUploader"] label { display: none; }

.stTextInput > div > div > input {
    background: #fff !important;
    border: 1px solid #e4e2de !important;
    border-radius: 8px !important;
    color: #1a1a1a !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 0.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #1a1a1a !important;
    box-shadow: none !important;
}
.stTextInput > div > div > input::placeholder { color: #bbb !important; }
.stTextInput label { display: none; }

.stButton > button {
    background: #1a1a1a !important;
    color: #f8f7f4 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 400 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.8rem !important;
    width: 100% !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.75 !important; }
.stButton > button:disabled {
    background: #e4e2de !important;
    color: #aaa !important;
}

.stProgress > div > div > div {
    background: #1a1a1a !important;
    border-radius: 99px !important;
}
.stProgress > div > div {
    background: #e4e2de !important;
    border-radius: 99px !important;
}

.status-ok {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #4a7c4e;
    letter-spacing: 0.06em;
}
.status-err {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #b94040;
    letter-spacing: 0.06em;
}
.prompt-text {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #999;
    line-height: 1.7;
    margin-top: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Endpoints**")
    vlm_url    = st.text_input("VLM",  value="http://127.0.0.1:8001")
    sdxl_url   = st.text_input("SDXL", value="http://127.0.0.1:8002/inpaint")
    output_dir = st.text_input("Output directory", value=r"D:\MM\streamlit_outputs")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1>Inpainter</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">single image · VLM + SDXL pipeline</p>', unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<div class="label">Image</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed"
)

if uploaded:
    st.image(uploaded, use_container_width=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Instruction ───────────────────────────────────────────────────────────────
st.markdown('<div class="label">Your Instruction</div>', unsafe_allow_html=True)
instruction = st.text_input(
    "Instruction",
    placeholder="Please type your instruction here, e.g. a floating house on the sea",
    label_visibility="collapsed",
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Run ───────────────────────────────────────────────────────────────────────
run = st.button("Generate", disabled=not (uploaded and instruction.strip()))

if run:
    os.makedirs(output_dir, exist_ok=True)

    tmp_dir  = tempfile.mkdtemp(prefix="inpaint_")
    img_path = os.path.join(tmp_dir, uploaded.name)
    with open(img_path, "wb") as f:
        f.write(uploaded.getbuffer())

    out_path = os.path.join(output_dir, f"{Path(img_path).stem}_result.png")
    progress = st.progress(0)
    status   = st.empty()

    try:
        # VLM prompt
        status.markdown('<p class="status-ok">→ Generating prompt…</p>', unsafe_allow_html=True)
        progress.progress(20)

        vlm_resp    = requests.post(
            f"{vlm_url}/get_prompt",
            json={"image_path": img_path, "instruction": instruction.strip()},
        )
        prompt_data = vlm_resp.json()["data"]
        print(f"生成Prompt: {prompt_data}")
        progress.progress(50)

        # SDXL inpaint
        status.markdown('<p class="status-ok">→ Running inpainting…</p>', unsafe_allow_html=True)

        sd_resp = requests.post(
            sdxl_url,
            json={
                "image_path":      img_path,
                "prompt":          prompt_data["prompt"],
                "negative_prompt": prompt_data["negative_prompt"],
                "output_path":     out_path,
            }
        )
        progress.progress(90)

        if sd_resp.json()["status"] == "success":
            progress.progress(100)
            status.markdown('<p class="status-ok">✓ Done</p>', unsafe_allow_html=True)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="label">Result</div>', unsafe_allow_html=True)
            st.image(out_path, use_container_width=True)
            st.markdown(
                f'<p class="prompt-text">{prompt_data["prompt"]}</p>',
                unsafe_allow_html=True,
            )
        else:
            msg = sd_resp.json().get("message", "unknown error")
            progress.empty()
            status.markdown(f'<p class="status-err">✗ {msg}</p>', unsafe_allow_html=True)

    except Exception as e:
        progress.empty()
        status.markdown(f'<p class="status-err">✗ {e}</p>', unsafe_allow_html=True)