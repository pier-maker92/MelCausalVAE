import json
from pathlib import Path

import streamlit as st


AUDIO_DIR = Path(__file__).resolve().parent / "audio"
TTS_MANIFEST = AUDIO_DIR / "tts_manifest.json"
SPEAKERS = [
    ("male", "Male"),
    ("female", "Female"),
    ("pier", "Pier"),
    ("fra", "Fra"),
    ("podcast", "Podcast"),
]
SWAP_TARGETS = [
    ("male", "Swap male"),
    ("female", "Swap female"),
    ("pier", "Swap pier"),
    ("fra", "Swap fra"),
    ("podcast", "Swap podcast"),
]


st.set_page_config(page_title="Dicodec Voice Demo", layout="wide")

st.title("Dicodec Voice Demo")
st.caption("Checkpoint: checkpoints/dropout/18")


def play(filename: str):
    path = AUDIO_DIR / filename
    if path.exists():
        st.audio(str(path))
    else:
        st.caption("missing")


def load_tts_manifest():
    if not TTS_MANIFEST.exists():
        return {}
    with TTS_MANIFEST.open() as f:
        return json.load(f)


def render_tts_section(title: str, rows):
    if not rows:
        return
    st.markdown(f"## {title}")
    for index, row in enumerate(rows, start=1):
        st.markdown(f"**{index}. {row['text']}**")
        cols = st.columns([1, 1])
        with cols[0]:
            st.caption("Male")
            play(row["audio"]["male"])
        with cols[1]:
            st.caption("Female")
            play(row["audio"]["female"])
    st.divider()


headers = [
    "Original",
    "Reconstructed",
    "First 4 dim",
    "4:64",
    "K-means only",
    "K-means + tail",
    *[label for _, label in SWAP_TARGETS],
]

header_cols = st.columns(len(headers))
for col, header in zip(header_cols, headers):
    col.markdown(f"**{header}**")

for speaker, label in SPEAKERS:
    st.markdown(f"### {label}")
    cols = st.columns(len(headers))

    with cols[0]:
        play(f"{speaker}.wav")
    with cols[1]:
        play(f"{speaker}_reconstruction.wav")
    with cols[2]:
        play(f"{speaker}_first4.wav")
    with cols[3]:
        play(f"{speaker}_dims4_64.wav")
    with cols[4]:
        play(f"{speaker}_kmeans.wav")
    with cols[5]:
        play(f"{speaker}_kmeans_tail.wav")

    for col, (target, _) in zip(cols[6:], SWAP_TARGETS):
        with col:
            if target == speaker:
                st.caption("")
            else:
                play(f"{speaker}_swap_{target}.wav")

    st.divider()

tts_manifest = load_tts_manifest()
render_tts_section("TTS", tts_manifest.get("tts"))
render_tts_section("TTS long sentence", tts_manifest.get("tts_long"))
render_tts_section("TTS multilingual", tts_manifest.get("tts_multilingual"))
