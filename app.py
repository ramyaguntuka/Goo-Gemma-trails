import os
from pathlib import Path

import streamlit as st
from openai import OpenAI

HF_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "google/gemma-4-31B-it:novita"
DEFAULT_IMAGE = (
    "https://cdn.britannica.com/61/93061-050-99147DCE/"
    "Statue-of-Liberty-Island-New-York-Bay.jpg"
)
DEFAULT_PROMPT = "Describe this image in one sentence."


def resolve_token() -> str:
    secrets_files = (
        Path(__file__).resolve().parent / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    )
    if any(path.is_file() for path in secrets_files):
        try:
            if "HF_TOKEN" in st.secrets:
                return st.secrets["HF_TOKEN"]
        except Exception:
            pass
    return os.environ.get("HF_TOKEN", "")


st.set_page_config(page_title="Gemma image chat", page_icon="🖼️", layout="centered")
st.title("Hugging Face Gemma")
st.caption("Streams a vision reply through the Hugging Face OpenAI-compatible router.")

with st.sidebar:
    st.header("Connection")
    token_input = st.text_input(
        "HF token",
        type="password",
        value=resolve_token(),
        help="A Hugging Face token with access to the Inference Providers router. "
        "You can also set HF_TOKEN in the environment or in .streamlit/secrets.toml.",
    )
    model = st.text_input("Model", value=DEFAULT_MODEL)
    st.markdown(
        "Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)."
    )

prompt = st.text_area("Prompt", value=DEFAULT_PROMPT, height=80)
image_url = st.text_input("Image URL", value=DEFAULT_IMAGE)

if image_url.strip():
    st.image(image_url, use_container_width=True)

if st.button("Describe image", type="primary", disabled=not token_input.strip()):
    if not image_url.strip():
        st.error("Enter an image URL.")
        st.stop()

    client = OpenAI(base_url=HF_BASE_URL, api_key=token_input.strip())

    def token_stream():
        stream = client.chat.completions.create(
            model=model.strip() or DEFAULT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt.strip() or DEFAULT_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url.strip()},
                        },
                    ],
                }
            ],
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content
            if text:
                yield text

    try:
        with st.chat_message("assistant"):
            st.write_stream(token_stream)
    except Exception as exc:
        st.error(f"Request failed: {exc}")

if not token_input.strip():
    st.info("Add your Hugging Face token in the sidebar to run the request.")