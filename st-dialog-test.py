import streamlit as st

st.set_page_config(layout="wide")

IMAGES = [
    "https://picsum.photos/id/1015/800/600",
    "https://picsum.photos/id/1016/800/600",
    "https://picsum.photos/id/1024/800/600",
    "https://picsum.photos/id/1025/800/600",
]

NUM_COLS = 4
cols = st.columns(NUM_COLS)

if "selected_image" not in st.session_state:
    st.session_state["selected_image"] = None

# Define modal
@st.dialog("Image preview", width="large")
def preview_dialog(img_url: str):
    st.image(img_url, use_container_width=True)
    if st.button("Close preview"):
        st.session_state["selected_image"] = None
        st.rerun()

# Show thumbnails
for i, img in enumerate(IMAGES):
    with cols[i % NUM_COLS]:
        st.image(img, use_container_width=True)
        if st.button("View", key=f"btn_{i}"):
            st.session_state["selected_image"] = img
            st.rerun()

# Trigger dialog if one is selected
if st.session_state["selected_image"]:
    preview_dialog(st.session_state["selected_image"])
