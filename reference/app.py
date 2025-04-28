import streamlit as st
from PIL import Image
import time
from style_transfer import run_style_transfer

st.set_page_config(page_title="Neural Style Transfer", layout="centered")

# ==========================================
#             App Introduction
# ==========================================

st.title("🎨 Neural Style Transfer App")
st.markdown("""
This app lets you blend the **content of one image** with the **style of another** using Neural Style Transfer (NST).

Below is an example of what NST can do:
""")

# ==========================================
#        Show Example Demonstration
# ==========================================

example_col = st.columns(3)
with example_col[0]:
    st.image("examples/content.jpg", caption="Content", use_container_width=True)
with example_col[1]:
    st.image("examples/style.jpg", caption="Style", use_container_width=True)
with example_col[2]:
    st.image("examples/output.png", caption="Stylized Result", use_container_width=True)

st.markdown("---")

# ==========================================
#             File Upload Section
# ==========================================

st.subheader("🖼️ Upload Your Own Images")

content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

# ==========================================
#             Settings Section
# ==========================================

st.subheader("⚙️ Settings")
resolution = st.radio("Choose output quality", ["Low (faster)", "High (slower)"])
resolution_key = "low" if resolution.startswith("Low") else "high"

est_time = "⏱ ~400s" if resolution_key == "low" else "⏱ ~400-600s"
st.markdown(f"Estimated processing time: {est_time}")

if content_file and style_file:
    content_img = Image.open(content_file).convert("RGB")
    style_img = Image.open(style_file).convert("RGB")

    st.subheader("📸 Preview")
    preview_col = st.columns(2)
    with preview_col[0]:
        st.image(content_img, caption="Content", use_container_width=True)
    with preview_col[1]:
        st.image(style_img, caption="Style", use_container_width=True)

    # ==========================================
    #           Run Style Transfer
    # ==========================================

    if st.button("✨ Stylize!"):
        with st.spinner("Running style transfer..."):
            start_time = time.time()
            out_lr, out_hr = run_style_transfer(content_img, style_img, resolution=resolution_key)
            elapsed = time.time() - start_time

        st.success(f"Done in {int(elapsed)} seconds!")
        st.subheader("🎨 Result")

        if resolution_key == "low":
            st.image(out_lr, caption="Stylized Image (512px)", use_container_width=True)
        else:
            st.image(out_hr, caption="Stylized Image (800px)", use_container_width=True)

else:
    st.info("👆 Upload both content and style images to begin.")

