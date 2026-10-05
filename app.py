import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Object Detection", page_icon="🔎", layout="centered")

st.title("🔎 Object Detection")
st.write("Upload an image to detect common objects using a pre-trained YOLO model.")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

confidence = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if st.button("🔎 Detect Objects", type="primary"):
        results = model.predict(image, conf=confidence, verbose=False)
        result = results[0]

        annotated = result.plot()
        annotated = Image.fromarray(annotated[..., ::-1])

        st.subheader("Detected Objects")
        st.image(annotated, use_container_width=True)

        names = result.names
        detected = []

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            detected.append((names[cls_id], conf))

        if detected:
            st.success(f"{len(detected)} object(s) detected.")
            for label, conf in detected:
                st.write(f"**{label}** — Confidence: **{conf:.2%}**")
        else:
            st.info("No objects were detected above the selected confidence threshold.")
else:
    st.info("Upload an image to begin.")
