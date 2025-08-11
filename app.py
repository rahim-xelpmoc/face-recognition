import streamlit as st
from deepface import DeepFace
from database import Database
from config import VECTOR_DB_PATH
from PIL import Image
import numpy as np
import pandas as pd

# Initialize face database
face_db = Database(db_path=VECTOR_DB_PATH)

# Helper function to extract face
def extract_face(img):
    face_dict = DeepFace.detection.extract_faces(img, detector_backend='yolov8',normalize_face=False)
    return face_dict

def register_user(name, image):
    if image is not None:
        # image = image.convert("RGB")  # Ensure RGB format
        # face_dict = extract_face(np.array(image))
        face_db.add_to_collection(image, {"name": name})
        return f"✅ User '{name}' registered successfully!"
    return "❌ Failed to register user."

def verify_user_image(image):
    if image is not None:
        # image = image.convert("RGB")  # Ensure RGB format
        # face_dict = extract_face(np.array(image))
        return face_db.verify(image)
    return "❌ Failed to verify user."

# Streamlit UI
st.set_page_config(page_title="Face Recognition App", layout="centered")
st.title("🧠 Face Recognition App")

tab1, tab2,tab3 = st.tabs(["Verify","Register","Delete"])


with tab1:
    st.subheader("Verify User")
    # verify_image = st.file_uploader("Upload or Capture Image", type=["jpg", "jpeg", "png"], key="verify")
    verify_image=st.camera_input("capture the webcam image")
    if verify_image:
        image = Image.open(verify_image).convert("RGB")
        face_dict=extract_face(np.array(image))
        st.image(face_dict[0]['face'], caption="Uploaded Image", use_container_width=True)
        if st.button("Verify"):
            status = verify_user_image(face_dict[0]['face'])
            # st.info(status)
            st.json(status,expanded=2)
        

with tab2:
    st.subheader("Register New User")
    name = st.text_input("Enter your name")
    uploaded_image = st.camera_input("Upload or Capture Image")
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        face_dict=extract_face(np.array(image))
        st.image(face_dict[0]['face'], caption="Uploaded Image", use_container_width=True)
        if st.button("Register"):
            status = register_user(name, face_dict[0]['face'])
            st.success(status)
            


with tab3:
    st.subheader("Delete User")
    record_id=st.text_input("enter the record id")
    if st.button(label="Delete"):
        status=face_db.delete_record(id=record_id)
        st.info(status)
    

