# Configuration settings for the Gradio face recognition application
from deepface import DeepFace
DATABASE_URL = "your_database_url_here"
EMBEDDING_MODEL = "Dlib"  # or any other model you prefer
DETECTOR_BACKEND = "dlib"  # or any other detector backend you prefer
IMAGE_SIZE = (224, 224)  # Size to which images will be resized
CONFIDENCE_THRESHOLD = DeepFace.verification.find_threshold(model_name=EMBEDDING_MODEL,distance_metric='cosine')  # Threshold for verification confidence
VECTOR_DB_PATH = "./face_vector_db"  # Path to the vector database storage