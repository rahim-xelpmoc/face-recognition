from typing import Any, Dict, List
import chromadb
from deepface import DeepFace
from uuid import uuid4
from chromadb.utils.distance_functions import cosine
from chromadb.api.types import EmbeddingFunction,Documents,Embeddings
from config import EMBEDDING_MODEL,DETECTOR_BACKEND

class DeepFaceEmebeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self,model_name:str="VGG-Face",detector_backend:str="opencv",enforce_detection=False):
        """model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n', 'yolov11s', 'yolov11m',
            'centerface' or 'skip' (default is opencv).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Set to False to avoid the exception for low-resolution images (default is True)."""
        self.model_name=model_name
        self.detector_backend=detector_backend
        self.enforce_detection=enforce_detection
    
    def get_embedding_with_metadata(self,input:Documents)->List[Dict[str,Any]]:
        """get un normalized embedding with metadata"""
        embs=DeepFace.represent(img_path=input,
                                model_name=self.model_name,
                                detector_backend=self.detector_backend,
                                enforce_detection=self.enforce_detection)
        for emb in embs:
            emb['embedding']=DeepFace.verification.l2_normalize(emb['embedding'])
            
        return embs
    
    def __call__(self,input:Documents)->Embeddings:
        embs=self.get_embedding_with_metadata(input=input)
        embeddings=[]
        for emb in embs:
            embeddings.append(emb['embedding'])
        return embeddings
    
    @staticmethod
    def name()->str:
        return "deepface"
    
    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction[Documents]":
        model_name=config.get("model_name","VGG-Face")
        detector_backend=config.get("detector_backend","opencv")
        enforce_detection=config.get("enforce_detection",False)
        return DeepFaceEmebeddingFunction(model_name=model_name,detector_backend=detector_backend,enforce_detection=enforce_detection)

    def get_config(self) -> Dict[str, Any]:
        config={"model_name":self.model_name,
                "detector_backend":self.detector_backend,
                "enforce_detection":self.enforce_detection}
        return config


class Database:
    def __init__(self, db_path: str):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_func=DeepFaceEmebeddingFunction(model_name=EMBEDDING_MODEL, detector_backend=DETECTOR_BACKEND)
        self.collection = self.client.get_or_create_collection(
            name="face-database",
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": "cosine"}
        )

    def add_to_collection(self,img_path:str,metadata):
        try:
            embs=self.embedding_func.get_embedding_with_metadata(input=[img_path])
            self.collection.add(
                ids=str(uuid4()),
                embeddings=embs[0]['embedding'],
                metadatas={**metadata,
                        "facial_area":str(embs[0]['facial_area']),
                        "face_confidence":embs[0]['face_confidence']}
            )
        except Exception as e:
            print(e)

    def verify(self,img_path:str)->Dict[str,Any]:
        input_emb=self.embedding_func([img_path])
        threshold=DeepFace.verification.find_threshold(model_name=EMBEDDING_MODEL,distance_metric='cosine')
        result=self.collection.query(query_embeddings=input_emb,include=['embeddings',"metadatas"],n_results=1)
        metadata=result['metadatas'][0][0]
        distance=cosine(result['embeddings'][0],input_emb[0])
        result={'verified': distance<=threshold,
        'distance': distance,
        'threshold': threshold,
        'confidence': DeepFace.verification.find_confidence(distance,EMBEDDING_MODEL,"cosine",distance<=threshold),
        'model': EMBEDDING_MODEL,
        'detector_backend': DETECTOR_BACKEND,
        'similarity_metric': 'cosine',
        'facial_areas': {**metadata},
        "id":result['ids'][0][0]}
        return result
    
    def delete_record(self,id):
        self.collection.delete(id)
        return f"{id} deleted successfully"
