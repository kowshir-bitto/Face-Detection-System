from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity
import joblib

app = FastAPI()

base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)
db_embeddings = np.load("embeddings.npy", allow_pickle=True)
labels = joblib.load("labels.pkl")

def crop_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return image[y:y+h, x:x+w]

def extract_embedding(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = img_to_array(face_img)
    face_img = preprocess_input(np.expand_dims(face_img, axis=0))
    embedding = model.predict(face_img)
    return embedding.flatten()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image."}

    face = crop_face(img)
    if face is None:
        return {"prediction": "No face detected", "confidence": 0.0}

    emb = extract_embedding(face)
    sims = cosine_similarity([emb], db_embeddings)[0]
    max_sim_idx = np.argmax(sims)
    confidence = float(sims[max_sim_idx])

    if confidence < 0.95:
        return {"prediction": "No name found", "confidence": confidence}

    name = labels[max_sim_idx]
    return {"prediction": name, "confidence": confidence}
