# build_face_db.py
import os
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
import joblib

base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

def crop_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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

def build_database(dataset_dir='Dataset'):
    embeddings = []
    labels = []

    for person in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person)
        if not os.path.isdir(person_path):
            continue

        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            face = crop_face(img)
            if face is None:
                continue
            emb = extract_embedding(face)
            embeddings.append(emb)
            labels.append(person)

    embeddings = np.array(embeddings)
    np.save("embeddings.npy", embeddings)
    joblib.dump(labels, "labels.pkl")
    print(f"Database built with {len(labels)} faces")

build_database()
