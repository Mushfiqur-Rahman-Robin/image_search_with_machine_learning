import os
import sqlite3
import uuid

from classify import class_names as classification_class_names
from classify import classify_image
from detect import detect_objects

DB_PATH = "detections.db"
SAVE_ROOT = "output_images"

# Setup DB
def init_db():
    os.makedirs(SAVE_ROOT, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            img_id TEXT,
            class_name TEXT,
            hierarchy TEXT
        )
    ''')
    conn.commit()
    conn.close()

def get_class_names():
    return classification_class_names

def save_to_db(img_id, detected_classes, bathroom_type):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Delete previous entries for this img_id to avoid duplicates
    cursor.execute('DELETE FROM detections WHERE img_id = ?', (img_id,))
    for class_name in detected_classes:
        cursor.execute('''
            INSERT INTO detections (img_id, class_name, hierarchy)
            VALUES (?, ?, ?)
        ''', (img_id, class_name, bathroom_type))
    conn.commit()
    conn.close()

def save_image_to_structure(img_id, bathroom_type, detected_classes, image_path):
    # Remove any previous instances of this img_id in the directory structure
    for root, dirs, files in os.walk(SAVE_ROOT):
        for file in files:
            if file.startswith(img_id):
                os.remove(os.path.join(root, file))

    # Save to new directory structure
    for class_name in detected_classes:
        dir_path = os.path.join(SAVE_ROOT, bathroom_type, class_name)
        os.makedirs(dir_path, exist_ok=True)
        save_path = os.path.join(dir_path, f"{img_id}.jpg")
        image_path.save(save_path)

def run_full_pipeline(pil_image):
    img_id = str(uuid.uuid4())
    temp_path = f"{img_id}.jpg"
    pil_image.save(temp_path)

    # Classification
    bathroom_type, confidence = classify_image(pil_image)

    # Detection
    detected_classes = detect_objects(temp_path)

    # Do NOT save to DB or directory structure here
    # Return the temp_path for later use
    return {
        "img_id": img_id,
        "bathroom_type": bathroom_type,
        "confidence": confidence,
        "detected_objects": detected_classes,
        "temp_path": temp_path
    }

def search_images(keyword):
    import unicodedata

    def normalize(text):
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()

    norm_keyword = normalize(keyword)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT img_id, class_name, hierarchy FROM detections")
    rows = cursor.fetchall()
    conn.close()

    matching_ids = set()
    paths = []

    for img_id, class_name, hierarchy in rows:
        if norm_keyword in normalize(class_name) or norm_keyword in normalize(hierarchy):
            matching_ids.add((img_id, hierarchy, class_name))

    for img_id, hierarchy, class_name in matching_ids:
        path = os.path.join(SAVE_ROOT, hierarchy, class_name, f"{img_id}.jpg")
        if os.path.exists(path):
            paths.append(path)
    return paths
