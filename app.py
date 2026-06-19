import pandas as pd
from PIL import Image
import numpy as np
import json
from pyngrok import ngrok
import tensorflow as tf
import shutil
import cv2
import os
import tempfile
from datetime import datetime
from dotenv import load_dotenv
import base64
from flask import Flask, request, render_template, make_response, redirect, url_for, send_from_directory, session, jsonify
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, BatchNormalization
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# CONFIGURATION
load_dotenv()
base_path = './Face_Recog_App'
user_img_dir = f'{base_path}/static/uploads/UserImages'
os.makedirs(user_img_dir, exist_ok=True)

csv_file = f'{user_img_dir}/users.csv'
vote_file = f'{user_img_dir}/votes.csv' 
train_log_file = f'{base_path}/model/train.log'

# รหัสผ่านสำหรับเข้า Admin Zone
ADMIN_PASSWORD = "1234"

# โหลด Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
yolo_model = None
YOLO_FACE_MODEL_PATH = os.getenv("YOLO_FACE_MODEL_PATH", f"{base_path}/model/yolov8n-face.pt")

if YOLO is not None and os.path.exists(YOLO_FACE_MODEL_PATH):
    try:
        yolo_model = YOLO(YOLO_FACE_MODEL_PATH)
        print(f"✅ YOLO face model loaded: {YOLO_FACE_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ YOLO model load failed, fallback to Haar: {e}")
elif YOLO is None:
    print("ℹ️ ultralytics not installed, using Haar Cascade.")
else:
    print(f"ℹ️ YOLO model not found at {YOLO_FACE_MODEL_PATH}, using Haar Cascade.")

if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=['name', 'surname', 'studentid', 'folder', 'has_voted'])
    df.to_csv(csv_file, index=False)

if not os.path.exists(vote_file):
    df_votes = pd.DataFrame(columns=['candidate_id', 'candidate_name', 'vote_count'])
    df_votes.to_csv(vote_file, index=False)

app = Flask(__name__)
app.secret_key = "2z7dny9VHttyHgA2yffzSjJVTmc_89ZSUD7CNsQGJDTi6tvER"

# HELPER FUNCTIONS

def write_train_log(message):
    os.makedirs(os.path.dirname(train_log_file), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(train_log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

@app.before_request
def log_request_path():
    # ช่วย debug ว่า request เข้า route ไหนจริง
    print(f"🌐 {request.method} {request.path}", flush=True)

def cv2_imread_utf8(path):
    try:
        stream = open(path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR) 
        stream.close()
        return img
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def cv2_imwrite_utf8(path, img):
    try:
        is_success, im_buf_arr = cv2.imencode(".jpg", img)
        if is_success:
            with open(path, "wb") as f:
                im_buf_arr.tofile(f)
            return True
        return False
    except Exception as e:
        print(f"Save Error: {e}")
        return False

def normalize_all_existing_images():
    """Normalize รูปภาพเก่าทั้งหมดที่มีอยู่แล้ว"""
    print("\n🔄 เริ่ม normalize รูปภาพเก่าทั้งหมด...")
    total_normalized = 0
    total_errors = 0
    
    if not os.path.exists(user_img_dir):
        print(f"❌ ไม่พบโฟลเดอร์: {user_img_dir}")
        return {'success': False, 'message': 'ไม่พบโฟลเดอร์รูปภาพ'}
    
    # วนลูปทุกโฟลเดอร์ผู้ใช้
    for folder_name in os.listdir(user_img_dir):
        folder_path = os.path.join(user_img_dir, folder_name)
        
        # ข้ามไฟล์ที่ไม่ใช่โฟลเดอร์
        if not os.path.isdir(folder_path):
            continue
        
        # ข้ามไฟล์ CSV
        if folder_name.endswith('.csv'):
            continue
        
        print(f"  📁 กำลัง normalize: {folder_name}")
        
        # วนลูปทุกรูปภาพในโฟลเดอร์
        for img_name in os.listdir(folder_path):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(folder_path, img_name)
            
            try:
                # อ่านรูปภาพ
                img = cv2_imread_utf8(img_path)
                if img is None:
                    print(f"    ⚠️ ไม่สามารถอ่านไฟล์: {img_name}")
                    total_errors += 1
                    continue
                
                # Normalize รูปภาพ
                normalized_img = normalize_image_for_camera_variation(img)
                
                # บันทึกทับไฟล์เดิม
                if cv2_imwrite_utf8(img_path, normalized_img):
                    total_normalized += 1
                    print(f"    ✅ Normalize: {img_name}")
                else:
                    print(f"    ❌ ไม่สามารถบันทึก: {img_name}")
                    total_errors += 1
                    
            except Exception as e:
                print(f"    ❌ Error processing {img_name}: {e}")
                total_errors += 1
    
    result = {
        'success': True,
        'total_normalized': total_normalized,
        'total_errors': total_errors,
        'message': f'Normalize สำเร็จ {total_normalized} รูป, เกิดข้อผิดพลาด {total_errors} รูป'
    }
    
    print(f"\n✅ เสร็จสิ้น: {result['message']}")
    return result

def normalize_image_for_camera_variation(img):
    """Normalize ภาพเพื่อลดผลกระทบจากกล้องต่างกัน"""
    # แปลงเป็น LAB color space เพื่อแยก brightness ออกจากสี
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # ใช้ CLAHE (Contrast Limited Adaptive Histogram Equalization) 
    # เพื่อ normalize brightness/contrast ให้เหมือนกัน
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # รวมกลับเป็น LAB แล้วแปลงกลับเป็น BGR
    lab = cv2.merge([l, a, b])
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # ปรับ gamma correction เพิ่มเติมเพื่อ normalize brightness
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    normalized = cv2.LUT(normalized, table)
    
    return normalized

def detect_faces(image_array):
    """
    ตรวจจับใบหน้าโดยใช้ YOLO ก่อน (ถ้ามี), แล้ว fallback ไป Haar Cascade
    คืนค่าเป็น list ของ (x, y, w, h)
    """
    h_img, w_img = image_array.shape[:2]

    if yolo_model is not None:
        try:
            results = yolo_model(image_array, verbose=False)
            boxes = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    x = max(0, x1)
                    y = max(0, y1)
                    w = min(w_img - x, max(0, x2 - x1))
                    h = min(h_img - y, max(0, y2 - y1))
                    if w > 0 and h > 0:
                        boxes.append((x, y, w, h))
            if boxes:
                return boxes
        except Exception as e:
            print(f"⚠️ YOLO detect failed, fallback to Haar: {e}")

    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.1, 5)

def detect_and_crop_face(image_array):
    faces = detect_faces(image_array)
    if len(faces) == 0: return None
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    face_roi = image_array[y:y+h, x:x+w]
    face_roi = cv2.resize(face_roi, (100, 100))
    
    # Normalize เพื่อลดผลกระทบจากกล้องต่างกัน
    face_roi = normalize_image_for_camera_variation(face_roi)
    
    return face_roi

def load_data_rgb(data_path, img_size=(100,100)):
    images = []
    labels = []
    label_map = {}
    label_id = 0
    print(f"กำลังสแกนโฟลเดอร์: {data_path}")
    for folder in sorted(os.listdir(data_path)):
        folder_path = os.path.join(data_path, folder)
        if not os.path.isdir(folder_path): continue
        if folder not in label_map:
            label_map[folder] = label_id
            label_id += 1     
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2_imread_utf8(img_path)
            if img is not None:
                # Normalize ภาพเพื่อลดผลกระทบจากกล้องต่างกัน
                img = normalize_image_for_camera_variation(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label_map[folder])

    if len(images) > 0:
        images = np.array(images, dtype=np.float32) / 255.0
        labels = np.array(labels)
    else:
        images = np.empty((0, 100, 100, 3))
        labels = np.array([])
    return images, labels, label_map

def process_upload_to_cv2(file_storage):
    in_memory_file = file_storage.read()
    nparr = np.frombuffer(in_memory_file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

class CompatibleInputLayer(tf.keras.layers.InputLayer):
    """รองรับโมเดลเก่าที่ serialize ด้วย key ชื่อ batch_shape"""
    def __init__(self, *args, **kwargs):
        if "batch_shape" in kwargs and "batch_input_shape" not in kwargs:
            kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
        super().__init__(*args, **kwargs)

def load_face_model(model_path):
    """
    โหลดโมเดลแบบทนทานกับไฟล์ .keras เก่า/ใหม่
    """
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as first_error:
        try:
            return tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects={"InputLayer": CompatibleInputLayer},
                safe_mode=False
            )
        except Exception as second_error:
            # Fallback สุดท้าย: patch config.json ในไฟล์ .keras ชั่วคราว
            try:
                if model_path.endswith(".keras"):
                    return load_face_model_with_patched_keras_config(model_path)
            except Exception as third_error:
                raise RuntimeError(
                    f"โหลดโมเดลไม่สำเร็จ: {first_error} | fallback error: {second_error} | patch error: {third_error}"
                ) from third_error

            raise RuntimeError(
                f"โหลดโมเดลไม่สำเร็จ: {first_error} | fallback error: {second_error}"
            ) from second_error

def _replace_batch_shape_key(obj):
    if isinstance(obj, dict):
        patched = {}
        for key, value in obj.items():
            new_key = "batch_input_shape" if key == "batch_shape" else key
            patched[new_key] = _replace_batch_shape_key(value)
        return patched
    if isinstance(obj, list):
        return [_replace_batch_shape_key(item) for item in obj]
    return obj

def load_face_model_with_patched_keras_config(model_path):
    with tempfile.TemporaryDirectory() as tmp_dir:
        extract_dir = os.path.join(tmp_dir, "model_extract")
        repack_path = os.path.join(tmp_dir, "patched_model.keras")
        os.makedirs(extract_dir, exist_ok=True)

        import zipfile
        with zipfile.ZipFile(model_path, "r") as zf:
            zf.extractall(extract_dir)

        config_path = os.path.join(extract_dir, "config.json")
        if not os.path.exists(config_path):
            raise RuntimeError("ไม่พบ config.json ในไฟล์ .keras")

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        patched_config = _replace_batch_shape_key(config_data)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(patched_config, f)

        with zipfile.ZipFile(repack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(extract_dir):
                for file_name in files:
                    full_path = os.path.join(root, file_name)
                    arc_name = os.path.relpath(full_path, extract_dir).replace("\\", "/")
                    zf.write(full_path, arc_name)

        return tf.keras.models.load_model(repack_path, compile=False, safe_mode=False)

class ConsoleTrainingLogger(tf.keras.callbacks.Callback):
    """แสดงสถานะการเทรนใน console แบบอ่านง่าย"""
    def on_train_begin(self, logs=None):
        msg = "🚀 เริ่มเทรนโมเดล..."
        print(msg, flush=True)
        write_train_log(msg)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss", 0.0)
        acc = logs.get("accuracy", 0.0)
        val_loss = logs.get("val_loss")
        val_acc = logs.get("val_accuracy")
        if val_loss is not None and val_acc is not None:
            msg = (
                f"  Epoch {epoch + 1:02d} | loss={loss:.4f} acc={acc:.4f} "
                f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            print(msg, flush=True)
            write_train_log(msg)
        else:
            msg = f"  Epoch {epoch + 1:02d} | loss={loss:.4f} acc={acc:.4f}"
            print(msg, flush=True)
            write_train_log(msg)

    def on_train_end(self, logs=None):
        msg = "✅ เทรนโมเดลเสร็จแล้ว"
        print(msg, flush=True)
        write_train_log(msg)

def retrain_face_model_from_existing_images():
    """
    เทรนโมเดลใหม่จากรูปผู้ใช้ที่มีอยู่ เพื่อซ่อมกรณีไฟล์โมเดลเก่าโหลดไม่ได้
    """
    all_images, all_labels, label_map = load_data_rgb(user_img_dir)
    if len(all_images) == 0:
        return {"success": False, "message": "ไม่พบรูปภาพสำหรับเทรนโมเดลใหม่"}

    num_classes = len(label_map)
    if num_classes > 1:
        trainX, testX, trainY, testY = train_test_split(
            all_images, all_labels, test_size=0.3, stratify=all_labels, random_state=42
        )
    else:
        trainX, trainY = all_images, all_labels
        testX, testY = all_images, all_labels

    model = Sequential([
        tf.keras.Input(shape=(100, 100, 3)),
        Conv2D(32, (3,3), padding='same'), BatchNormalization(), Activation('relu'), MaxPooling2D(2,2), Dropout(0.2),
        Conv2D(64, (3,3), padding='same'), BatchNormalization(), Activation('relu'), MaxPooling2D(2,2), Dropout(0.2),
        Conv2D(128, (3,3), activation='relu'), MaxPooling2D(2,2), Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'), Dropout(0.5),
        Dense(num_classes if num_classes > 1 else 2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        trainX,
        trainY,
        epochs=30,
        batch_size=8,
        verbose=2,
        callbacks=[ConsoleTrainingLogger()]
    )

    model_dir = f'{base_path}/model'
    os.makedirs(model_dir, exist_ok=True)
    model.save(f'{model_dir}/face_cnn_model.keras')

    with open(f'{model_dir}/label_map.json', 'w') as f:
        json.dump(label_map, f)

    loss, acc = model.evaluate(testX, testY, verbose=0)
    result_msg = f"✅ Re-train complete. Accuracy: {acc:.4f} ({acc * 100:.2f}%)"
    print(result_msg)
    write_train_log(result_msg)
    return {"success": True, "message": "retrain success", "accuracy": float(acc)}

# FLASK ROUTES

@app.route('/register', methods=['GET'])
def register_get():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register_post():
    try:
        print("\n=== เริ่มกระบวนการลงทะเบียน ===")
        name = request.form['name']
        surname = request.form['surname']
        studentid = request.form['studentid']

        df = pd.read_csv(csv_file, dtype={'studentid': str})
        if studentid in df['studentid'].values:
            return jsonify({'status': 'error', 'message': "รหัสนักเรียนนี้นี้ลงทะเบียนไปแล้ว"}), 400

        user_folder = f"{name}_{surname}".replace(" ", "_")
        save_path = os.path.join(user_img_dir, user_folder)
        os.makedirs(save_path, exist_ok=True)

        valid_images_count = 0
        example_image_b64 = None 

        for i in range(8):
            file = request.files.get(f'image_{i}')
            if file:
                img = process_upload_to_cv2(file)
                if img is not None:
                    # 1. พยายามตัดเฉพาะหน้าก่อน
                    face_roi = detect_and_crop_face(img)
                    
                    # 2. ถ้าหาหน้าไม่เจอ ให้ใช้รูปเต็มแทน (Fallback)
                    if face_roi is None:
                        # print(f" รูปที่ {i+1} ไม่เจอกรอบหน้า -> บันทึกรูปเต็มแทน")
                        try:
                            face_roi = cv2.resize(img, (100, 100))
                        except:
                            face_roi = None

                    # 3. บันทึกรูป
                    if face_roi is not None:
                        save_success = cv2_imwrite_utf8(os.path.join(save_path, f'img_{i+1}.jpg'), face_roi)
                        if save_success:
                            valid_images_count += 1
                        
                        # [จุดที่แก้] สร้างรูปตัวอย่าง + วาดกรอบสี่เหลี่ยมกลับมา
                        if example_image_b64 is None:
                            debug_img = img.copy()
                            # Detect อีกครั้งเพื่อเอาพิกัดมาวาดรูป
                            faces = detect_faces(img)
                            
                            if len(faces) > 0:
                                # ถ้าเจอหน้า วาดกรอบสีเขียว
                                (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                            
                            # แปลงเป็น Base64 ส่งกลับไปโชว์
                            _, buffer = cv2.imencode('.jpg', debug_img)
                            example_image_b64 = base64.b64encode(buffer).decode('utf-8')

        if valid_images_count == 0:
            shutil.rmtree(save_path, ignore_errors=True)
            return jsonify({'status': 'error', 'message': "ไม่สามารถประมวลผลรูปภาพได้เลย กรุณาถ่ายใหม่"}), 400
        
        new_row = pd.DataFrame([[name, surname, studentid, save_path, 0]], 
                            columns=['name', 'surname', 'studentid', 'folder', 'has_voted'])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(csv_file, index=False)
        print(f"✅ บันทึกข้อมูล {name} สำเร็จ ({valid_images_count} รูป)")

        print("⏳ กำลังเตรียมข้อมูลสำหรับเทรนโมเดล...")
        # [STEP 1: โหลดข้อมูล]
        # เรียกใช้ฟังก์ชัน load_data_rgb เพื่อสแกนโฟลเดอร์รูปภาพผู้ใช้ทั้งหมด
        # ผลลัพธ์: all_images (รูปภาพ), all_labels (ID ผู้ใช้), label_map (ชื่อผู้ใช้ -> ID ตัวเลข)
        all_images, all_labels, label_map = load_data_rgb(user_img_dir)
        
        training_accuracy = None
        # ตรวจสอบว่ามีรูปภาพในระบบหรือไม่ก่อนดำเนินการต่อ
        if len(all_images) > 0:
            # นับจำนวนคลาส (ID ผู้ใช้ที่แตกต่างกัน)
            num_classes = len(label_map) 
            
            # [STEP 2: แบ่งชุดข้อมูล]
            # ถ้ามีผู้ใช้มากกว่า 1 คน จะทำการแบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
            if num_classes > 1:
                # แบ่งข้อมูลเป็น 70% สำหรับฝึก (trainX/trainY) และ 30% สำหรับทดสอบ (testX/testY)
                # stratify=all_labels ช่วยให้สัดส่วนผู้ใช้ในชุดฝึกและชุดทดสอบเท่ากัน
                trainX, testX, trainY, testY = train_test_split(all_images, all_labels, test_size=0.3, stratify=all_labels, random_state=42)
            else:
                # กรณีมีผู้ใช้แค่คนเดียว ให้ใช้ข้อมูลทั้งหมดเป็นทั้งชุดฝึกและชุดทดสอบ
                trainX, trainY = all_images, all_labels
                testX, testY = all_images, all_labels 

            # [STEP 3: สร้างสถาปัตยกรรมโมเดล CNN]
            model = Sequential([
                # กำหนดขนาด Input: รูปภาพ 100x100 พิกเซล, 3 สี (RGB)
                tf.keras.Input(shape=(100, 100, 3)), 
                
                # ชั้น Conv2D แรก (32 ฟิลเตอร์) เพื่อสกัดคุณลักษณะ + ลดขนาดภาพ (MaxPooling) + ป้องกัน Overfitting (Dropout)
                Conv2D(32, (3,3), padding='same'), BatchNormalization(), Activation('relu'), MaxPooling2D(2,2), Dropout(0.2), 
                
                # ชั้น Conv2D ที่สอง (64 ฟิลเตอร์) เพื่อสกัดคุณลักษณะที่ซับซ้อนขึ้น
                Conv2D(64, (3,3), padding='same'), BatchNormalization(), Activation('relu'), MaxPooling2D(2,2), Dropout(0.2), 

                # ชั้น Conv2D ที่สาม (ที่เพิ่มมา): 128 ฟิลเตอร์ + Max Pooling + Dropout 20%
                # ช่วยให้โมเดลเรียนรู้คุณลักษณะที่ซับซ้อนและละเอียดมากขึ้น
                Conv2D(128, (3,3), activation='relu'), MaxPooling2D(2,2), Dropout(0.2),
                
                # แปลงข้อมูลจาก 2D เป็น 1D (เวกเตอร์) เพื่อป้อนเข้าชั้น Dense
                Flatten(), 
                
                # ชั้นเชื่อมต่อเต็มรูปแบบ (Dense Layer) 128 โหนด พร้อม Dropout 50%
                Dense(128, activation='relu'), Dropout(0.5), 
                
                # ชั้น Output: จำนวนโหนดเท่ากับจำนวนผู้ใช้ (num_classes) ใช้ Softmax เพื่อระบุความน่าจะเป็น
                Dense(num_classes if num_classes > 1 else 2, activation='softmax') 
            ])
            
            # [STEP 4: คอมไพล์และฝึกโมเดล]
            # กำหนดวิธีการฝึก: Optimizer=adam, Loss Function สำหรับป้ายกำกับที่เป็นตัวเลข, และวัดผลด้วย Accuracy
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            model.fit(
                trainX,
                trainY,
                epochs=30,
                batch_size=8,
                verbose=2,
                callbacks=[ConsoleTrainingLogger()]
            )
            
            # [STEP 5: บันทึกโมเดล]
            model_dir = f'{base_path}/model'
            os.makedirs(model_dir, exist_ok=True)
            model.save(f'{model_dir}/face_cnn_model.keras') # บันทึกโมเดลที่ฝึกเสร็จแล้ว
            
            # บันทึกไฟล์ JSON สำหรับแปลง ID ตัวเลขที่ทำนายกลับเป็นชื่อผู้ใช้
            with open(f'{model_dir}/label_map.json', 'w') as f:
                json.dump(label_map, f)
                
            print("บันทึกโมเดลเรียบร้อยแล้ว")
            loss, acc = model.evaluate(testX, testY, verbose=0)
            training_accuracy = float(acc)
            print(f"Training Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
        else:
            print("ไม่พบรูปภาพ")
        
        target_url = url_for('index', msg='registered')
        
        # ส่ง example_image กลับไปด้วย
        return jsonify({
            'status': 'success', 
            'redirect_url': target_url, 
            'example_image': example_image_b64,
            'training_accuracy': training_accuracy
        })
    
    except Exception as e:
        print(f" Error: {e}")
        return jsonify({'status': 'error', 'message': f"เกิดข้อผิดพลาด: {str(e)}"}), 500

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            model_path = f'{base_path}/model/face_cnn_model.keras'
            if not os.path.exists(model_path):
                return jsonify({'status': 'error', 'message': "ระบบยังไม่พร้อม (ยังไม่มีโมเดล)"})
                
            try:
                model = load_face_model(model_path)
            except Exception as model_error:
                print(f"โหลดโมเดลเดิมไม่สำเร็จ, กำลัง re-train: {model_error}")
                retrain_result = retrain_face_model_from_existing_images()
                if not retrain_result.get("success"):
                    return jsonify({
                        'status': 'error',
                        'message': f"โหลดโมเดลไม่สำเร็จ และเทรนใหม่ไม่ผ่าน: {retrain_result.get('message')}"
                    })
                model = load_face_model(model_path)
            with open(f'{base_path}/model/label_map.json', 'r') as f:
                label_map = json.load(f)
            inv_label_map = {v: k for k, v in label_map.items()}
            
            studentid = request.form['studentid']
            file = request.files['image']
            
            users_df = pd.read_csv(csv_file, dtype={'studentid': str})
            user_row = users_df[users_df['studentid'] == studentid]

            if user_row.empty:
                return jsonify({'status': 'error', 'message': f"ไม่พบรหัสนักเรียน {studentid} ในระบบ"})

            expected_folder = os.path.basename(user_row.iloc[0]['folder'])
            
            img_cv = process_upload_to_cv2(file)
            cropped_face = detect_and_crop_face(img_cv)

            if cropped_face is None:
                return jsonify({'status': 'error', 'message': "ไม่พบใบหน้าในกล้อง กรุณาถ่ายใหม่ให้ชัดเจน"})

            img_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)

            pred = model.predict(img_batch)
            pred_idx = np.argmax(pred)
            
            if pred_idx in inv_label_map:
                pred_label = inv_label_map[pred_idx]
            else:
                pred_label = "Unknown"

            if pred_label == expected_folder:
                session['studentid'] = str(user_row.iloc[0]['studentid'])
                session['name'] = user_row.iloc[0]['name']
                session['surname'] = user_row.iloc[0]['surname']
                
                has_voted = int(user_row.iloc[0]['has_voted'])
                target_url = url_for('results') if has_voted == 1 else url_for('vote_page')
                
                return jsonify({'status': 'success', 'redirect_url': target_url})
            else:
                return jsonify({'status': 'error', 'message': "ใบหน้าไม่ตรงกับข้อมูลในระบบ (Face Mismatch)"})

        except Exception as e:
            return jsonify({'status': 'error', 'message': f"Error: {str(e)}"})

    return render_template('index.html')

@app.route('/vote')
def vote_page():
    if 'studentid' not in session: return redirect(url_for('index'))
    vote_df = pd.read_csv(vote_file)
    candidates = vote_df.to_dict('records')
    return render_template('vote.html', name=session.get('name'), surname=session.get('surname'), studentid=session.get('studentid'),candidates=candidates)

@app.route('/submit_vote', methods=['POST'])
def submit_vote():
    if 'studentid' not in session: return redirect(url_for('index'))
    studentid = session['studentid']
    candidate_id = int(request.form['candidate_id'])
    users_df = pd.read_csv(csv_file, dtype={'studentid': str})
    idx = users_df.index[users_df['studentid'] == studentid].tolist()
    if not idx: return "User error", 400
    if users_df.at[idx[0], 'has_voted'] == 1: return "ใช้สิทธิ์ไปแล้ว!"
    users_df.at[idx[0], 'has_voted'] = 1
    users_df.to_csv(csv_file, index=False)
    vote_df = pd.read_csv(vote_file)
    c_idx = vote_df.index[vote_df['candidate_id'] == candidate_id].tolist()
    if c_idx:
        vote_df.at[c_idx[0], 'vote_count'] += 1
        vote_df.to_csv(vote_file, index=False)
    return redirect(url_for('results'))

@app.route('/results')
def results():
    vote_df = pd.read_csv(vote_file)
    vote_df = vote_df.sort_values(by='vote_count', ascending=False)
    rows = vote_df.to_dict('records')
    return render_template('result.html', rows=rows)

@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        pwd = request.form['password']
        if pwd == ADMIN_PASSWORD:
            session['is_admin'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            return "<h1>รหัสผ่านผิด! <a href='/admin'>ลองใหม่</a></h1>"
    return render_template('admin_login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('is_admin'): return redirect(url_for('admin_login'))
    df = pd.read_csv(vote_file)
    candidates = df.to_dict('records')
    train_status = request.args.get('train_status', '')
    train_message = request.args.get('train_message', '')
    return render_template(
        'admin_dashboard.html',
        candidates=candidates,
        train_status=train_status,
        train_message=train_message
    )

@app.route('/admin/add_candidate', methods=['POST'])
def add_candidate():
    if not session.get('is_admin'): return redirect(url_for('admin_login'))
    name = request.form['candidate_name']
    df = pd.read_csv(vote_file)
    new_id = 1
    if not df.empty: new_id = df['candidate_id'].max() + 1
    new_row = pd.DataFrame([[new_id, name, 0]], columns=['candidate_id', 'candidate_name', 'vote_count'])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(vote_file, index=False)
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete_candidate', methods=['POST'])
def delete_candidate():
    if not session.get('is_admin'): return redirect(url_for('admin_login'))
    c_id = int(request.form['id'])
    df = pd.read_csv(vote_file)
    df = df[df['candidate_id'] != c_id]
    df.to_csv(vote_file, index=False)
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/normalize_images', methods=['POST'])
def normalize_images():
    """Route สำหรับ normalize รูปภาพเก่าทั้งหมด (ต้องเป็น admin)"""
    if not session.get('is_admin'):
        return jsonify({'status': 'error', 'message': 'ต้องเป็น admin'}), 403
    
    try:
        result = normalize_all_existing_images()
        if result['success']:
            return jsonify({
                'status': 'success',
                'message': result['message'],
                'total_normalized': result['total_normalized'],
                'total_errors': result['total_errors']
            })
        else:
            return jsonify({'status': 'error', 'message': result['message']}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'เกิดข้อผิดพลาด: {str(e)}'}), 500

@app.route('/admin/reset_votes', methods=['POST'])
def reset_votes():
    if not session.get('is_admin'): return redirect(url_for('admin_login'))
    df_votes = pd.read_csv(vote_file)
    df_votes['vote_count'] = 0
    df_votes.to_csv(vote_file, index=False)
    df_users = pd.read_csv(csv_file, dtype={'studentid': str})
    df_users['has_voted'] = 0
    df_users.to_csv(csv_file, index=False)
    return redirect(url_for('admin_dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/static/uploads/UserImages/<user_folder>/<image_name>')
def uploaded_file(user_folder, image_name):
    return send_from_directory(os.path.join(user_img_dir, user_folder), image_name)


#NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
#ngrok.set_auth_token(NGROK_AUTH_TOKEN)
#public_url = ngrok.connect(5000)
#print(f"👉 Public URL: {public_url}")

if __name__ == '__main__':
    app.run(debug=True,port=5000)