import base64
import os
import numpy as np
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
import cv2
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
from gtts import gTTS
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['PROCESSED_FOLDER'] = 'runs/detect/exp/'

# Load pre-trained model, feature extractor, and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load YOLO model for object detection
yolo_model = YOLO('yolov8n.pt')  # Use appropriate YOLO model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set generation parameters
gen_kwargs = {"max_length": 50, "num_beams": 5, "early_stopping": True}

def preprocess_image(image_path):
    try:
        # Open the image
        image = Image.open(image_path).convert("RGB")
        # Enhance the image using contrast and edge sharpening
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        # Sharpen edges
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced_gray, -1, kernel)
        # Merge back to RGB
        enhanced_image_np = cv2.merge([sharpened]*3)
        enhanced_image = Image.fromarray(enhanced_image_np)
        return enhanced_image
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None

def detect_objects(image_path):
    try:
        # Load and process image
        image = cv2.imread(image_path)
        results = yolo_model(image)
        detected_objects = []

        # Ensure save directory exists
        save_dir = app.config['PROCESSED_FOLDER']
        os.makedirs(save_dir, exist_ok=True)

        for result in results:
            for obj in result.boxes:
                class_id = int(obj.cls[0].item())
                object_name = yolo_model.names[class_id]
                detected_objects.append(object_name)
                
            # Save the image with detected objects
            save_path = os.path.join(save_dir, os.path.basename(image_path))
            result.save(save_path)

        return detected_objects
    except Exception as e:
        print(f"Error in object detection: {e}")
        return []

def predict_step(image_paths):
    images = [preprocess_image(image_path) for image_path in image_paths if preprocess_image(image_path)]
    if not images:
        return []
    
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        output_ids = model.generate(pixel_values, **gen_kwargs)
    captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    detected_objects_list = [detect_objects(image_path) for image_path in image_paths]
    
    combined_results = []
    for caption, detected_objects in zip(captions, detected_objects_list):
        objects_str = ", ".join(detected_objects) if detected_objects else "No objects detected"
        combined_caption = f"{caption}. Detected objects: {objects_str}."
        combined_results.append({
            'caption': combined_caption,
            'objects': detected_objects
        })
        print(f"Caption: {caption}")
        print(f"Detected objects: {objects_str}")
    
    return combined_results

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload_image')
def upload_image():
    return render_template('index.html')

@app.route('/capture_image')
def capture_image():
    return render_template('index1.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('upload_image'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('upload_image'))
    if file:
        try:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            results = predict_step([file_path])
            if results:
                caption = results[0]['caption']
                processed_image_path = os.path.join('runs', 'detect', 'exp', filename)  # Path to the processed image
                detected_objects = ", ".join(results[0]['objects']) if results[0]['objects'] else "No objects detected"
                
                tts = gTTS(caption, lang='en')
                tts.save(os.path.join(app.config['UPLOAD_FOLDER'], 'caption.mp3'))
                
                return render_template('result.html', 
                                       image_url=url_for('uploaded_file', filename=filename),
                                       processed_image_url=url_for('processed_file', filename=filename),
                                       caption=caption,
                                       detected_objects=detected_objects)
        except Exception as e:
            print(f"Error in file upload: {e}")
            return redirect(url_for('upload_image'))
    return redirect(url_for('upload_image'))

@app.route('/upload_capture', methods=['POST'])
def upload_capture():
    if 'image' not in request.form:
        return redirect(url_for('capture_image'))
    image_data = request.form['image'].split(",")[1]
    try:
        image_bytes = base64.b64decode(image_data)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_image.png')
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        results = predict_step([image_path])
        if results:
            caption = results[0]['caption']
            processed_image_path = os.path.join('runs', 'detect', 'exp', 'captured_image.png')  # Path to the processed image
            detected_objects = ", ".join(results[0]['objects']) if results[0]['objects'] else "No objects detected"
            
            tts = gTTS(caption, lang='en')
            tts.save(os.path.join(app.config['UPLOAD_FOLDER'], 'caption.mp3'))
            
            return render_template('result.html', 
                                   image_url=url_for('uploaded_file', filename='captured_image.png'), 
                                   processed_image_url=url_for('processed_file', filename='captured_image.png'),
                                   caption=caption,
                                   detected_objects=detected_objects)
    except Exception as e:
        print(f"Error in image capture upload: {e}")
        return redirect(url_for('capture_image'))
    return redirect(url_for('capture_image'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
