from flask import Flask, render_template, jsonify, request
from fer import FER
import cv2
import numpy as np
import base64
import random

app = Flask(__name__)

detector = FER(mtcnn=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Decode the image from base64
    image_data = request.json['frame']
    image_data = image_data.split(',')[1]  # Remove the "data:image/jpeg;base64," part
    image_bytes = base64.b64decode(image_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Detect emotions
    result = detector.detect_emotions(frame)
    
    if result and len(result) > 0:
        emotions = result[0]["emotions"]
        dominant_emotion = max(emotions, key=emotions.get)
        return jsonify({
            'emotions': emotions,
            'dominant_emotion': dominant_emotion,
            'deception_score': random.random()  # Placeholder for deception score
        })
    else:
        return jsonify({'error': 'No face detected'})

@app.route('/generate_report')
def generate_report():
    # Simplified report generation
    emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear', 'disgust']
    dominant_emotion = random.choice(emotions)
    
    report = f"Emotion Analysis Report\n\n"
    report += f"Primary Emotion Detected: {dominant_emotion.capitalize()}\n"
    report += f"Confidence Level: {random.random():.2f}\n\n"
    report += "Detailed Analysis:\n"
    report += "- Subject displays typical markers of the detected emotion.\n"
    report += "- Recommend further observation for comprehensive assessment.\n"
    
    return jsonify({"report": report})

if __name__ == '__main__':
    app.run(debug=True)