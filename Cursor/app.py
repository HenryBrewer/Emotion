from flask import Flask, render_template, Response, jsonify
import cv2
from fer import FER
import threading
import time
import random
import numpy as np

app = Flask(__name__)

detector = FER(mtcnn=True)

global_frame = None
global_result = None
global_deception_score = 0
processing = False

def process_frame():
    global global_frame, global_result, global_deception_score, processing
    prev_emotions = None
    emotion_changes = 0
    frames_count = 0
    while True:
        if global_frame is not None and not processing:
            processing = True
            global_result = detector.detect_emotions(global_frame)
            if global_result and len(global_result) > 0:
                emotions = global_result[0]["emotions"]
                
                # Check for rapid emotion changes
                if prev_emotions:
                    if max(emotions, key=emotions.get) != max(prev_emotions, key=prev_emotions.get):
                        emotion_changes += 1
                prev_emotions = emotions
                
                # Check for microexpressions (rapid, subtle changes)
                micro_expression_score = sum(abs(emotions[e] - prev_emotions.get(e, 0)) for e in emotions) if prev_emotions else 0
                
                # Calculate gaze aversion (assuming looking away means eyes are not detected)
                gaze_aversion = 0 if "eyes" in global_result[0] else 1
                
                # Update deception score
                frames_count += 1
                if frames_count >= 30:  # Update score every 30 frames
                    global_deception_score = (emotion_changes / 30 + micro_expression_score + gaze_aversion) / 3
                    emotion_changes = 0
                    frames_count = 0
            
            processing = False
        time.sleep(0.03)

threading.Thread(target=process_frame, daemon=True).start()

def generate_frames():
    global global_frame, global_result
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            global_frame = frame.copy()
            
            if global_result:
                for face in global_result:
                    bounding_box = face["box"]
                    emotions = face["emotions"]
                    emotion = max(emotions, key=emotions.get)
                    
                    color = (0, 255, 0)  # Default green
                    if emotions['angry'] > 0.3:  # Lower threshold for angry
                        color = (0, 0, 255)  # Red for angry
                        emotion = 'angry'
                    
                    cv2.rectangle(frame, (bounding_box[0], bounding_box[1]),
                                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                                  color, 2)
                    
                    cv2.putText(frame, f"{emotion}: {emotions[emotion]:.2f}", (bounding_box[0], bounding_box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    if emotion == 'angry':
                        cv2.circle(frame, (bounding_box[0] + bounding_box[2] // 2, bounding_box[1] - 30), 10, (0, 0, 255), -1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotions')
def get_emotions():
    if global_result and len(global_result) > 0:
        return jsonify(global_result[0]["emotions"])
    else:
        return jsonify({})

@app.route('/generate_report')
def generate_report():
    if global_result and len(global_result) > 0:
        emotions = global_result[0]["emotions"]
        dominant_emotion = max(emotions, key=emotions.get)
        confidence = emotions[dominant_emotion]
        
        intro_phrases = [
            "Initiating deep scan of subject's emotional state...",
            "Analyzing neural patterns for emotional resonance...",
            "Quantum emotional assessment in progress...",
            "Engaging psionic empathy modules for analysis...",
        ]
        
        detail_phrases = {
            'angry': [
                "Detecting elevated levels of cortisol and adrenaline.",
                "Subject's amygdala shows increased activity.",
                "Micro-expressions indicate suppressed aggression.",
                "Vocal analysis reveals underlying tension.",
            ],
            'happy': [
                "Endorphin levels are significantly above baseline.",
                "Facial muscles show genuine Duchenne smile patterns.",
                "Oxytocin surge detected, indicating positive social bonds.",
                "Brainwave patterns consistent with states of joy and contentment.",
            ],
            'sad': [
                "Serotonin levels are below optimal range.",
                "Pupillary response suggests emotional distress.",
                "Voice modulation indicates melancholic undertones.",
                "Posture analysis reveals subtle signs of emotional withdrawal.",
            ],
            'surprise': [
                "Sudden spike in norepinephrine levels detected.",
                "Eyebrow elevation and mouth aperture consistent with astonishment.",
                "Galvanic skin response indicates unexpected stimuli processing.",
                "Cognitive processing speed momentarily accelerated.",
            ],
            'neutral': [
                "Emotional indicators within standard deviation of baseline.",
                "Facial muscle tension at equilibrium.",
                "Autonomic nervous system in balanced state.",
                "Brainwave patterns suggest focused, non-emotional processing.",
            ],
            'fear': [
                "Elevated heart rate and respiratory patterns detected.",
                "Pupil dilation suggests heightened state of alertness.",
                "Micro-tremors detected in peripheral limbs.",
                "Amygdala activation consistent with threat response.",
            ],
            'disgust': [
                "Activation of insular cortex detected.",
                "Nasal wrinkling and upper lip elevation observed.",
                "Subtle recoil in postural analysis.",
                "Gustatory cortex shows unexpected activity.",
            ]
        }
        
        conclusion_phrases = [
            "Recommendation: Proceed with caution and adapt approach based on emotional state.",
            "Advise: Calibrate interaction protocols to align with subject's emotional frequency.",
            "Action required: Adjust environmental parameters to optimize emotional equilibrium.",
            "Note: Continue monitoring for potential emotional state fluctuations.",
        ]
        
        report = f"{random.choice(intro_phrases)}\n\n"
        report += f"Primary Emotion Detected: {dominant_emotion.capitalize()}\n"
        report += f"Confidence Level: {confidence:.2f}\n\n"
        report += "Detailed Analysis:\n"
        for phrase in random.sample(detail_phrases[dominant_emotion], 2):
            report += f"- {phrase}\n"
        report += f"\n{random.choice(conclusion_phrases)}"
        
        return jsonify({"report": report})
    else:
        return jsonify({"error": "No face detected in the current frame. Recalibrating sensors..."})

@app.route('/get_deception_score')
def get_deception_score():
    return jsonify({"deception_score": global_deception_score})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)