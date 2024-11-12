
import base64
import os

import cv2
from deepface import DeepFace
from flask import Blueprint, render_template, url_for, request, jsonify
from flask_login import login_required, current_user
import moviepy.editor as mp
import numpy as np

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
webcam_statistics = {}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def label_cv_frame(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    face_emotions = []

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        
        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']
        face_emotions.append(emotion)

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return frame, face_emotions



# -----
ai = Blueprint('ai', __name__)


@ai.route('/webcam')
def webcam():
    webcam_statistics[str(current_user.id)] = {emotion : 0 for emotion in EMOTIONS}
    webcam_statistics[str(current_user.id)]["n_faces"] = 0
    return render_template('webcam.html')

@ai.route('/process_image', methods=['POST'])
def process_webcam_image():
    # Recebe a imagem em formato base64 do frontend
    data = request.get_json()
    image_data = data['image']

    if webcam_statistics[str(current_user.id)]["n_faces"] == 0:
        emotions = [0 for emotion in EMOTIONS]
    else:
        emotions = [webcam_statistics[str(current_user.id)][emotion] / webcam_statistics[str(current_user.id)]["n_faces"] * 100 for emotion in EMOTIONS]

    image_np = np.frombuffer(base64.b64decode(image_data.split(',')[1]), dtype=np.uint8)
    if len(image_np) > 0:
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Image processing
        processed_image, face_emotions = label_cv_frame(frame = image)

        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

        for emotion in face_emotions:
            webcam_statistics[str(current_user.id)][emotion] += 1
        webcam_statistics[str(current_user.id)]["n_faces"] += len(face_emotions)

        if webcam_statistics[str(current_user.id)]["n_faces"] == 0:
            emotions = [0 for emotion in EMOTIONS]
        else:
            emotions = [webcam_statistics[str(current_user.id)][emotion] / webcam_statistics[str(current_user.id)]["n_faces"] * 100 for emotion in EMOTIONS]

        return jsonify({'processedImage': 'data:image/jpeg;base64,' + processed_image_base64, 'emotions': emotions })
    return jsonify({'processedImage': 'data:image/jpeg;base64,' + image_data, 'emotions': emotions})

@ai.route('/upload')
def upload():
    return render_template("upload.html")

@ai.route('/upload_video', methods = ['POST'])
def upload_video():
    ALLOWED_EXTENSIONS = ['mp4']
    def allowed_file(filename) -> bool:
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    if 'video' not in request.files:
        return "No video file found."
    
    video = request.files['video']
    
    if video.filename == "":
        return "No video file selected."
    
    if video and allowed_file(video.filename):
        os.makedirs('./ai_app/static/videos/', exist_ok = True)
        input_path = f'./ai_app/static/videos/{video.filename}'
        output_path = f'./ai_app/static/videos/processed_{video.filename}'

        # Save the uploaded video temporarily
        video.save(input_path)

        # Process the video
        video_statistics = process_video(input_path, output_path)
        print(video_statistics)

        return render_template('preview.html', video_name = f'processed_{video.filename}', emotions = video_statistics)

    return "Invalid file type."

def process_video(input_path, output_path):   
    video_statistics = {emotion : 0 for emotion in EMOTIONS}
    video_statistics["n_faces"] = 0

    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec

    # Prepare the output video writer
    temp_output_path = 'temp_processed_video.mp4'
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    # Process each frame
    while True:
        ret, image = cap.read()
        if not ret:
            break

        # Image processing
        processed_image, face_emotions = label_cv_frame(frame = image)
        video_statistics['n_faces'] += len(face_emotions)
        for face_emotion in face_emotions:
            video_statistics[face_emotion] += 1

        # Write the processed frame to the output video
        out.write(processed_image)

    # Release resources
    cap.release()
    out.release()

    # Combine processed video with original audio
    add_audio_to_video(temp_output_path, input_path, output_path)

    # Cleanup temporary file
    os.remove(temp_output_path)

    video_statistics = [video_statistics[emotion] / video_statistics['n_faces'] * 100 for emotion in EMOTIONS]

    return video_statistics

def add_audio_to_video(video_path, original_path, output_path):
    # Load the video clips
    original_video = mp.VideoFileClip(original_path)
    processed_video = mp.VideoFileClip(video_path)

    # Combine the processed video with the original audio
    final_video = processed_video.set_audio(original_video.audio)

    # Write the output video file
    final_video.write_videofile(output_path, codec="libx264")
