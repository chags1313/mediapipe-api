from flask import Flask, request, jsonify, send_file
import mediapipe as mp
import numpy as np
import cv2
from io import BytesIO

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

@app.route('/process_image', methods=['POST'])
def process_image():
    # Get the image from the request
    image_stream = request.files['image'].read()
    image_np = np.frombuffer(image_stream, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Process the image using MediaPipe
    results = pose.process(image)

    # Draw the pose landmarks on the image
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Convert the image back to a format suitable for sending via HTTP
    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = img_encoded.tobytes()

    # Extract keypoints data
    keypoints = []
    for landmark in results.pose_landmarks.landmark:
        keypoints.append({
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z,
            'visibility': landmark.visibility
        })

    # Send the processed image and keypoints data as response
    return jsonify({
        'image': img_bytes.decode('latin1'),  # Convert bytes to string for JSON serialization
        'keypoints': keypoints
    })

if __name__ == '__main__':
    app.run(debug=True)
