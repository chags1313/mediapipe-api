from flask import Flask, request, jsonify, send_file
import mediapipe as mp
import numpy as np
import cv2
from io import BytesIO
import os

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

@app.route('/process_image', methods=['POST'])
def process_image():
    # Get the image from the request
    image_stream = request.files['image'].read()
    image_np = np.frombuffer(image_stream, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Define custom drawing specifications
    white_landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
    white_connection_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2)
    
    # Process the image using MediaPipe
    results = pose.process(image)
    
    # Draw the pose landmarks on the image
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=white_landmark_drawing_spec,
            connection_drawing_spec=white_connection_drawing_spec
        )
    
        # Extract keypoints data
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
    else:
        keypoints = []

    # Send the processed image and keypoints data as response
    return jsonify({
        'image': img_bytes.decode('latin1'),  # Convert bytes to string for JSON serialization
        'keypoints': keypoints
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
