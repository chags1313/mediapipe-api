from flask import Flask, request, jsonify
import mediapipe as mp
import numpy as np
import cv2

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Define custom drawing specifications for white keypoints and lines
white_landmark_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
white_connection_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2)

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Check if an image URL is provided
        image_url = request.json.get('image_url', None)
        if image_url:
            response = requests.get(image_url)
            response.raise_for_status()  # Raise an error for bad responses
            image_np = np.frombuffer(response.content, np.uint8)
        else:
            # Get the image from the request if direct image data is sent
            image_stream = request.files['image'].read()
            image_np = np.frombuffer(image_stream, np.uint8)

        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Process the image using MediaPipe
        results = pose.process(image)

        # Draw the pose landmarks on the cartoon image
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=white_landmark_drawing_spec,
                connection_drawing_spec=white_connection_drawing_spec
            )

        # Convert the cartoon image back to a format suitable for sending via HTTP
        _, img_encoded = cv2.imencode('.jpg', image)
        img_bytes = img_encoded.tobytes()

        # Extract keypoints data
        keypoints = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                keypoints.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })

        return jsonify({
            'image': img_bytes.decode('latin1'),  # Convert bytes to string for JSON serialization
            'keypoints': keypoints
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

