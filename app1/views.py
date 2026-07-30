from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
import sqlite3
from collections import Counter
from datetime import datetime
import cv2
from ultralytics import YOLO
from .ml_model import get_predicted_violations

# ------------------- Config -------------------

DB_PATH = 'detections.db'

# Load PPE detection model
model = YOLO("best.pt")

# Automatically get class names from model
CLASS_NAMES = model.names

# Initialize webcam
camera = cv2.VideoCapture(0)


# ------------------- Frame Generator -------------------

def gen_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        # Run YOLO detection
        results = model(frame, conf=0.3)

        annotated_frame = frame.copy()

        for result in results:

            boxes = result.boxes

            for box in boxes:

                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = CLASS_NAMES[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])



                # Color for violation
                if class_name in ["No Helmet", "NO-Safety Vest"]:
                    color = (0,0,255)
                else:
                    color = (0,255,0)

                # Draw bounding box
                cv2.rectangle(
                    annotated_frame,
                    (x1,y1),
                    (x2,y2),
                    color,
                    2
                )

                label = f"{class_name} {confidence:.2f}"

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                # Save detection
                save_detection(cls_id, class_name, confidence, [x1,y1,x2,y2])

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ------------------- Video Feed -------------------

def video_feed(request):

    return StreamingHttpResponse(
        gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


# ------------------- Database -------------------

def save_detection(class_id, class_name, confidence, bbox):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        """
        INSERT INTO detections
        (class_id, class_name, confidence, track_id, bbox, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            class_id,
            class_name,
            confidence,
            0,
            str(bbox),
            timestamp
        )
    )

    conn.commit()
    conn.close()


def get_detections_from_db():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM detections ORDER BY timestamp DESC")

    detections = cursor.fetchall()

    conn.close()

    return detections


# ------------------- Analytics -------------------

def calculate_time_based_counts(detections):

    time_counts = {
        '6 AM - 12 PM': 0,
        '12 PM - 6 PM': 0,
        '6 PM - 12 AM': 0,
        '12 AM - 6 AM': 0
    }

    for d in detections:

        timestamp = datetime.strptime(d[6], "%Y-%m-%d %H:%M:%S")

        hour = timestamp.hour

        if 6 <= hour < 12:
            time_counts['6 AM - 12 PM'] += 1

        elif 12 <= hour < 18:
            time_counts['12 PM - 6 PM'] += 1

        elif 18 <= hour < 24:
            time_counts['6 PM - 12 AM'] += 1

        else:
            time_counts['12 AM - 6 AM'] += 1

    return time_counts


def calculate_most_frequent_classes(detections, top_n=5):

    class_names = [d[2] for d in detections]

    counts = Counter(class_names)

    return counts.most_common(top_n)


# ------------------- Views -------------------

def ppe_detection(request):

    return render(request, 'ppe_detection.html')


def detection_list(request):

    detections = get_detections_from_db()

    # Use Machine Learning Regression model to predict upcoming violations
    predicted_violations = get_predicted_violations()

    return render(request, 'detection_logs.html', {

        'detections': detections,
        'total_detections': len(detections),
        'total_class_ids': len({d[1] for d in detections}),
        'total_class_names': len({d[2] for d in detections}),
        'most_frequent_classes': calculate_most_frequent_classes(detections),
        'time_based_counts': calculate_time_based_counts(detections),
        'predicted_violations': predicted_violations,

    })


def fetch_detections(request):

    detections = get_detections_from_db()

    detections_data = [
        {
            'id': d[0],
            'class_id': d[1],
            'class_name': d[2],
            'confidence': d[3],
            'track_id': d[4],
            'bbox': d[5],
            'timestamp': d[6]
        }
        for d in detections
    ]

    return JsonResponse({

        'detections': detections_data,
        'total_detections': len(detections),
        'total_class_ids': len({d[1] for d in detections}),
        'total_class_names': len({d[2] for d in detections}),
        'most_frequent_classes': calculate_most_frequent_classes(detections),
        'time_based_counts': calculate_time_based_counts(detections),

    })