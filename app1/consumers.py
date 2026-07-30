import base64
import json
import cv2
import pygame
import asyncio
import time
import pytz
import sqlite3
from collections import deque
from channels.generic.websocket import AsyncWebsocketConsumer
from ultralytics import YOLO
from datetime import datetime

CONFIDENCE_THRESHOLD = 0.3

# Load YOLO model
model = YOLO("best.pt")

# Camera initialization (FIXED)
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video_capture.set(cv2.CAP_PROP_FPS, 30)

if not video_capture.isOpened():
    print("ERROR: Camera not accessible")

# Sound alert
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("app1/alert.mp3")


# Database setup
def setup_database():
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()

    cursor.execute(
        """CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        class_id INTEGER,
                        class_name TEXT,
                        confidence REAL,
                        track_id INTEGER,
                        bbox TEXT,
                        timestamp TEXT
                    )"""
    )

    conn.commit()
    return conn, cursor


conn, cursor = setup_database()


class VideoStreamConsumer(AsyncWebsocketConsumer):

    active_connections = set()
    current_class_ids = []
    tracked_objects = {}
    object_entry_time = {}
    object_counter = 0
    ALERT_TIME_THRESHOLD = 3

    async def connect(self):
        print("WebSocket connected")

        self.active_connections.add(self)
        await self.accept()

        if len(self.active_connections) == 1:
            asyncio.create_task(self.stream_video())

    async def disconnect(self, close_code):
        print("WebSocket disconnected")
        self.active_connections.discard(self)

    async def receive(self, text_data):

        data = json.loads(text_data)

        if data.get("action") == "set_class_ids":
            self.current_class_ids = data.get("class_ids", [])

        elif data.get("action") == "set_confidence_threshold":
            new_threshold = data.get("confidence_threshold")

            if isinstance(new_threshold, (int, float)) and 0 <= new_threshold <= 1:
                global CONFIDENCE_THRESHOLD
                CONFIDENCE_THRESHOLD = new_threshold

    def calculate_iou(self, box1, box2):

        x1, y1, x2, y2 = box1
        x1b, y1b, x2b, y2b = box2

        inter_x1 = max(x1, x1b)
        inter_y1 = max(y1, y1b)
        inter_x2 = min(x2, x2b)
        inter_y2 = min(y2, y2b)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2b - x1b) * (y2b - y1b)

        union = area1 + area2 - inter_area

        if union == 0:
            return 0

        return inter_area / union

    def save_detection(self, detection):

        riyadh_tz = pytz.timezone("Asia/Kolkata")

        timestamp = datetime.now(riyadh_tz).strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            """INSERT INTO detections
               (class_id,class_name,confidence,track_id,bbox,timestamp)
               VALUES (?,?,?,?,?,?)""",
            (
                detection["class_id"],
                detection["class_name"],
                detection["confidence"],
                detection["track_id"],
                str(detection["bbox"]),
                timestamp,
            ),
        )

        conn.commit()

    def assign_tracking_ids(self, results, frame_id):

        detections = []

        current_time = time.time()

        for result in results[0].boxes:

            x1, y1, x2, y2 = map(int, result.xyxy[0])

            confidence = float(result.conf[0])
            class_id = int(result.cls[0])

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            detection = {
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence,
                "class_id": class_id,
                "class_name": model.names[class_id],
            }

            assigned = False

            for track_id, tracked_data in list(self.tracked_objects.items()):

                tracked_bbox = tracked_data[-1][1]

                iou = self.calculate_iou(detection["bbox"], tracked_bbox)

                if iou > 0.3:
                    tracked_data.append((frame_id, detection["bbox"]))
                    detection["track_id"] = track_id
                    assigned = True
                    break

            if not assigned:
                self.object_counter += 1
                track_id = self.object_counter
                self.tracked_objects[track_id] = deque([(frame_id, detection["bbox"])])
                detection["track_id"] = track_id

            self.save_detection(detection)

            if class_id in [1, 2]:

                if detection["track_id"] not in self.object_entry_time:
                    self.object_entry_time[detection["track_id"]] = current_time

                else:

                    duration = current_time - self.object_entry_time[detection["track_id"]]

                    if duration > self.ALERT_TIME_THRESHOLD:
                        pygame.mixer.Sound.play(alert_sound)
                        del self.object_entry_time[detection["track_id"]]

            detections.append(detection)

        return detections

    async def stream_video(self):

        frame_id = 0

        while self.active_connections:

            ret, frame = video_capture.read()

            if not ret:
                print("Camera frame error")
                await asyncio.sleep(0.1)
                continue

            results = model(frame, conf=0.3, verbose=False)

            detections = self.assign_tracking_ids(results, frame_id)

            for detection in detections:

                x1, y1, x2, y2 = detection["bbox"]

                color = (0, 255, 0)

                if detection["class_id"] in [1, 2]:
                    color = (0, 0, 255)

                label = f'{detection["class_name"]} {detection["confidence"]:.2f}'

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            _, buffer = cv2.imencode(".jpg", frame)

            frame_data = base64.b64encode(buffer).decode("utf-8")

            message = json.dumps({"frame": frame_data, "detections": detections})

            tasks = [
                connection.send(text_data=message)
                for connection in self.active_connections
            ]

            await asyncio.gather(*tasks, return_exceptions=True)

            await asyncio.sleep(0.03)

            frame_id += 1