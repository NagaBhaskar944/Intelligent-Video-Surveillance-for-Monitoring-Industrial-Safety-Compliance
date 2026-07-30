from django.urls import path
from app1 import views

urlpatterns = [
    # Main PPE Detection Page
    path('', views.ppe_detection, name='ppe_detection'),

    # Detection logs page
    path('detections/', views.detection_list, name='detection_list'),

    # JSON API to fetch latest detections
    path('fetch-detections/', views.fetch_detections, name='fetch_detections'),

    # Live video feed from webcam (for streaming)
    path('video_feed/', views.video_feed, name='video_feed'),
]