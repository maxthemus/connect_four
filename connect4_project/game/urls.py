from django.urls import path
from . import views

urlpatterns = [
    path('play/', views.play_move, name='play_move'),  # API endpoint
]
