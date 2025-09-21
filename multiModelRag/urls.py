from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chat_page, name="chat_page"),
    path("chat_api/", views.chat_api, name="chat_api"),
    path("chat_history/<str:session_id>/", views.get_chat_history, name="chat_history"),
    path("upload_pdf/", views.upload_pdf, name="upload_pdf"),
]
