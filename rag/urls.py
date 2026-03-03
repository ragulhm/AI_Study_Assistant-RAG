from django.urls import path
from .views import ask_page, upload_pdf

urlpatterns = [
    path("", ask_page, name="home"),
    path("upload/", upload_pdf, name="upload"),
]