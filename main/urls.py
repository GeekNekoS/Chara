from django.urls import path
from .views import IndexView, PhotoView, ImageUploadView

urlpatterns = [
    path('', IndexView.as_view(), name='home'),
    path('photo/', PhotoView.as_view(), name='photo'),
    path('upload/', ImageUploadView.as_view(), name='image_upload'),
]
