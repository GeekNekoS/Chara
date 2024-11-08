from django.shortcuts import render
from django.views import View
from .forms import ImageForm


class IndexView(View):
    def get(self, request):
        return render(request, 'main/index.html')


class PhotoView(View):
    def get(self, request):
        form = ImageForm()
        return render(request, 'main/photo.html', {'form': form})


class ImageUploadView(View):
    """Обработка изображений, загруженных пользователями"""

    def get(self, request):
        form = ImageForm()
        return render(request, 'main/photo.html', {'form': form})

    def post(self, request):
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            img_obj = form.instance
            return render(request, 'main/photo.html', {'form': form, 'img_obj': img_obj})
        return render(request, 'main/photo.html', {'form': form})

