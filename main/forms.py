from django import forms
from .models import Image


class ImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ('title', 'image')

    def clean_image(self):
        image = self.cleaned_data.get('image')
        # Добавьте дополнительные проверки для валидации изображения, если необходимо
        return image
