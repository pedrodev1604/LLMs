from django.urls import path
from .views import ExtrairCSVAPIView, ExtrairConlluAPIView
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('extrair-csv/', ExtrairCSVAPIView.as_view(), name='extrair-csv'),
    path('extrair-conllu/', ExtrairConlluAPIView.as_view()),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
