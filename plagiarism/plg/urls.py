
from django.urls import path, include

from plg.views import *
urlpatterns = [
    path('api/v1/preprocess_text_view/', PreprocessTextView.as_view(), name="anything")
]
