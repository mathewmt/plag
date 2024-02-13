
from django.urls import path, include

from plg.views import *
urlpatterns = [
    path('', PreprocessTextView.as_view(), name="anything")
]
