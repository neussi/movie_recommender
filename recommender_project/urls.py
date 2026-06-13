from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('recommend/', views.recommend, name='recommend'),
    path('user-history/', views.user_history, name='user_history'),
    path('contact/', views.contact, name='contact'),
]
