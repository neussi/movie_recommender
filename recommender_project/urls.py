from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('recommender/', views.recommender_view, name='recommender_view'),
    path('analytics/', views.analytics_view, name='analytics_view'),
    path('contact/', views.contact_view, name='contact_view'),
    path('recommend/', views.recommend, name='recommend'),
    path('user-history/', views.user_history, name='user_history'),
    path('contact-submit/', views.contact, name='contact'),
]
