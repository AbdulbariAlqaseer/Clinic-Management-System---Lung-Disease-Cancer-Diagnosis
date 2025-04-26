from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import PatientSignUpView, PatientViewSet, DoctorViewSet, SecretaryViewSet, patient_retrive, Secretary_retrive

router = DefaultRouter()
router.register(r'doctors', DoctorViewSet)
router.register(r'secretaries', SecretaryViewSet)
router.register(r'patients', PatientViewSet, basename='patient')

urlpatterns = [
    path('', include(router.urls)),
    path('register/', PatientSignUpView.as_view(), name='patient-signup'),
    path('patient_retrive/', patient_retrive),
    path('secretary_retrive/', Secretary_retrive),
]