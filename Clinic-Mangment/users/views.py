from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework import viewsets, status
from rest_framework import generics
from rest_framework.response import Response

from .permissions import IsDoctor, IsSecretary, IsPatient, IsTargetUser
from .serializers import PatientSerializer, SecretarySerializer, DoctorSerializer
from .models import Patient, Secretary, Doctor

from django.contrib.auth import get_user_model
User = get_user_model()

class PatientSignUpView(generics.CreateAPIView):
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer
    permission_classes= (AllowAny,)

class DoctorViewSet(viewsets.ModelViewSet):
    queryset = Doctor.objects.all()
    serializer_class = DoctorSerializer
    permission_classes = [IsAuthenticated, IsDoctor]
    
    def perform_destroy(self, instance):
        instance.user.delete()
        return super().perform_destroy(instance)

class SecretaryViewSet(viewsets.ModelViewSet):
    queryset = Secretary.objects.all()
    serializer_class = SecretarySerializer
    def get_permissions(self):
        if self.action == 'retrieve':            
            permission_classes = [IsAuthenticated, IsTargetUser | IsDoctor]        
        else:            
            permission_classes = [IsAuthenticated, IsDoctor]
        return [permission() for permission in permission_classes]   
    
    def perform_destroy(self, instance):
        instance.user.delete()
        return super().perform_destroy(instance)

class PatientViewSet(viewsets.ModelViewSet):
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer

    def get_permissions(self):
        if self.action == 'retrieve':            
            permission_classes = [IsAuthenticated, IsTargetUser | IsDoctor | IsSecretary]        
        else:            
            permission_classes = [IsAuthenticated, IsDoctor | IsSecretary]
        return [permission() for permission in permission_classes]   

    def perform_destroy(self, instance):
        instance.user.delete()
        return super().perform_destroy(instance)

@api_view(['GET'])
def patient_retrive(request):
    if request.method == 'GET':
        try:
            patient = Patient.objects.get(user__id=request.user.id)
            serializer = PatientSerializer(patient)
            return Response(serializer.data)
        except:
            return Response(
                {"error": "User not found"},
                status=status.HTTP_404_NOT_FOUND
            )

@api_view(['GET'])
def Secretary_retrive(request):
    if request.method == 'GET':
        try:
            secretary = Secretary.objects.get(user__id=request.user.id)
            serializer = SecretarySerializer(secretary)
            return Response(serializer.data)
        except:
            return Response(
                {"error": "User not found"},
                status=status.HTTP_404_NOT_FOUND
            )