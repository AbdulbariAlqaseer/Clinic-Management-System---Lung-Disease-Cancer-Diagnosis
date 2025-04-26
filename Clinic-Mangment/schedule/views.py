from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework import viewsets, status
from rest_framework import generics
from rest_framework.response import Response
from rest_framework.parsers import JSONParser
from .serializers import AppointmentSerializer, VisitSerializer
from .models import Appointment, Visit
from users.permissions import IsPatient, IsDoctor, IsSecretary
from django.conf import settings
from users.models import Patient
from django.utils.timezone import localdate
from django.shortcuts import get_object_or_404
from django.contrib.auth import get_user_model
User = get_user_model()

from lung_ai.Deep_Learing import LungModel
from lung_ai.lung_cancer import cancerDiagnosis

@api_view(['POST'])
@permission_classes([IsAuthenticated, IsDoctor | IsSecretary])
def appointmentView_list(request):
    start_date = request.data['start_date']
    end_date = request.data['end_date']
    appointments = Appointment.objects.filter(date__range=[start_date, end_date])
    serializer = AppointmentSerializer(appointments, many=True)
    return Response(serializer.data, status=status.HTTP_200_OK)

@api_view(['POST'])
@permission_classes([IsAuthenticated, IsPatient])
def appointmentView_create(request):    
    patient = Patient.objects.get(user__id=request.user.id)
    appointment = Appointment.objects.filter(patient=patient, date__gte=localdate())
    if len(appointment) != 0:
        serializer = AppointmentSerializer(appointment.first())
        return Response({"message":"you have another appointment", "appointment": serializer.data}, status=status.HTTP_400_BAD_REQUEST)
    
    appointments_in_date = Appointment.objects.filter(date=request.data.get('date'))
    if len(appointments_in_date) >= settings.NUM_APPOINTMENT_IN_DAY:
        return Response({"message":"this day is full, choose another day"}, status=status.HTTP_400_BAD_REQUEST)
    elif len(appointments_in_date) == 0:
        start_time = settings.TIME_OPEN
    else:
        start_time = appointments_in_date.first().get_time_next_appointment()

    request.data['patient'] = patient.id
    request.data['start_time'] = start_time
    
    serializer = AppointmentSerializer(data=request.data)
    
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([IsAuthenticated, IsPatient])
def appointmentView_retrive(request):
    appointments = Appointment.objects.filter(patient__user__id=request.user.id)
    serializer = AppointmentSerializer(appointments, many=True)
    return Response(serializer.data, status=status.HTTP_200_OK)

@api_view(['Post'])
@permission_classes([IsAuthenticated, IsDoctor | IsSecretary])
def visitView_list(request):
    start_date = request.data['start_date']
    end_date = request.data['end_date']
    visits = Visit.objects.filter(date__range=[start_date, end_date])
    serializer = VisitSerializer(visits, many=True)
    return Response(serializer.data, status=status.HTTP_200_OK)

@api_view(['POST'])
@permission_classes([IsAuthenticated, IsDoctor | IsSecretary])
def visitView_create(request):
    serializer = VisitSerializer(data=request.data)
    if serializer.is_valid():
        visit = serializer.save()
        out = []
        for k, v in LungModel.LungModel.predict(visit.radio_image.path).items():
            out += [k, v]
        visit.analysis_image = str(out)
        visit.save()
        
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def visitView_retrive_list(request):
    try:
        if request.user.user_type == User.UsersType.PATIENT:
            visits = Visit.objects.filter(patient__user__id=request.user.id)
        else:
            visits = Visit.objects.filter(patient__id=request.data['patient'])
    except:
        return Response({'message': 'The patient is messing or not found.'}, status=status.HTTP_400_BAD_REQUEST)
    
    serializer = VisitSerializer(visits, many=True)
    return Response(serializer.data, status=status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def visitView_retrive_detail(request, pk):
    visit = Visit.objects.get(pk=pk)
    return Response(VisitSerializer(visit).data, status=status.HTTP_200_OK)

@api_view(['PUT'])
@permission_classes([IsAuthenticated, IsDoctor | IsSecretary])
def visitView_update(request, pk):
    visit = get_object_or_404(Visit, pk=pk)
    visit.diagnosis_details = request.data['diagnosis_details']
    visit.save()
    return Response(VisitSerializer(visit).data, status=status.HTTP_200_OK)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated, IsDoctor | IsSecretary])
def visitView_delete(request, pk):
    visit = get_object_or_404(Visit, pk=pk)
    visit.delete()
    return Response(VisitSerializer(visit).data, status=status.HTTP_200_OK)

@api_view(['POST'])
@permission_classes([IsAuthenticated, IsPatient])
def cancer_diagnosis_view(request):
    try:
        age = request.data["age"]
        smoke = request.data["smoke"]
        cough = request.data["cough"]
        blood_cough = request.data["blood_cough"]
        weight_loss = request.data["weight_loss"]
        shortness_breath = request.data["shortness_breath"]
        bone_pains = request.data["bone_pains"]
        hoarseness_voice = request.data["hoarseness_voice"]
        chest_pain = request.data["chest_pain"]
        cancer_predict = cancerDiagnosis.get_membership_predict(
            age = age,
            smoke = smoke,
            cough = cough,
            blood_cough = blood_cough,
            weight_loss = weight_loss,
            shortness_breath = shortness_breath,
            bone_pains = bone_pains,
            hoarseness_voice = hoarseness_voice,
            chest_pain = chest_pain
        )
        # out = []
        # for k, v in cancer_predict.items():
        #     out += [k, v]
        return Response(cancer_predict, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"message": str(e)}, status=status.HTTP_400_BAD_REQUEST)