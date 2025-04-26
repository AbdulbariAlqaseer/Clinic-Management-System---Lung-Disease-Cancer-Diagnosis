from rest_framework import serializers

from .models import Appointment, Visit
from users.serializers import PatientSerializer, DoctorSerializer
from users.models import Patient

class AppointmentSerializer(serializers.ModelSerializer):
    patient = serializers.PrimaryKeyRelatedField(queryset=Patient.objects.all(), write_only=True)
    patient_info = PatientSerializer(source='patient', read_only=True)
    class Meta:
        model = Appointment
        fields = '__all__'

class VisitSerializer(serializers.ModelSerializer):
    patient = serializers.PrimaryKeyRelatedField(queryset=Patient.objects.all(), required=False, write_only=True)
    patient_info = PatientSerializer(source='patient', read_only=True)
    class Meta:
        model = Visit
        fields = '__all__'
        
    def validate(self, data):
        try:
            self.Meta.model.objects.get(date=data['date'], start_time=data['start_time'])
            raise serializers.ValidationError('found visit in same date')
        except self.Meta.model.DoesNotExist:
            pass

        return data