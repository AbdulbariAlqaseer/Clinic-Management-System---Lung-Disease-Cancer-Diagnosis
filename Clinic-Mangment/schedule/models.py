from django.db import models
from users.models import Doctor, Patient
from django.conf import settings
from datetime import datetime, time, timedelta
from django.utils.timezone import localdate 

class Appointment(models.Model):
    date = models.DateField()
    start_time = models.TimeField()
    
    # doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE)
    patient = models.ForeignKey(Patient, on_delete=models.SET_NULL, null=True)
    
    class Meta:
        ordering = ['-date', '-start_time']
    
    def __str__(self):
        return f"date:{self.date}, start time:{self.start_time}"

    def day_name(self):
        return self.date.strftime('%A')

    def get_time_next_appointment(self):
        return (datetime.combine(localdate(), self.start_time) + settings.APPOINTMENT_DUR).time()


class Visit(models.Model):
    diagnosis_details = models.TextField(blank=True, default="")
    radio_image = models.ImageField(upload_to='images/')
    analysis_image = models.CharField(max_length=1024, null=True, default=None)
    # appointment = models.OneToOneField(Appointment, on_delete=models.SET_NULL, null=True)
    patient = models.ForeignKey(Patient, on_delete=models.SET_NULL, null=True)
    date = models.DateField(null=True, default=None)
    start_time = models.TimeField(null=True, default=None)
    
    class Meta:
        ordering = ['-date', '-start_time']

    def __str__(self):
        return f"date:{self.date}, start time:{self.start_time}"