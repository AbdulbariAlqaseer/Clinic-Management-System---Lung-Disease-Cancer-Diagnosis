from django.contrib.auth.models import AbstractUser
from django.core.validators import RegexValidator
from django.db import models

class User(AbstractUser):
    class UsersType:
        DOCTOR = 1
        SECRETARY = 2
        PATIENT = 3
        USER_TYPE_CHOICES = (
            (DOCTOR, 'doctor'),
            (SECRETARY, 'secretary'),
            (PATIENT, 'patient'),
        )
    
    user_type = models.PositiveSmallIntegerField(choices=UsersType.USER_TYPE_CHOICES, default=UsersType.DOCTOR)
    phone_regex = RegexValidator(regex=r'^\+963\d{9}$|^09\d{8}$', message="Phone number must be entered in the format: '+' with 9 digit OR '0' with 9 digit.")
    phone_number = models.CharField(validators=[phone_regex], max_length=12, blank=False)
    username = models.CharField(max_length=255, unique=True)
    birthdate = models.DateField()

    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = ['first_name', 'last_name', 'birthdate', 'phone_number', 'email', 'password']
    
    def __str__(self):
        return self.username + ': ' + self.first_name + ' ' + self.last_name + ((', ' + self.email) if self.email else '')

class Doctor(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    general_info = models.TextField()

    class Meta:
        verbose_name = 'Doctor'
        verbose_name_plural = 'Doctors'
    
    def __str__(self):
        return str(self.user)

class Secretary(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)

    class Meta:
        verbose_name = 'Secretary'
        verbose_name_plural = 'Secretaries'
    
    def __str__(self):
        return str(self.user)

class Patient(models.Model):
    class Gender:
        MALE = 'M'
        FEMALE = 'F'
        GENDER_CHOICES = [
            (MALE, 'Male'),
            (FEMALE, 'Female'),
        ]

    gender = models.CharField(max_length=1, choices=Gender.GENDER_CHOICES)
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    last_visit_date = models.DateField(null=True, default=None)
    cancer_diagnostic = models.TextField(blank=True)
    address = models.TextField(blank=True)
    
    class Meta:
        verbose_name = 'Patient'
        verbose_name_plural = 'Patients'
    
    def __str__(self):
        return str(self.user)

