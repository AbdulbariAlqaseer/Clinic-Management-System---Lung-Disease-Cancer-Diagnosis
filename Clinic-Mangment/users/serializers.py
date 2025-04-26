from rest_framework import serializers
from .models import Doctor, Secretary, Patient
from rest_framework.validators import UniqueValidator
from django.contrib.auth.password_validation import validate_password
from django.contrib.auth import get_user_model
User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = '__all__'

    def create(self, validated_data):
        user = User.objects.create(
            username=validated_data['username'],
            email=validated_data['email'],
            first_name=validated_data['first_name'],
            last_name=validated_data['last_name'],
            birthdate=validated_data['birthdate'],
            phone_number=validated_data['phone_number'],
            user_type=validated_data['user_type']
        )

        user.set_password(validated_data['password'])
        user.save()
        print(f"done create user: {user.user_type = }")
        return user
    
    def update(self, instance, validated_data):
        password = validated_data.pop('password', None)  # Get the password if provided

        # Update other fields
        for attr, value in validated_data.items():
            setattr(instance, attr, value)

        if password:
            instance.set_password(password)  # Hash the new password
        instance.save()

        return instance

# class UserSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = User
#         fields = '__all__'
#         # exclude = ['password']

class DoctorSerializer(serializers.ModelSerializer):
    user = UserSerializer()

    class Meta:
        model = Doctor
        fields = '__all__'

    def create(self, validated_data):        
        user_data = validated_data.pop('user')
        user_serializer = UserSerializer(data=user_data)
        user_serializer.is_valid(raise_exception=True)
        user = user_serializer.save(user_type=User.UsersType.DOCTOR)
        doctor = Doctor.objects.create(user=user, **validated_data)
        return doctor
    
    def update(self, instance, validated_data):
        user_data = validated_data.pop('user')
        user = instance.user

        # Use UserSerializer to update the user
        user_serializer = UserSerializer(user, data=user_data, partial=True)
        user_serializer.is_valid(raise_exception=True)
        user_serializer.save()

        # Update the patient fields
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()

        return instance

class SecretarySerializer(serializers.ModelSerializer):
    user = UserSerializer()

    class Meta:
        model = Secretary
        fields = '__all__'

    def create(self, validated_data):        
        user_data = validated_data.pop('user')
        user_serializer = UserSerializer(data=user_data)
        user_serializer.is_valid(raise_exception=True)
        user = user_serializer.save(user_type=User.UsersType.SECRETARY)
        secretary = Secretary.objects.create(user=user, **validated_data)
        return secretary

    def update(self, instance, validated_data):
        user_data = validated_data.pop('user')
        user = instance.user

        # Use UserSerializer to update the user
        user_serializer = UserSerializer(user, data=user_data, partial=True)
        user_serializer.is_valid(raise_exception=True)
        user_serializer.save()

        # Update the patient fields
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()

        return instance

class PatientSerializer(serializers.ModelSerializer):
    user = UserSerializer()
    gender = serializers.CharField()

    class Meta:
        model = Patient
        fields = '__all__'

    def create(self, validated_data):
        user_data = validated_data.pop('user')
        user_serializer = UserSerializer(data=user_data)
        user_serializer.is_valid(raise_exception=True)
        user = user_serializer.save(user_type=User.UsersType.PATIENT)  # Assuming you have a user_type field

        patient = Patient.objects.create(user=user, **validated_data)
        return patient

    def update(self, instance, validated_data):
        user_data = validated_data.pop('user')
        user = instance.user

        # Use UserSerializer to update the user
        user_serializer = UserSerializer(user, data=user_data, partial=True)
        user_serializer.is_valid(raise_exception=True)
        user_serializer.save()

        # Update the patient fields
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()

        return instance
    
    def validate_gender(self, value):
        if value != Patient.Gender.MALE and value != Patient.Gender.FEMALE:
            raise serializers.ValidationError(
                f"Gender field is required. Choose from: '{Patient.Gender.MALE}' OR '{Patient.Gender.FEMALE}'"
            )
        return value