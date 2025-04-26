from rest_framework.permissions import BasePermission
from .models import User
from .serializers import PatientSerializer

class IsDoctor(BasePermission):
    def has_permission(self, request, view):
        return request.user.user_type == User.UsersType.DOCTOR

class IsSecretary(BasePermission):
    def has_permission(self, request, view):
        return request.user.user_type == User.UsersType.SECRETARY

class IsPatient(BasePermission):
    def has_permission(self, request, view):
        return request.user.user_type == User.UsersType.PATIENT

class IsTargetUser(BasePermission):
    def has_object_permission(self, request, view, obj):
        return request.user.id == obj.user.id