from django.contrib import admin
from django.urls import path, include
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from django.conf import settings
from django.conf.urls.static import static
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework import serializers, status
from rest_framework.response import Response

class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        token['user_type'] = user.user_type
        return token

    def validate(self, attrs):
        data = super().validate(attrs)
        data['user_type'] = self.user.user_type
        return data

class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer
    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        try:
            serializer.is_valid(raise_exception=True)
        except serializers.ValidationError:
            return Response({"detail": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)
        response_data = serializer.validated_data
        return Response(response_data, status=status.HTTP_200_OK)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('schedule.urls')),
    path('api/accounts/', include('users.urls')),
    path('login/', CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('login/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)