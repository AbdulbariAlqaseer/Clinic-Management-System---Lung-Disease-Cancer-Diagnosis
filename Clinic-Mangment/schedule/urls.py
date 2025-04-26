from django.urls import path, include
from .views import (
    appointmentView_create, 
    appointmentView_list, 
    appointmentView_retrive,
    visitView_create,
    visitView_list,
    visitView_retrive_list,
    visitView_retrive_detail,
    visitView_update,
    visitView_delete,
    cancer_diagnosis_view
    )

urlpatterns = [
    path('appointment/list/', appointmentView_list),
    path('appointment/create/', appointmentView_create),
    path('appointment/retrive/', appointmentView_retrive),
    path('visit/list/', visitView_list),
    path('visit/create/', visitView_create),
    path('visit/retrive/', visitView_retrive_list),
    path('visit/retrive/<int:pk>', visitView_retrive_detail),
    path('visit/update/<int:pk>', visitView_update),
    path('visit/delete/<int:pk>', visitView_delete),
    path('cancerPredict/', cancer_diagnosis_view),
]