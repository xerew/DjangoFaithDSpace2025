from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views
from django.conf.urls import handler404
from django.shortcuts import redirect

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', lambda request: redirect('/authoringtool/', permanent=True)),
    path('authoringtool/', include('authoringtool.urls')),
    path('accounts/', include('accounts.urls')),
    path('organization/', include('organization.urls')),
    path('studentview/', include('studentview.urls')),
    path('usergroups/', include('usergroups.urls')),
    #path('login/', auth_views.LoginView.as_view(template_name='accounts/login.html'), name='login'),
    #path('logout/', auth_views.LogoutView.as_view(), name='logout'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

urlpatterns += static('/rag_pdfs/', document_root=settings.RAG_PDFS_ROOT)

handler404 = 'accounts.views.view404'