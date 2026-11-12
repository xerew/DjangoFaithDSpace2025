from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views
from django.conf.urls import handler404
from django.shortcuts import redirect

from authoringtool.ollama_tunnel_views import ai_server_connection_view

urlpatterns = [
    # Before admin.site.urls, which would otherwise treat it as an app label.
    path(
        'admin/ai-server/',
        admin.site.admin_view(ai_server_connection_view),
        name='ai_server_connection',
    ),
    path('admin/', admin.site.urls),
    path('', lambda request: redirect('/home/', permanent=True)),
    path('home/', include('home.urls')),
    path('authoringtool/', include('authoringtool.urls')),
    path('accounts/', include('accounts.urls')),
    path('organization/', include('organization.urls')),
    path('studentview/', include('studentview.urls')),
    path('usergroups/', include('usergroups.urls')),
    path('messaging/', include('messaging.urls')),
    path('feedback/', include('feedback.urls')),
    #path('login/', auth_views.LoginView.as_view(template_name='accounts/login.html'), name='login'),
    #path('logout/', auth_views.LogoutView.as_view(), name='logout'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

# RAG PDFs are NOT served as unauthenticated static files.
# They are served through the authenticated `serve_rag_pdf` view registered in authoringtool/urls.py.

handler404 = 'accounts.views.view404'