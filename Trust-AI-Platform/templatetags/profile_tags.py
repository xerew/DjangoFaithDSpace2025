from django import template
from django.templatetags.static import static

register = template.Library()


@register.filter
def avatar_url(user):
    profile = getattr(user, 'profile', None)
    if not profile:
        return ''
    if profile.picture:
        return profile.picture.url
    if profile.gender == 'male':
        return static('img/profile_d_man.webp')
    if profile.gender == 'female':
        return static('img/profile_d_woman.jpg')
    return ''
