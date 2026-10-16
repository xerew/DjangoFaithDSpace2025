from django import template
from django.templatetags.static import static
from django.urls import reverse
from django.utils.html import format_html

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


def _in_teachers(user):
    # Cached on the user object: a page can show the same person many times.
    if not hasattr(user, '_profile_tags_is_teacher'):
        user._profile_tags_is_teacher = user.groups.filter(name='teachers').exists()
    return user._profile_tags_is_teacher


@register.simple_tag(takes_context=True)
def user_link(context, user):
    """A person's name, linked to their profile when the viewer may open it.

    Same rules as the profile pages: the viewer must be a teacher, and only
    teachers and administrators have a profile others can open.
    """
    if not user:
        return ''
    name = user.get_full_name() or user.username
    request = context.get('request')
    viewer = getattr(request, 'user', None)
    if not (viewer and viewer.is_authenticated and _in_teachers(viewer)):
        return name
    if user.pk == viewer.pk:
        url = reverse('profile')
    elif user.is_staff or user.is_superuser or _in_teachers(user):
        url = reverse('view_profile', args=[user.pk])
    else:
        return name
    return format_html('<a href="{}" class="user-link">{}</a>', url, name)
