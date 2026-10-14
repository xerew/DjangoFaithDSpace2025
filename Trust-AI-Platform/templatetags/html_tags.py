from django import template
from django.utils.safestring import mark_safe

from authoringtool.html_safety import json_for_script, sanitize_html

register = template.Library()


@register.filter
def clean_html(value):
    """Render teacher- or LLM-authored HTML with scripts and handlers removed."""
    return mark_safe(sanitize_html(value))


@register.filter
def script_json(value):
    """Output a JSON string inside <script type="application/json"> safely."""
    return mark_safe(json_for_script(value))
