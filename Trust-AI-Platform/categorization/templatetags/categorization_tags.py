from django import template

register = template.Library()


@register.filter
def pct(value):
    """0.75 -> "75.0%"; a missing value (None, '') -> an en dash."""
    if value is None or value == '':
        return '\u2013'
    return f'{value * 100:.1f}%'
