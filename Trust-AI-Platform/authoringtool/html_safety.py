"""Make teacher- and LLM-authored HTML safe to show in a browser.

Activity text comes from TinyMCE (often pasted from Word) and is shown to
students and other teachers, so it is cleaned every time it is rendered. The
allowlist keeps what teachers actually use (inline styles, tables, images,
video embeds) and drops scripts, event handlers and javascript: links.
"""
from urllib.parse import urlparse

import nh3
from django.conf import settings

ALLOWED_TAGS = {
    'a', 'abbr', 'audio', 'b', 'big', 'blockquote', 'br', 'caption', 'center',
    'cite', 'code', 'col', 'colgroup', 'dd', 'del', 'details', 'div', 'dl',
    'dt', 'em', 'figcaption', 'figure', 'font', 'h1', 'h2', 'h3', 'h4', 'h5',
    'h6', 'hr', 'i', 'iframe', 'img', 'ins', 'kbd', 'li', 'mark', 'ol', 'p',
    'pre', 'q', 's', 'samp', 'small', 'source', 'span', 'strike', 'strong',
    'sub', 'summary', 'sup', 'table', 'tbody', 'td', 'tfoot', 'th', 'thead',
    'tr', 'tt', 'u', 'ul', 'var', 'video',
}

ALLOWED_ATTRIBUTES = {
    '*': {
        'align', 'class', 'dir', 'height', 'id', 'lang', 'role', 'style',
        'title', 'valign', 'width',
    },
    'a': {'href', 'name', 'target'},
    'audio': {'controls', 'src'},
    'col': {'span'},
    'colgroup': {'span'},
    'font': {'color', 'face', 'size'},
    'iframe': {'allow', 'allowfullscreen', 'frameborder', 'loading', 'src'},
    'img': {'alt', 'border', 'src'},
    'ol': {'start', 'type'},
    'source': {'src', 'type'},
    'table': {'border', 'cellpadding', 'cellspacing'},
    'td': {'colspan', 'rowspan'},
    'th': {'colspan', 'rowspan', 'scope'},
    'video': {'controls', 'poster', 'src'},
}

URL_SCHEMES = {'http', 'https', 'mailto', 'tel', 'data'}

DEFAULT_EMBED_HOSTS = (
    'www.youtube.com', 'youtube.com', 'www.youtube-nocookie.com',
    'youtube-nocookie.com', 'player.vimeo.com', 'phet.colorado.edu',
    'wordwall.net', 'compassproject.net', 'labsland.com',
    'pendulum-v2.labs.labsland.com',
)


def _embed_hosts():
    return set(getattr(settings, 'HTML_EMBED_HOSTS', DEFAULT_EMBED_HOSTS))


def _filter_attribute(tag, attribute, value):
    # Iframes may only embed https pages from known learning-content hosts.
    if tag == 'iframe' and attribute == 'src':
        url = urlparse(value)
        if url.scheme != 'https' or url.hostname not in _embed_hosts():
            return None
    return value


def sanitize_html(html):
    """Return HTML with anything that can run script removed."""
    if not html:
        return ''
    return nh3.clean(
        str(html),
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        attribute_filter=_filter_attribute,
        url_schemes=URL_SCHEMES,
    )


def json_for_script(json_text):
    """Escape JSON text so it cannot close an enclosing <script> element.

    Same escaping as Django's json_script, for JSON that is already a string.
    """
    return (
        str(json_text or '')
        .replace('&', '\\u0026')
        .replace('<', '\\u003C')
        .replace('>', '\\u003E')
    )
