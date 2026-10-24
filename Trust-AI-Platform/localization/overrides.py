"""Reviewers' translation corrections, applied at runtime on top of the catalogs.

How it works
------------
Django keeps one ``DjangoTranslation`` per language and process
(``trans_real.translation(code)``). Its ``_catalog`` is a ``TranslationCatalog``
(Django 3.2+, so both 4.0 in production and 5.2 locally): a list of dicts
(``_catalogs``), each with its plural function (``_plurals``), searched in order.

``apply_overrides(code)`` puts one extra dict, the override layer, at the front
of that list, with the project catalog's plural function. Singular texts are
keyed ``msgid`` (``"msgctxt\\x04msgid"`` with a context) and plural forms
``(key, plural_index)``, exactly like the compiled ``.mo``. Because the layer is
searched first it wins; because the ``.po``/``.mo`` dicts underneath are never
written to, removing an entry from the layer brings back the original value,
and the English fallback (a separate translation object) is never touched.

Several processes
-----------------
Each gunicorn worker and Celery process has its own catalogs. Every save calls
``bump_overrides_version()``, which stores a new value under the shared cache
key ``localization:overrides-version`` (Redis in production).
``ensure_overrides(code)`` compares that value with the one this process last
applied for ``code`` and reloads the layer from the database when they differ,
or when Django has rebuilt the translation object since. It is called:

- by ``TranslationOverrideMiddleware`` for the language of every request, and
- by ``localization.languages.recipient_language``, which every email site uses
  to pick the recipient's language, so emails rendered in Celery (no request,
  no middleware) also carry the corrections.

When the cache gives no value (the dummy cache used by tests, or Redis down),
the layer is reloaded on every call: overrides are few, so this stays cheap.
English is the source language and is never overridden.
"""
import gettext as gettext_module
import threading
import time

from django.conf import settings
from django.core.cache import cache
from django.utils.translation import trans_real

VERSION_KEY = 'localization:overrides-version'
CONTEXT_SEPARATOR = '\x04'

_lock = threading.Lock()
# code -> {'catalog': the TranslationCatalog the layer lives in,
#          'layer': the override dict, 'version': the version applied}
_state = {}


def overrides_version():
    """The shared overrides version, or None when no cache is available."""
    try:
        version = cache.get(VERSION_KEY)
        if version is None:
            # First use or the cache was emptied: start a fresh value no
            # process has applied yet, so every process reloads once.
            cache.add(VERSION_KEY, time.time_ns(), timeout=None)
            version = cache.get(VERSION_KEY)
    except Exception:  # cache server unreachable: reload on every call
        return None
    return version


def bump_overrides_version():
    try:
        cache.set(VERSION_KEY, time.time_ns(), timeout=None)
    except Exception:
        pass


def catalog_key(msgid, msgctxt=''):
    return f'{msgctxt}{CONTEXT_SEPARATOR}{msgid}' if msgctxt else msgid


def _project_plural(code, translation):
    """The plural function of this project's catalog for ``code``.

    Plural indexes in the .po (and so in overrides) follow its Plural-Forms
    header, which can differ from Django's own catalog for the language.
    """
    locale = trans_real.to_locale(code)
    for localedir in settings.LOCALE_PATHS:
        project = gettext_module.translation(
            domain='django', localedir=str(localedir), languages=[locale], fallback=True)
        plural = getattr(project, 'plural', None)
        if plural is not None:
            return plural
    return translation.plural


def _layer_for(code):
    """The override dict at the front of ``code``'s catalog (created if needed)."""
    translation = trans_real.translation(code)
    catalog = translation._catalog
    state = _state.get(code)
    if state is not None and state['catalog'] is catalog:
        return state
    catalogs = getattr(catalog, '_catalogs', None)
    plurals = getattr(catalog, '_plurals', None)
    if catalogs is None or plurals is None:
        return None  # not a TranslationCatalog (Django < 3.2): no overrides
    layer = {}
    catalogs.insert(0, layer)
    plurals.insert(0, _project_plural(code, translation))
    state = {'catalog': catalog, 'layer': layer, 'version': None}
    _state[code] = state
    return state


def _wanted(code):
    from .models import TranslationOverride

    rows = list(TranslationOverride.objects.filter(language=code).exclude(msgstr='').values_list(
        'msgctxt', 'msgid', 'plural_index', 'msgstr'))
    # Plural entries are saved with every form (see views.save_translation),
    # so an entry with a form above 0 is a plural one.
    plural_keys = {(msgctxt, msgid) for msgctxt, msgid, index, _ in rows if index}
    wanted = {}
    for msgctxt, msgid, plural_index, msgstr in rows:
        key = catalog_key(msgid, msgctxt)
        if plural_index or (msgctxt, msgid) in plural_keys:
            wanted[(key, plural_index)] = msgstr
        else:
            wanted[key] = msgstr
    return wanted


def apply_overrides(language, version=None):
    """Load ``language``'s overrides from the database into its catalog."""
    code = (language or '').split('-')[0]
    if not code or code == settings.LANGUAGE_CODE:
        return
    from .languages import supported_codes
    if code not in supported_codes():
        return
    with _lock:
        state = _layer_for(code)
        if state is None:
            return
        wanted = _wanted(code)
        layer = state['layer']
        for key in [k for k in layer if k not in wanted]:
            layer.pop(key, None)
        layer.update(wanted)
        state['version'] = version


def ensure_overrides(language):
    """Apply ``language``'s overrides unless this process already has the current ones."""
    code = (language or '').split('-')[0]
    if not code or code == settings.LANGUAGE_CODE:
        return
    version = overrides_version()
    state = _state.get(code)
    if (version is not None and state is not None and state['version'] == version
            and state['catalog'] is getattr(trans_real.translation(code), '_catalog', None)):
        return
    apply_overrides(code, version=version)


def reset_overrides():
    """Remove every override layer from this process's catalogs (used by tests)."""
    with _lock:
        for state in _state.values():
            catalog = state['catalog']
            for index, layer in enumerate(catalog._catalogs):
                if layer is state['layer']:
                    del catalog._catalogs[index]
                    del catalog._plurals[index]
                    break
        _state.clear()
