"""The interface languages of the platform, in display order."""

LANGUAGE_META = [
    {'code': 'en', 'native': 'English', 'table_name': 'English', 'flag': 'gb'},
    {'code': 'eu', 'native': 'Euskara', 'table_name': 'Basque', 'flag': 'es-pv'},
    {'code': 'ca', 'native': 'Català', 'table_name': 'Catalan', 'flag': 'es-ct'},
    {'code': 'et', 'native': 'Eesti', 'table_name': 'Estonian', 'flag': 'ee'},
    {'code': 'fr', 'native': 'Français', 'table_name': 'French', 'flag': 'fr'},
    {'code': 'de', 'native': 'Deutsch', 'table_name': 'German', 'flag': 'de'},
    {'code': 'el', 'native': 'Ελληνικά', 'table_name': 'Greek', 'flag': 'gr'},
    {'code': 'it', 'native': 'Italiano', 'table_name': 'Italian', 'flag': 'it'},
    {'code': 'pt', 'native': 'Português', 'table_name': 'Portuguese', 'flag': 'pt'},
    {'code': 'ro', 'native': 'Română', 'table_name': 'Romanian', 'flag': 'ro'},
    {'code': 'es', 'native': 'Español', 'table_name': 'Spanish', 'flag': 'es'},
    {'code': 'tr', 'native': 'Türkçe', 'table_name': 'Turkish', 'flag': 'tr'},
    {'code': 'uk', 'native': 'Українська', 'table_name': 'Ukrainian', 'flag': 'ua'},
]


def supported_codes():
    return {m['code'] for m in LANGUAGE_META}


def code_for_table_name(name):
    """Map a language name (table name, native name or code) to a code."""
    if not name:
        return None
    key = str(name).strip().casefold()
    if not key:
        return None
    for meta in LANGUAGE_META:
        if key in (meta['code'].casefold(), meta['table_name'].casefold(),
                   meta['native'].casefold()):
            return meta['code']
    return None


def meta_for(code):
    for meta in LANGUAGE_META:
        if meta['code'] == code:
            return meta
    return LANGUAGE_META[0]
