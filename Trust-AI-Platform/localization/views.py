from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db import transaction
from django.http import Http404, HttpResponseBadRequest, HttpResponseForbidden, JsonResponse
from django.shortcuts import redirect, render
from django.utils.http import url_has_allowed_host_and_scheme
from django.utils.translation import gettext as _
from django.views.decorators.http import require_GET, require_POST

from . import catalog
from .languages import meta_for, supported_codes
from .models import TranslationOverride, TranslationOverrideLog
from .overrides import bump_overrides_version, ensure_overrides
from .reviewers import REVIEW_CODES, REVIEW_LANGUAGES, reviewable_codes

REVIEW_PAGE_SIZE = 100


@require_POST
def switch_language(request):
    code = request.POST.get('language', '')
    if code not in supported_codes():
        return HttpResponseBadRequest('Unknown language')

    target = request.POST.get('next', '')
    if not url_has_allowed_host_and_scheme(
            target, allowed_hosts={request.get_host()},
            require_https=request.is_secure()):
        target = '/'
    response = redirect(target)
    response.set_cookie(
        settings.LANGUAGE_COOKIE_NAME, code,
        max_age=settings.LANGUAGE_COOKIE_AGE,
        path=settings.LANGUAGE_COOKIE_PATH,
        domain=settings.LANGUAGE_COOKIE_DOMAIN,
        secure=settings.LANGUAGE_COOKIE_SECURE or None,
        httponly=settings.LANGUAGE_COOKIE_HTTPONLY,
        samesite=settings.LANGUAGE_COOKIE_SAMESITE,
    )

    if request.user.is_authenticated:
        from accounts.models import UserProfile
        profile, _created = UserProfile.objects.get_or_create(user=request.user)
        profile.language = code
        profile.save(update_fields=['language'])
    return response


# ── Reviewer corrections ────────────────────────────────────────────────────

def _review_access(request, code):
    """Return None when ``request.user`` may review ``code``, else the error response."""
    if code not in REVIEW_CODES:
        raise Http404
    if code not in reviewable_codes(request.user):
        return HttpResponseForbidden(_('You are not a translation reviewer for this language.'))
    return None


def _forms(entry, overrides):
    """One dict per translation form of a .po entry (one, or one per plural form)."""
    context = entry.msgctxt or ''
    if entry.msgid_plural:
        pairs = [
            (int(index), entry.msgid if int(index) == 0 else entry.msgid_plural, text)
            for index, text in sorted(entry.msgstr_plural.items(), key=lambda item: int(item[0]))
        ]
    else:
        pairs = [(0, entry.msgid, entry.msgstr)]
    forms = []
    for index, source, po_text in pairs:
        override = overrides.get((context, entry.msgid, index))
        forms.append({
            'index': index,
            'source': source,
            'po': po_text,
            'override': override,
            'current': override if override is not None else po_text,
        })
    return forms


def _rows(code):
    overrides = {
        (o.msgctxt, o.msgid, o.plural_index): o.msgstr
        for o in TranslationOverride.objects.filter(language=code)
    }
    needs_review = {(e.msgctxt or '', e.msgid) for e in catalog.untranslated(code)}
    rows = []
    for entry in catalog.load_po(code):
        if entry.obsolete or catalog.is_terms_only(entry):
            continue
        forms = _forms(entry, overrides)
        corrected = all(f['override'] is not None for f in forms)
        rows.append({
            'msgid': entry.msgid,
            'msgid_plural': entry.msgid_plural,
            'msgctxt': entry.msgctxt or '',
            'forms': forms,
            'corrected': any(f['override'] is not None for f in forms),
            # A reviewer's correction of every form settles an untranslated entry.
            'needs_review': (entry.msgctxt or '', entry.msgid) in needs_review and not corrected,
        })
    return rows


def _matches(row, query):
    texts = [row['msgid'], row['msgid_plural'], row['msgctxt']]
    for form in row['forms']:
        texts += [form['po'], form['current']]
    return any(query in (text or '').casefold() for text in texts)


@login_required
@require_GET
def review_index(request):
    codes = reviewable_codes(request.user)
    if not codes:
        return HttpResponseForbidden(_('You are not a translation reviewer.'))
    return redirect('review_translations', code=codes[0])


@login_required
@require_GET
def review_translations(request, code):
    denied = _review_access(request, code)
    if denied:
        return denied
    query = request.GET.get('q', '').strip()
    show = 'needs_review' if request.GET.get('filter') == 'needs_review' else 'all'
    rows = _rows(code)
    if show == 'needs_review':
        rows = [row for row in rows if row['needs_review']]
    if query:
        folded = query.casefold()
        rows = [row for row in rows if _matches(row, folded)]
    page_obj = Paginator(rows, REVIEW_PAGE_SIZE).get_page(request.GET.get('page'))
    codes = set(reviewable_codes(request.user))
    return render(request, 'localization/review.html', {
        'language': meta_for(code),
        'languages': [m for m in REVIEW_LANGUAGES if m['code'] in codes],
        'page_obj': page_obj,
        'query': query,
        'show': show,
    })


def _problem_text(problem):
    fixed = {
        catalog.UNBALANCED_BRACKETS: _(
            'The angle brackets do not match the source text. Check for an unclosed HTML tag.'),
        catalog.EXTRA_DOUBLE_QUOTE: _(
            'The translation has more straight double quotes than the source text. '
            'Use typographic quotes such as « » or „ “ instead.'),
        catalog.STRAY_PERCENT: _(
            'A percent sign that is not part of a placeholder would break this text. '
            'Write two percent signs for a literal one.'),
        catalog.UNEXPECTED_LINE_BREAK: _(
            'The source text has no line break, so the translation cannot have one.'),
        catalog.UNSAFE_BRACES: _(
            'The curly braces do not match the source text. Keep every placeholder exactly '
            'as it is and do not add other curly braces.'),
    }
    if problem in fixed:
        return fixed[problem]
    kind, _sep, token = problem.partition(' ')
    if kind == 'missing':
        return _('Missing from your translation: %(token)s') % {'token': token}
    return _('Not in the source text: %(token)s') % {'token': token}


def _find_entry(code, msgctxt, msgid):
    for entry in catalog.load_po(code):
        if (not entry.obsolete and entry.msgid == msgid and (entry.msgctxt or '') == msgctxt
                and not catalog.is_terms_only(entry)):
            return entry
    return None


def _posted_text(request, name):
    value = request.POST.get(name, '').replace('\r\n', '\n')
    return value if value.strip() else ''


@login_required
@require_POST
def save_translation(request, code):
    denied = _review_access(request, code)
    if denied:
        return denied
    msgid = request.POST.get('msgid', '').replace('\r\n', '\n')
    msgctxt = request.POST.get('msgctxt', '')
    entry = _find_entry(code, msgctxt, msgid)
    if entry is None:
        return JsonResponse({
            'success': False, 'problems': [],
            'error': _('This text is not in the interface catalog.'),
        }, status=400)

    forms = _forms(entry, {})
    if entry.msgid_plural:
        values = {f['index']: _posted_text(request, f'msgstr_{f["index"]}') for f in forms}
        if any(values.values()) and not all(values.values()):
            return JsonResponse({
                'success': False, 'problems': [],
                'error': _('Fill in every plural form, or leave them all empty to use the '
                           'translation from the catalog.'),
            }, status=400)
    else:
        values = {0: _posted_text(request, 'msgstr')}

    problems = []
    for form in forms:
        if values[form['index']]:
            problems += catalog.translation_problems(form['source'], values[form['index']])
    if problems:
        return JsonResponse({
            'success': False,
            'problems': problems,
            'messages': [_problem_text(p) for p in problems],
            'error': _('The translation was not saved:'),
        }, status=400)

    fuzzy = 'fuzzy' in entry.flags
    with transaction.atomic():
        existing = {
            o.plural_index: o
            for o in TranslationOverride.objects.select_for_update().filter(
                language=code, msgctxt=msgctxt, msgid=msgid)
        }
        for form in forms:
            index, value, po_text = form['index'], values[form['index']], form['po']
            override = existing.get(index)
            old = override.msgstr if override else po_text
            # An override equal to a live catalog translation adds nothing.
            keep = bool(value) and (value != po_text or fuzzy)
            if keep:
                if override is None:
                    override = TranslationOverride.objects.create(
                        language=code, msgctxt=msgctxt, msgid=msgid, plural_index=index,
                        msgstr=value, updated_by=request.user)
                elif override.msgstr != value:
                    override.msgstr = value
                    override.updated_by = request.user
                    override.save(update_fields=['msgstr', 'updated_by', 'updated_at'])
                new = value
            else:
                if override is not None:
                    override.delete()
                override, new = None, po_text
            if new != old:
                TranslationOverrideLog.objects.create(
                    override=override, language=code, msgctxt=msgctxt, msgid=msgid,
                    plural_index=index, old=old, new=new, user=request.user)
    bump_overrides_version()
    ensure_overrides(code)

    saved = _forms(entry, {
        (o.msgctxt, o.msgid, o.plural_index): o.msgstr
        for o in TranslationOverride.objects.filter(language=code, msgctxt=msgctxt, msgid=msgid)
    })
    return JsonResponse({
        'success': True,
        'message': _('Translation saved.'),
        'forms': [{'index': f['index'], 'current': f['current'],
                   'corrected': f['override'] is not None} for f in saved],
    })
