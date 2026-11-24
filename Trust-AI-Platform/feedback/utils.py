from .languages import LANGUAGE_CODES, language_code
from .models import FeedbackForm, FeedbackResponse


def get_applicable_form(scenario, audience):
    """Newest active form for this audience that applies to the scenario, or None."""
    for form in FeedbackForm.objects.filter(audience=audience, is_active=True):
        if form.applies_to(scenario):
            return form
    return None


def user_has_responded(form, user, scenario):
    return FeedbackResponse.objects.filter(form=form, user=user, scenario=scenario).exists()


def _has_translation(translations, language):
    entry = (translations or {}).get(language)
    if not isinstance(entry, dict):
        return False
    for value in entry.values():
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, list) and any(isinstance(v, str) and v.strip() for v in value):
            return True
    return False


def display_language(form, scenario):
    """Code of the version this scenario's users see: the scenario's language
    when the form has a translation for it, otherwise the form's own language."""
    code = language_code(scenario.language)
    if code in LANGUAGE_CODES and code != form.source_language:
        if _has_translation(form.translations, code) or any(
            _has_translation(q.translations, code) for q in form.questions.all()
        ):
            return code
    return form.source_language


def _pick(translated, original):
    """The translation when it has text, otherwise the original."""
    if isinstance(translated, str) and translated.strip():
        return translated.strip()
    return original


def _entry(translations, language):
    entry = (translations or {}).get(language) if language else None
    return entry if isinstance(entry, dict) else {}


def serialize_form(form, language=None):
    """The form as the popup shows it, translated field by field into
    `language` where a translation exists."""
    if language not in LANGUAGE_CODES or language == form.source_language:
        language = form.source_language
        form_tr, translate = {}, False
    else:
        form_tr, translate = _entry(form.translations, language), True
    questions = []
    for q in form.questions.all():
        q_tr = _entry(q.translations, language) if translate else {}
        tr_options = q_tr.get('options') if isinstance(q_tr.get('options'), list) else []
        options = [
            _pick(tr_options[i] if i < len(tr_options) else None, original)
            for i, original in enumerate(q.options)
        ]
        questions.append({
            'id': q.id,
            'text': _pick(q_tr.get('text'), q.text),
            'type': q.question_type,
            'options': options,
            'required': q.is_required,
        })
    return {
        'id': form.id,
        'language': language,
        'title': _pick(form_tr.get('title'), form.title),
        'description': _pick(form_tr.get('description'), form.description),
        'questions': questions,
    }
