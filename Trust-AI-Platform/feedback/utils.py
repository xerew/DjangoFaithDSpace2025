from .models import FeedbackForm, FeedbackResponse


def get_applicable_form(scenario, audience):
    """Newest active form for this audience that applies to the scenario, or None."""
    for form in FeedbackForm.objects.filter(audience=audience, is_active=True):
        if form.applies_to(scenario):
            return form
    return None


def user_has_responded(form, user, scenario):
    return FeedbackResponse.objects.filter(form=form, user=user, scenario=scenario).exists()


def serialize_form(form):
    return {
        'id': form.id,
        'title': form.title,
        'description': form.description,
        'questions': [
            {
                'id': q.id,
                'text': q.text,
                'type': q.question_type,
                'options': q.options,
                'required': q.is_required,
            }
            for q in form.questions.all()
        ],
    }
