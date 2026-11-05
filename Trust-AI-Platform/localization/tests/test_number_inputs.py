"""Number inputs need a dot: a browser drops the value "0,75" of type="number"."""
from django.contrib.auth.models import Group, User
from django.test import TestCase, override_settings
from django.urls import reverse

from accounts.models import UserProfile
from authoringtool.models import Activity, EvQuestionBranching, Phase, Scenario


@override_settings(SCENARIO_REVISION_PROTECTION=False)
class ScoreLimitInputTests(TestCase):
    def test_score_limits_use_a_dot_in_a_comma_language(self):
        teacher = User.objects.create_user('criterion_teacher')
        teacher.groups.add(Group.objects.get_or_create(name='teachers')[0])
        UserProfile.objects.create(user=teacher, language='el')
        scenario = Scenario.objects.create(name='Criterion', visibility_status='private',
                                           created_by=teacher, updated_by=teacher)
        phase = Phase.objects.create(name='Phase', scenario=scenario,
                                     created_by=teacher, updated_by=teacher)

        def activity(name, score_limit=0.0):
            return Activity.objects.create(name=name, text='<p>x</p>', phase=phase, scenario=scenario,
                                           score_limit=score_limit,
                                           created_by=teacher, updated_by=teacher)

        primary = activity('Primary')
        EvQuestionBranching.objects.create(
            activity=primary,
            next_question_on_high=activity('High', 2.75),
            next_question_on_mid=activity('Mid', 1.5),
            next_question_on_low=activity('Low', 0.25),
        )
        self.client.force_login(teacher)

        response = self.client.get(reverse('updateCriterion', args=[scenario.id, phase.id, primary.id]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['LANGUAGE_CODE'], 'el')
        self.assertContains(response, 'name="high_score_limit" value="2.75"')
        self.assertContains(response, 'name="mid_score_limit" value="1.5"')
        self.assertContains(response, 'name="low_score_limit" value="0.25"')
