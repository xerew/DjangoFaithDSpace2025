"""A small scenario as export_scenario_bank writes it.

P1 (page) -> Q1. Q1 right -> Q2, wrong -> P2 (remedial page) -> Q2.
Q2 evaluates Q1+Q2: both right -> H (limit 1.0), one -> M (limit 0.5), none -> L.
"""
import copy

P1, Q1, P2, Q2, H, M, L = 1, 2, 3, 4, 5, 6, 7


def page(activity_id, text, next_id=None, images=()):
    return {'id': activity_id, 'kind': 'page', 'type': 'Explanation', 'phase': 'Explore',
            'text': text, 'images': list(images), 'missing_images': 0, 'interactive': False,
            'options': [], 'next': next_id, 'evaluation': None}


def question(activity_id, text, options, evaluation=None, images=()):
    return {'id': activity_id, 'kind': 'question', 'type': 'Question', 'phase': 'Explore',
            'text': text, 'images': list(images), 'missing_images': 0, 'interactive': False,
            'options': [
                {'id': option_id, 'text': label, 'is_correct': correct, 'weight': int(correct),
                 'image': None, 'next': next_id}
                for option_id, label, correct, next_id in options
            ],
            'next': None, 'evaluation': evaluation}


def scenario():
    nodes = [
        page(P1, 'Watch the pendulum swing.', Q1),
        question(Q1, 'Which is faster?', [
            (21, 'Short one', True, Q2), (22, 'Heavy one', False, P2), (23, 'Same', False, P2),
        ]),
        page(P2, 'A shorter pendulum swings faster.', Q2),
        question(Q2, 'Does mass matter?', [
            (41, 'No', True, None), (42, 'Yes', False, None), (43, 'A little', False, None),
        ], evaluation={
            'bunch': [Q1, Q2],
            'high': {'to': H, 'limit': 1.0},
            'mid': {'to': M, 'limit': 0.5},
            'low': {'to': L},
        }),
        page(H, 'Well done.'),
        page(M, 'Almost.'),
        page(L, 'Let us try again.'),
    ]
    return copy.deepcopy({
        'scenario_id': 9, 'scenario': 'Pendulum', 'language': 'English',
        'age_min': 12, 'age_max': 15, 'learning_goals': 'Period of a pendulum',
        'start': P1,
        'activities': {str(node['id']): node for node in nodes},
        'real': {
            'students': 25, 'runs_used': 20, 'follows_routes': 0.8, 'completed': 18,
            'transitions': {
                str(P1): {str(Q1): 20},
                str(Q1): {str(Q2): 10, str(P2): 8, 'STOP': 2},
                str(P2): {str(Q2): 8},
                str(Q2): {str(H): 9, str(M): 6, str(L): 3},
                str(H): {'END': 9}, str(M): {'END': 6}, str(L): {'END': 3},
            },
            'branching': {str(Q2): {'high': 9, 'mid': 6, 'low': 3}},
            'items': {
                str(Q1): {'n': 20, 'p_correct': 0.6, 'option_shares': {}, 'point_biserial': 0.5},
                str(Q2): {'n': 18, 'p_correct': 0.75, 'option_shares': {}, 'point_biserial': 0.4},
            },
            'score_histogram': {'2': 9, '1': 6, '0': 3},
            'path_length_histogram': {'4': 10, '5': 8},
            'paths': [{'path': [P1, Q1, Q2, H], 'n': 9}, {'path': [P1, Q1, P2, Q2, M], 'n': 5}],
        },
    })
