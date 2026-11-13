"""Excel downloads of a dashboard tab, with the same filters applied."""

from io import BytesIO

from django.utils.translation import gettext as _
from openpyxl import Workbook
from openpyxl.styles import Font

from authoringtool.performance_formula import CATEGORIES

from . import chatbot_report
from .report import early_view, filter_options, final_view, over_time_view

PERCENT = '0.0%'


def _sheet(book, title, header, rows, percent_columns=()):
    sheet = book.create_sheet(title)
    sheet.append(header)
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for row in rows:
        sheet.append(list(row))
    for column in percent_columns:
        for cell in sheet.iter_cols(min_col=column, max_col=column, min_row=2):
            for item in cell:
                item.number_format = PERCENT
    for index, column in enumerate(sheet.columns, start=1):
        width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column)
        sheet.column_dimensions[column[0].column_letter].width = min(max(width + 2, 10), 60)
    return sheet


def _filter_rows(results, filters):
    names = dict(filter_options(results)['scenarios'])
    scenario = names.get(int(filters['scenario'])) if filters['scenario'].isdigit() else None
    version = {'current': _('Current version'), 'older': _('Older or legacy versions')}
    return [
        (_('Scenario filter'), scenario or _('All scenarios')),
        (_('Language filter'), filters['language'] or _('All languages')),
        (_('Version filter'), version.get(filters['version'], _('All versions'))),
    ]


def _confusion_rows(matrix, row_label):
    rows = []
    for label, counts in zip(CATEGORIES, matrix or [[0] * 3] * 3):
        rows.append([label, *counts])
    return [row_label, *CATEGORIES], rows


def _final_book(run, filters):
    view = final_view(run.results, filters)
    overall = view['overall']
    book = Workbook()
    book.remove(book.active)
    summary = [
        (_('Training run finished'), run.finished_at.strftime('%Y-%m-%d %H:%M') if run.finished_at else ''),
        *_filter_rows(run.results, filters),
        (_('Runs'), overall['runs']),
        (_('Agreement with PISA'), overall['accuracy']),
        (_('Macro F1'), overall['macro_f1']),
        *((_('F1 %(category)s') % {'category': label}, overall['per_class'][label]['f1'])
          for label in reversed(CATEGORIES)),
    ]
    sheet = _sheet(book, 'Summary', [_('Measure'), _('Value')], summary)
    for row in sheet.iter_rows(min_row=2):
        if isinstance(row[1].value, float):
            row[1].number_format = PERCENT
    header, rows = _confusion_rows(view['confusion'], _('PISA \\ Model'))
    _sheet(book, 'Confusion matrix', header, rows)
    _sheet(book, 'Disagreements', [
        _('Student'), _('Scenario'), _('Language'), _('PISA'), _('Model'), _('Confidence'),
        _('PISA final %'), _('Distance to nearest threshold'),
    ], [
        (row['username'], row['scenario_name'], row['language'], row['pisa'], row['model'],
         row['confidence'], row['final_percentage'], row['threshold_distance'])
        for row in view['disagreements']
    ], percent_columns=(6,))
    _sheet(book, 'Importance', [_('Input'), _('Mean absolute SHAP')],
           [(row['input'], row['mean_abs_shap']) for row in view['importances']])
    _sheet(book, 'Per scenario', [_('Scenario'), _('Runs'), _('Agreement'), _('Unseen scenario')],
           [(row['names'], row['runs'], row['agreement'], row['loso']) for row in view['per_scenario']],
           percent_columns=(3, 4))
    return book


def _early_book(run, filters, checkpoint):
    view = early_view(run.results, filters, checkpoint)
    book = Workbook()
    book.remove(book.active)
    sheet = book.create_sheet('Checkpoints')
    for row in _filter_rows(run.results, filters):
        sheet.append(row)
    sheet.append([
        _('Checkpoint'), _('Runs'), _('XGBoost accuracy'), _('Formula so far accuracy'),
        _('XGBoost macro F1'), _('Formula so far macro F1'),
        _('XGBoost F1 High'), _('XGBoost F1 Moderate'), _('XGBoost F1 Low'),
        _('Formula F1 High'), _('Formula F1 Moderate'), _('Formula F1 Low'),
    ])
    header_row = sheet.max_row
    for cell in sheet[header_row]:
        cell.font = Font(bold=True)
    for row in view['rows']:
        xgb, base = row['xgb'], row['baseline']
        sheet.append([
            _('After phase %(k)s') % {'k': row['checkpoint']}, xgb['runs'],
            xgb['accuracy'], base['accuracy'], xgb['macro_f1'], base['macro_f1'],
            *(xgb['per_class'][label]['f1'] for label in reversed(CATEGORIES)),
            *(base['per_class'][label]['f1'] for label in reversed(CATEGORIES)),
        ])
        for cell in sheet[sheet.max_row][2:]:
            cell.number_format = PERCENT
    header, rows = _confusion_rows(view['confusion'], _('Final \\ Predicted'))
    _sheet(book, 'Confusion matrix', header, rows)
    _sheet(book, 'Importance', [_('Input'), _('Mean absolute SHAP')],
           [(row['input'], row['mean_abs_shap']) for row in view['importances']])
    _sheet(book, 'Unseen scenarios',
           [_('Scenario'), _('Runs'), _('XGBoost accuracy'), _('Formula so far accuracy')],
           [(row['scenario_names'], row['runs'], row['accuracy'], row['baseline_accuracy'])
            for row in view['loso']],
           percent_columns=(3, 4))
    return book


def _text_only(book):
    """Store text that looks like a formula ("=...") as text.

    The export never writes formulas; scenario names are written by
    teachers, the language filter comes from the URL, and chatbot
    sentences are typed by people.
    """
    for sheet in book.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                if cell.data_type == 'f':
                    cell.data_type = 's'


def _save(book):
    _text_only(book)
    buffer = BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def _time_book(run):
    """Every evaluated period, for every prediction, in one long sheet."""
    book = Workbook()
    book.remove(book.active)
    first = over_time_view(run.results)
    rows = []
    for option in (first['options'] if first else []):
        name = _('Final categorization') if option == 'final' else _('After phase %(k)s') % {'k': option}
        for row in over_time_view(run.results, option)['rows']:
            if not row['evaluated']:
                continue
            xgb, base = row['xgb'], row['baseline'] or {}
            rows.append((row['label'], name, row['trained_on'], row['students_trained'], row['tested'],
                         xgb['accuracy'], base.get('accuracy'), xgb['macro_f1'], base.get('macro_f1')))
    _sheet(book, 'Over time', [
        _('Tested on'), _('Prediction'), _('Runs learned from'), _('Students learned from'),
        _('Runs tested'), _('XGBoost accuracy'), _('Formula so far accuracy'),
        _('XGBoost macro F1'), _('Formula so far macro F1'),
    ], rows, percent_columns=(6, 7, 8, 9))
    return book


def workbook(run, tab, filters, checkpoint=None):
    if tab == 'time':
        book = _time_book(run)
    elif tab == 'final':
        book = _final_book(run, filters)
    else:
        book = _early_book(run, filters, checkpoint)
    return _save(book)


def chatbot_workbook(evaluation):
    """The sheets of the hand-run intent report, for one chatbot evaluation."""
    view = chatbot_report.build(evaluation)
    summary = view['summary']
    book = Workbook()
    book.remove(book.active)
    rows = [
        (_('Evaluation finished'), evaluation.finished_at.strftime('%Y-%m-%d %H:%M') if evaluation.finished_at else ''),
        (_('Folds'), summary['folds']),
        (_('Rasa version'), evaluation.rasa_version),
        (_('Training sentences (SHA-256)'), evaluation.nlu_sha256),
        (_('Examples tested'), summary['examples']),
        (_('Intents tested'), summary['intents']),
        (_('Accuracy (all folds pooled)'), summary['accuracy']),
        (_('Macro F1 (each intent counts equally)'), summary['macro_f1']),
        (_('Weighted F1 (by number of examples)'), summary['weighted_f1']),
        (_('Wrong predictions'), summary['wrong']),
        (_('Wrong with confidence at or above the fallback %(threshold)s (the fallback would not catch them)')
         % {'threshold': summary['fallback']}, summary['wrong_above_fallback']),
    ]
    for key, (mean, std) in summary['per_fold'].items():
        rows.append((_('%(measure)s per fold, mean (std)') % {'measure': key}, f'{mean:.3f} ({std:.3f})'))
    sheet = _sheet(book, 'Summary', [_('Measure'), _('Value')], rows)
    for row in sheet.iter_rows(min_row=2):
        if isinstance(row[1].value, float):
            row[1].number_format = PERCENT
    _sheet(book, 'Per intent',
           [_('Intent'), _('Examples'), _('Precision'), _('Recall'), _('F1'), _('Wrong')],
           [(r['intent'], r['examples'], r['precision'], r['recall'], r['f1'], r['wrong'])
            for r in sorted(view['per_intent'], key=lambda r: r['intent'])],
           percent_columns=(3, 4, 5))
    confusion = view['confusion']
    _sheet(book, 'Confusion matrix', [_('True \\ Predicted'), *confusion['columns']],
           [(true, *counts) for true, counts in zip(confusion['rows'], confusion['matrix'])])
    header = [_('Text'), _('True intent'), _('Predicted intent'), _('Confidence'), _('Correct'),
              _('Below fallback')]

    def prediction_rows(rows):
        return [(r['text'], r['intent'], r['predicted'], r['confidence'], r['correct'], r['below_fallback'])
                for r in rows]

    _sheet(book, 'Wrong predictions', header, prediction_rows(view['wrong']))
    _sheet(book, 'All predictions', header, prediction_rows(view['predictions']))
    return _save(book)
