import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.datavalidation import DataValidation

_HEADER_FONT = Font(bold=True, color='FFFFFF')
_HEADER_FILL = PatternFill('solid', fgColor='1D4ED8')
_REQ_FILL = PatternFill('solid', fgColor='DC2626')
_BOOL_DV = DataValidation(type='list', formula1='"Yes,No"', showDropDown=False, showErrorMessage=True)


def _write_sheet(wb, title, columns, bool_cols=(), example_rows=()):
    """columns: list of (name, required_bool)"""
    ws = wb.create_sheet(title)
    for col_idx, (name, required) in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=col_idx, value=f'{name}*' if required else name)
        cell.font = _HEADER_FONT
        cell.fill = _REQ_FILL if required else _HEADER_FILL
        cell.alignment = Alignment(horizontal='center')
        ws.column_dimensions[cell.column_letter].width = max(len(name) + 4, 16)

    col_name_to_idx = {name: i + 1 for i, (name, _) in enumerate(columns)}
    dv = DataValidation(type='list', formula1='"Yes,No"', showDropDown=False)
    ws.add_data_validation(dv)
    for bool_col in bool_cols:
        if bool_col in col_name_to_idx:
            idx = col_name_to_idx[bool_col]
            letter = ws.cell(row=1, column=idx).column_letter
            dv.sqref = f'{letter}2:{letter}200'

    for row_data in example_rows:
        ws.append(row_data)

    return ws


def _write_readme(ws):
    ws['A1'] = 'SCENARIO TEMPLATE — INSTRUCTIONS'
    ws['A1'].font = Font(bold=True, size=14)
    lines = [
        '',
        'SHEETS AND THEIR PURPOSE',
        '  Scenario  — One row of scenario metadata (name, description, etc.)',
        '  Phases    — One row per phase; row order = phase order in the scenario',
        '  Activities — One row per activity; row order = activity order within each phase',
        '  Answers   — One row per answer (multiple rows per activity are fine)',
        '  Next Activity — One row per routing rule (which activity comes after which)',
        '  Evaluation — One row per evaluatable activity (scoring bunches + score branches)',
        '',
        'REFERENCING RULES',
        '  Activities reference Phases by Phase Name — must match exactly.',
        '  Answers, Next Activity, and Evaluation reference Activities by Activity Name.',
        '  Activity names must be unique within the file.',
        '  TIP: Copy-paste names from the Activities sheet to avoid typos.',
        '',
        'TEXT FIELDS',
        '  Activity Text and Answer Text support Markdown formatting:',
        '    **bold**   _italic_   # Heading   - bullet list   [Link](https://...)',
        '  Markdown is converted to HTML automatically on import.',
        '',
        'BOOLEAN FIELDS (Is Evaluatable, Is Correct, etc.)',
        '  Use the dropdown: Yes or No (case-insensitive).',
        '',
        'NEXT ACTIVITY SHEET',
        '  Leave "Answer Text" blank for an unconditional (default) next activity.',
        '  Leave "Next Activity Name" blank to end the scenario at that activity.',
        '  Example:',
        '    Source Activity | Answer Text | Next Activity',
        '    Welcome         |             | Quiz 1         (default route)',
        '    Quiz 1          | Correct     | Well Done      (per-answer route)',
        '    Quiz 1          | Wrong       | Try Again',
        '',
        'EVALUATION SHEET',
        '  Primary Activity Name: an activity with Is Evaluatable = Yes.',
        '  Grouped Activities: comma-separated activity names that form the scoring group.',
        '    Include the primary activity itself in this list.',
        '  High/Mid/Low Branch Activity: where students go based on their score.',
        '',
        'COLUMNS MARKED WITH *',
        '  Red headers are required. Leave others blank if not needed.',
    ]
    for i, line in enumerate(lines, start=2):
        ws.cell(row=i, column=1, value=line)
    ws.column_dimensions['A'].width = 80


def generate_blank_template():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'README'
    _write_readme(ws)

    _write_sheet(wb, 'Scenario', [
        ('Name', True), ('Description', False), ('Learning Goals', False),
        ('Language', False), ('Subject Domains', False), ('Age Min', False),
        ('Age Max', False), ('Suggested Time (min)', False), ('Video URL', False),
        ('Visibility', False),
    ], example_rows=[
        ['My Scenario', 'A brief description', 'Students will learn...', 'English',
         'Physics,STEM', '14', '18', '90', '', 'private'],
    ])

    _write_sheet(wb, 'Phases', [
        ('Phase Name', True), ('Description', False), ('Video URL', False),
    ], example_rows=[
        ['Introduction', 'Students explore the topic', ''],
        ['Experiment', '', ''],
        ['Evaluation', '', ''],
    ])

    _write_sheet(wb, 'Activities', [
        ('Activity Name', True), ('Phase Name', True), ('Text', True),
        ('Activity Type', False), ('Helper', False), ('Is Evaluatable', False),
        ('Is Primary Evaluation', False), ('Must Wait', False), ('Score Limit', False),
        ('Simulation Name', False), ('Remote Lab Name', False), ('VR Lab Name', False),
        ('Image URL', False), ('Video URL', False),
    ], bool_cols=['Is Evaluatable', 'Is Primary Evaluation', 'Must Wait'], example_rows=[
        ['Welcome', 'Introduction', 'Welcome to this scenario! **Read carefully.**',
         '', '', 'No', 'No', 'No', '', '', '', '', '', ''],
        ['Quiz 1', 'Evaluation', 'What is the boiling point of water?',
         'Question', '', 'Yes', 'Yes', 'No', '', '', '', '', '', ''],
    ])

    _write_sheet(wb, 'Answers', [
        ('Activity Name', True), ('Answer Text', True), ('Is Correct', False),
        ('Answer Weight', False), ('Image URL', False), ('Video URL', False),
        ('Feedback Text', False), ('Feedback Image URL', False), ('Feedback Video URL', False),
    ], bool_cols=['Is Correct'], example_rows=[
        ['Quiz 1', '100°C', 'Yes', '1', '', '', 'Correct! Water boils at 100°C.', '', ''],
        ['Quiz 1', '50°C', 'No', '0', '', '', 'Not quite. Try again!', '', ''],
    ])

    _write_sheet(wb, 'Next Activity', [
        ('Source Activity Name', True), ('Answer Text', False), ('Next Activity Name', False),
    ], example_rows=[
        ['Welcome', '', 'Quiz 1'],
        ['Quiz 1', '100°C', 'Well Done'],
        ['Quiz 1', '50°C', 'Try Again'],
    ])

    _write_sheet(wb, 'Evaluation', [
        ('Primary Activity Name', True), ('Grouped Activities', True),
        ('High Branch Activity', False), ('High Branch Feedback', False),
        ('Mid Branch Activity', False), ('Mid Branch Feedback', False),
        ('Low Branch Activity', False), ('Low Branch Feedback', False),
    ], example_rows=[
        ['Quiz 1', 'Quiz 1,Quiz 2', 'Advanced', 'Great job!', 'Standard', 'Good effort!', 'Review', 'Keep trying!'],
    ])

    return wb
