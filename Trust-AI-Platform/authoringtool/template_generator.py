import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.datavalidation import DataValidation

_HEADER_FONT = Font(bold=True, color='FFFFFF')
_HEADER_FILL = PatternFill('solid', fgColor='1D4ED8')
_REQ_FILL = PatternFill('solid', fgColor='DC2626')


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
    for bool_col in bool_cols:
        if bool_col in col_name_to_idx:
            idx = col_name_to_idx[bool_col]
            letter = ws.cell(row=1, column=idx).column_letter
            dv = DataValidation(type='list', formula1='"Yes,No"', showDropDown=False)
            dv.sqref = f'{letter}2:{letter}200'
            ws.add_data_validation(dv)

    for row_data in example_rows:
        ws.append(row_data)

    return ws


def _write_readme(ws):
    ws['A1'] = 'SCENARIO TEMPLATE — INSTRUCTIONS'
    ws['A1'].font = Font(bold=True, size=14)
    lines = [
        '',
        'SHEETS AND THEIR PURPOSE',
        '  Scenario     — One row of scenario metadata (name, description, etc.)',
        '  Phases       — One row per phase; row order = phase order in the scenario',
        '  Activities   — One row per activity; row order = activity order within each phase',
        '  Answers      — One row per answer choice (multiple rows per activity)',
        '  Next Activity — One row per routing rule (which activity comes after which)',
        '  Evaluation   — One row per evaluatable activity (scoring groups + branching)',
        '',
        'REFERENCING RULES',
        '  Activities reference Phases by Phase Name — must match exactly.',
        '  Answers, Next Activity, and Evaluation reference Activities by Activity Name.',
        '  Activity names must be unique within the file.',
        '  Next Activity references answers by Answer Key — use the key, not the display text.',
        '  TIP: Copy-paste names and keys from other sheets to avoid typos.',
        '',
        'ACTIVITY TYPE CHOICES  (Activity Type is required)',
        '  Explanation — Present information (text, images, video) to the student',
        '  Question    — Ask a question; student selects from answer choices',
        '  Experiment  — Embed a simulation or remote lab',
        '               NOTE: After import, you must manually assign the simulation or',
        '               lab to each Experiment activity in the authoring tool.',
        '  Guidance    — Provide targeted feedback or guidance to the student',
        '',
        'TEXT FIELDS',
        '  Activity Text supports Markdown formatting:',
        '    **bold**   _italic_   # Heading   - bullet list   [Link](https://...)',
        '  Markdown is converted to HTML automatically on import.',
        '',
        'BOOLEAN FIELDS (Is Evaluatable, Is Correct)',
        '  Use the dropdown: Yes or No (case-insensitive).',
        '',
        'STUDENT PERFORMANCE CATEGORIES',
        '  After evaluation, students are placed into one of three groups:',
        '    High     — average answer weight ≥ 2.5',
        '    Moderate — average answer weight ≥ 1.5',
        '    Low      — all remaining students (fallback)',
        '  Thresholds (2.5 / 1.5 / 1.0) are fixed and applied automatically on import.',
        '  You do not need to enter them.',
        '',
        'NEXT ACTIVITY SHEET',
        '  Leave "Answer Key" blank for an unconditional (default) next activity.',
        '  Leave "Next Activity Name" blank to end the scenario at that activity.',
        '  Example:',
        '    Source Activity | Answer Key  | Next Activity',
        '    Welcome         |             | Quiz 1         (default route)',
        '    Quiz 1          | ans_correct | Result High    (per-answer route)',
        '    Quiz 1          | ans_wrong   | Result Low',
        '',
        'EVALUATION SHEET',
        '  Primary Activity Name: an activity with Is Evaluatable = Yes.',
        '  Grouped Activities: comma-separated activity names that form the scoring group.',
        '    Include the primary activity itself in this list.',
        '  High / Moderate / Low Performers Activity: where students go based on their score.',
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
        ('Language', False), ('Subject Domains', False),
        ('Age Min', False), ('Age Max', False),
        ('Suggested Time (min)', False), ('Visibility', False),
    ], example_rows=[
        ['My Scenario', 'A brief description', 'Students will learn...', 'English',
         'Physics,STEM', '14', '18', '90', 'private'],
    ])

    _write_sheet(wb, 'Phases', [
        ('Phase Name', True), ('Description', False),
    ], example_rows=[
        ['Engagement', ''],
        ['Hypothesis', ''],
        ['Experiment', ''],
        ['Analysis', ''],
        ['Reflection', ''],
    ])

    _write_sheet(wb, 'Activities', [
        ('Activity Name', True), ('Phase Name', True), ('Text', True),
        ('Activity Type', True), ('Helper', False),
        ('Is Evaluatable', False), ('Is Primary Evaluation', False),
        ('Simulation Name', False), ('Remote Lab Name', False), ('VR Lab Name', False),
    ], bool_cols=['Is Evaluatable', 'Is Primary Evaluation'], example_rows=[
        ['Welcome', 'Engagement', 'Welcome to this scenario! **Read carefully.**',
         'Explanation', '', 'No', 'No', '', '', ''],
        ['Quiz 1', 'Analysis', 'What is the boiling point of water?',
         'Question', '', 'Yes', 'Yes', '', '', ''],
    ])

    ans_ws = _write_sheet(wb, 'Answers', [
        ('Activity Name', True), ('Answer Key', True), ('Answer Text', True),
        ('Is Correct', False), ('Answer Weight', False),
    ], bool_cols=['Is Correct'], example_rows=[
        ['Quiz 1', 'ans_correct', '100°C', 'Yes', '1'],
        ['Quiz 1', 'ans_wrong',   '50°C',  'No',  '0'],
    ])
    # Pre-fill Answer Key column (B) with a formula so new rows get a unique key automatically.
    # Teachers can overwrite it with their own key; the formula returns "" when A is empty.
    for _row in range(4, 201):
        ans_ws.cell(row=_row, column=2, value=f'=IF(A{_row}="","","ans_"&(ROW()-1))')

    _write_sheet(wb, 'Next Activity', [
        ('Source Activity Name', True), ('Answer Key', False), ('Next Activity Name', False),
    ], example_rows=[
        ['Welcome',  '',            'Quiz 1'],
        ['Quiz 1',   'ans_correct', 'Well Done'],
        ['Quiz 1',   'ans_wrong',   'Try Again'],
    ])

    _write_sheet(wb, 'Evaluation', [
        ('Primary Activity Name', True), ('Grouped Activities', True),
        ('High Performers Activity', False),
        ('Moderate Performers Activity', False),
        ('Low Performers Activity', False),
    ], example_rows=[
        ['Quiz 1', 'Quiz 1,Quiz 2', 'Result High', 'Result Standard', 'Result Low'],
    ])

    return wb
