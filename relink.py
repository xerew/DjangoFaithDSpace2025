import re
from authoringtool.models import ActivityProposal, ActivityFlag  # βάλε το σωστό app label

SCENARIO_ID = 29  # βάλε scenario id

pattern = re.compile(r"\[(High|Moderate|Low)\]\s*([^\:]+)\:")

qs = ActivityProposal.objects.filter(scenario_id=SCENARIO_ID)
fixed = 0

for p in qs:
    # καθάρισε παλιά links
    p.flag.clear()

    text = p.suggested_action or ""
    matches = pattern.findall(text)  # list of (category, flag_type)

    if not matches:
        continue

    # φέρε flags που ταιριάζουν με activity + (category, flag_type)
    flags_to_attach = []
    for category, flag_type in matches:
        fqs = ActivityFlag.objects.filter(
            activity=p.activity,
            category=category.strip(),
            flag_type=flag_type.strip(),
            is_at_risk=True,
        )
        flags_to_attach.extend(list(fqs))

    if flags_to_attach:
        p.flag.add(*set(flags_to_attach))
        fixed += 1

print("Relinked proposals:", fixed, "out of", qs.count())