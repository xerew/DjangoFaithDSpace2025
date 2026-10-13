"""Text of the "Ask for help" button and pause overlay, in the scenario's language.

Keys are the language names in the authoringtool Language table (seeded in
migrations 0045 and 0048). Unknown or empty languages fall back to English.
"""

HELP_LABELS = {
    'English': {
        'ask': 'Ask for help',
        'title': 'Paused: help requested',
        'message': 'Your teacher has been asked for help. You can continue when you are ready.',
        'resume': 'Resume',
        'pause_error': 'Could not pause the scenario. Please try again.',
        'resume_error': 'Could not resume the scenario. Please try again.',
    },
    'Basque': {
        'ask': 'Laguntza eskatu',
        'title': 'Etenda: laguntza eskatu da',
        'message': 'Irakasleari laguntza eskatu zaio. Prest zaudenean jarrai dezakezu.',
        'resume': 'Jarraitu',
        'pause_error': 'Ezin izan da eszenatokia eten. Saiatu berriro.',
        'resume_error': 'Ezin izan da eszenatokiarekin jarraitu. Saiatu berriro.',
    },
    'Catalan': {
        'ask': 'Demana ajuda',
        'title': 'En pausa: ajuda sol·licitada',
        'message': "S'ha demanat ajuda al teu professor. Pots continuar quan vulguis.",
        'resume': 'Reprèn',
        'pause_error': "No s'ha pogut posar en pausa l'escenari. Torna-ho a provar.",
        'resume_error': "No s'ha pogut reprendre l'escenari. Torna-ho a provar.",
    },
    'Estonian': {
        'ask': 'Küsi abi',
        'title': 'Peatatud: abi küsitud',
        'message': 'Sinu õpetajalt on abi küsitud. Võid jätkata, kui oled valmis.',
        'resume': 'Jätka',
        'pause_error': 'Stsenaariumi ei õnnestunud peatada. Proovi uuesti.',
        'resume_error': 'Stsenaariumi ei õnnestunud jätkata. Proovi uuesti.',
    },
    'French': {
        'ask': "Demander de l'aide",
        'title': 'En pause : aide demandée',
        'message': "Ton enseignant a été prévenu que tu as besoin d'aide. Tu peux continuer quand tu veux.",
        'resume': 'Reprendre',
        'pause_error': 'Impossible de mettre le scénario en pause. Réessaie.',
        'resume_error': 'Impossible de reprendre le scénario. Réessaie.',
    },
    'German': {
        'ask': 'Um Hilfe bitten',
        'title': 'Pausiert: Hilfe angefordert',
        'message': 'Deine Lehrkraft wurde um Hilfe gebeten. Du kannst weitermachen, sobald du bereit bist.',
        'resume': 'Fortsetzen',
        'pause_error': 'Das Szenario konnte nicht pausiert werden. Bitte versuche es erneut.',
        'resume_error': 'Das Szenario konnte nicht fortgesetzt werden. Bitte versuche es erneut.',
    },
    'Greek': {
        'ask': 'Ζήτησε βοήθεια',
        'title': 'Παύση: ζητήθηκε βοήθεια',
        'message': 'Ζητήθηκε βοήθεια από τον εκπαιδευτικό σου. Μπορείς να συνεχίσεις όποτε θέλεις.',
        'resume': 'Συνέχεια',
        'pause_error': 'Δεν ήταν δυνατή η παύση του σεναρίου. Δοκίμασε ξανά.',
        'resume_error': 'Δεν ήταν δυνατή η συνέχιση του σεναρίου. Δοκίμασε ξανά.',
    },
    'Italian': {
        'ask': 'Chiedi aiuto',
        'title': 'In pausa: aiuto richiesto',
        'message': "Il tuo insegnante è stato avvisato che hai bisogno di aiuto. Puoi continuare quando vuoi.",
        'resume': 'Riprendi',
        'pause_error': 'Impossibile mettere in pausa lo scenario. Riprova.',
        'resume_error': 'Impossibile riprendere lo scenario. Riprova.',
    },
    'Persian': {
        'ask': 'درخواست کمک',
        'title': 'متوقف شد: کمک درخواست شد',
        'message': 'از معلم شما کمک خواسته شد. هر وقت آماده بودید می‌توانید ادامه دهید.',
        'resume': 'ادامه',
        'pause_error': 'توقف سناریو ممکن نشد. لطفاً دوباره تلاش کنید.',
        'resume_error': 'ادامهٔ سناریو ممکن نشد. لطفاً دوباره تلاش کنید.',
        'dir': 'rtl',
    },
    'Portuguese': {
        'ask': 'Pedir ajuda',
        'title': 'Em pausa: ajuda pedida',
        'message': 'O teu professor foi avisado de que precisas de ajuda. Podes continuar quando quiseres.',
        'resume': 'Retomar',
        'pause_error': 'Não foi possível pausar o cenário. Tenta novamente.',
        'resume_error': 'Não foi possível retomar o cenário. Tenta novamente.',
    },
    'Romanian': {
        'ask': 'Cere ajutor',
        'title': 'Pauză: ajutor solicitat',
        'message': 'Profesorul tău a fost anunțat că ai nevoie de ajutor. Poți continua când vrei.',
        'resume': 'Continuă',
        'pause_error': 'Scenariul nu a putut fi pus pe pauză. Încearcă din nou.',
        'resume_error': 'Scenariul nu a putut fi reluat. Încearcă din nou.',
    },
    'Spanish': {
        'ask': 'Pedir ayuda',
        'title': 'En pausa: ayuda solicitada',
        'message': 'Se ha pedido ayuda a tu profesor. Puedes continuar cuando quieras.',
        'resume': 'Reanudar',
        'pause_error': 'No se pudo pausar el escenario. Inténtalo de nuevo.',
        'resume_error': 'No se pudo reanudar el escenario. Inténtalo de nuevo.',
    },
    'Turkish': {
        'ask': 'Yardım iste',
        'title': 'Duraklatıldı: yardım istendi',
        'message': 'Öğretmeninden yardım istendi. Hazır olduğunda devam edebilirsin.',
        'resume': 'Devam et',
        'pause_error': 'Senaryo duraklatılamadı. Lütfen tekrar dene.',
        'resume_error': 'Senaryo devam ettirilemedi. Lütfen tekrar dene.',
    },
    'Ukrainian': {
        'ask': 'Попросити допомоги',
        'title': 'Пауза: запит на допомогу',
        'message': 'Твого вчителя попросили про допомогу. Можеш продовжити будь-коли.',
        'resume': 'Продовжити',
        'pause_error': 'Не вдалося призупинити сценарій. Спробуй ще раз.',
        'resume_error': 'Не вдалося відновити сценарій. Спробуй ще раз.',
    },
}

_BY_NAME = {name.casefold(): labels for name, labels in HELP_LABELS.items()}


def help_labels(language):
    """Return the help texts for a scenario language, falling back to English."""
    labels = _BY_NAME.get((language or '').strip().casefold(), HELP_LABELS['English'])
    return {'dir': 'ltr', **labels}
