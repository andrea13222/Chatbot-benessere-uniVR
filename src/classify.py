import json
from pathlib import Path
from llama_cpp import LlamaGrammar
from safety_check import controllo_semantico_urgenza
from llm_engine import get_llm

BASE_DIR = Path(__file__).resolve().parent.parent
GRAMMAR_PATH = BASE_DIR / "grammars" / "labels.gbnf"

SITUAZIONI_URGENTI = {
    "Comportamenti autolesivi",
    "Ideazione suicidaria",
    "Allucinazioni e deliri",
    "Aggressivita eterodiretta",
}

KEYWORD_URGENTI = [
    "farmi del male", "farmi male", "mi faccio del male", "mi sono fatto del male", "mi taglio", "tagliarmi", "tagliarsi", "mi ferisco", "ferirmi",
    "autolesion", "farmi la pelle", "bruciarmi", "mi brucio", "mi faccio male apposta", "mi voglio tagliare",

    "suicid",
    "impiccar",
    "ammazzar",
    "uccider",
    "essere morto",
    "togliermi la vita", "togliersi la vita", "porre fine alla mia vita", "farla finita", "non voglio piu vivere", "non voglio piu' vivere",
    "non ho piu voglia di vivere", "non ce la faccio piu a vivere", "non ce la faccio piu'", "voglio morire", "vorrei morire", "voglio la morte",
    "meglio se non ci fossi", "meglio se non ci fossi piu", "sarebbe meglio se morissi", "sarebbe meglio se non ci fossi", "non voglio piu esistere", "vorrei sparire per sempre",
    "voglio farla finita", "buttarmi giu", "buttarmi dal", "buttarmi sotto", "gettarmi giu", "gettarmi dal",
    "prendere tutte le pastiglie", "prendere tutti i farmaci",

    "sento voci", "sento delle voci",
    "vedo cose che non ci sono", "vedo cose che non esistono",

    "voglio fare del male a", "voglio uccidere",
    "ho intenzione di uccidere", "ho intenzione di fare del male a",
    "faro del male a", "faro' del male a", "mi sento male",
]


def _controllo_keyword_urgenza(testo):
    testo_normalizzato = testo.lower()
    for parola_chiave in KEYWORD_URGENTI:
        if parola_chiave in testo_normalizzato:
            return True
    return False


#_grammar = None


def _get_grammar():
    return LlamaGrammar.from_file(str(GRAMMAR_PATH))


FEW_SHOT_EXAMPLES = """Esempio 1:
Messaggio: "sono al primo anno di ingegneria e sono in ansia per l'esame di analisi 1"
Classificazione: {"tipo_utente": "Studente triennale", "stato_psicologico": "Ansia", "causa": "Difficolta legate al percorso / agli esami", "tipologia": {"modalita_erogazione": "Presenza", "formato": "Individuale", "lingua": "Italiano", "format": "Attivita di supporto/intervento guidato/Consulenza"}}

Esempio 2:
Messaggio: "sono un dottorando e sento un forte burnout per il carico di lavoro sulla mia ricerca"
Classificazione: {"tipo_utente": "Dottorando", "stato_psicologico": "Burnout", "causa": "Difficolta con il progetto di ricerca", "tipologia": {"modalita_erogazione": "On-line sincrona", "formato": "Individuale", "lingua": "Italiano", "format": "Attivita di supporto/intervento guidato/Consulenza"}}

Esempio 2bis (attenzione: "ricercatore" e' un ruolo DIVERSO da "dottorando"):
Messaggio: "sono un ricercatore e mi sento molto irritabile per motivi familiari"
Classificazione: {"tipo_utente": "Ricercatore", "stato_psicologico": "Irritabilita / scatti d'ira", "causa": "Malattia propria o di un familiare", "tipologia": {"modalita_erogazione": "On-line sincrona", "formato": "Individuale", "lingua": "Italiano", "format": "Attivita di supporto/intervento guidato/Consulenza"}}

Esempio 3:
Messaggio: "sono una professoressa e ho un conflitto continuo con un collega, mi sento molto stressata"
Classificazione: {"tipo_utente": "Professore", "stato_psicologico": "Stress", "causa": "Difficolta legate al rapporto con i pari", "tipologia": {"modalita_erogazione": "Presenza", "formato": "Individuale", "lingua": "Italiano", "format": "Attivita di supporto/intervento guidato/Consulenza"}}

Esempio 4:
Messaggio: "Sono un tecnico di laboratorio su sedia a rotelle e l'elevatore dell'edificio didattico è rotto da due giorni. Chi devo contattare per l'accessibilità?"
Classificazione: {"tipo_utente": "Personale tecnico-amministrativo (TA)", "stato_psicologico": "Disagio", "causa": "Disabilità", "tipologia": {"modalita_erogazione": "Presenza", "formato": "Individuale", "lingua": "Italiano", "format": "Attività di supporto/intervento guidato/Consulenza"}}

Esempio 5:
Messaggio: "Un professore continua a fare battute inappropriate e umilianti durante il ricevimento. Con chi posso parlare in forma anonima?"
Classificazione: {"tipo_utente": "Studente triennale", "stato_psicologico": "Disagio", "causa": "Molestie", "tipologia": {"modalita_erogazione": "Presenza", "formato": "Individuale", "lingua": "Italiano", "format": "Attività di supporto/intervento guidato/Consulenza"}}

Esempio 6:
Messaggio: "In reparto facciamo turni da 12 ore continue senza cambi regolari. Temo di commettere errori medici per la troppa stanchezza."
Classificazione: {"tipo_utente": "Specializzandi", "stato_psicologico": "Stanchezza cronica", "causa": "Sovraccarico lavorativo", "tipologia": {"modalita_erogazione": "Presenza", "formato": "Individuale", "lingua": "Italiano", "format": "Attività di supporto/intervento guidato/Consulenza"}}

Esempio 7:
Messaggio: "Con il nuovo software gestionale il carico di lavoro alla segreteria è raddoppiato. Siamo solo in due allo sportello."
Classificazione: {"tipo_utente": "Personale tecnico-amministrativo (TA)", "stato_psicologico": "Stress", "causa": "Sovraccarico lavorativo", "tipologia": {"modalita_erogazione": "Presenza", "formato": "Individuale", "lingua": "Italiano", "format": "Attività di supporto/intervento guidato/Consulenza"}}

Esempio 8:
Messaggio: "Tutti gli altri dottorandi del mio ciclo pubblicano continuamente, mentre io sento di aver truffato la commissione."
Classificazione: {"tipo_utente": "Dottorando", "stato_psicologico": "Senso di inadeguatezza", "causa": "Difficoltà con il progetto di ricerca", "tipologia": {"modalita_erogazione": "Presenza", "formato": "Individuale", "lingua": "Italiano", "format": "Attività di supporto/intervento guidato/Consulenza"}}

Esempio 9:
Messaggio: "Lavoro nell'ufficio archivio al piano seminterrato, non vedo mai la luce del sole e non interagisco con nessun collega per giorni."
Classificazione: {"tipo_utente": "Personale tecnico-amministrativo (TA)", "stato_psicologico": "Isolamento sociale", "causa": "Difficoltà di conciliazione vita lavoro o studio", "tipologia": {"modalita_erogazione": "Presenza", "formato": "Individuale", "lingua": "Italiano", "format": "Attività di supporto/intervento guidato/Consulenza"}}"""


#################################AUMENTARE CASI D'ESEMPIO SE LLM NON ETICHETTA CORRETTAMENTE ##################################################################################
#Esempio 4 (livello NON specificato -> usa il default triennale):
#Messaggio: "sono uno studente e ho paura di fallire l'esame di chimica organica"
#Classificazione: {"tipo_utente": "Studente triennale", "stato_psicologico": "Paura del fallimento", "causa": "Difficolta legate al percorso / agli esami", "tipologia": {"modalita_erogazione": "Presenza", "formato": "Individuale", "lingua": "Italiano", "format": "Attivita di supporto/intervento guidato/Consulenza"}}

#Esempio 5 (livello magistrale ESPLICITO):
#Messaggio: "sono al secondo anno di magistrale e sono in ansia per la tesi"
#Classificazione: {"tipo_utente": "Studente magistrale", "stato_psicologico": "Ansia", "causa": "Difficolta legate alla tesi", "tipologia": {"modalita_erogazione": "Presenza", "formato": "Individuale", "lingua": "Italiano", "format": "Attivita di supporto/intervento guidato/Consulenza"}}

#Esempio 6 (situazione URGENTE - autolesionismo, priorita' massima):
#Messaggio: "non ce la faccio piu, mi faccio del male"
#Classificazione: {"tipo_utente": "Studente triennale", "stato_psicologico": "Comportamenti autolesivi", "causa": "Difficolta legate al percorso / agli esami", "tipologia": {"modalita_erogazione": "Presenza", "formato": "Individuale", "lingua": "Italiano", "format": "Attivita di supporto/intervento guidato/Consulenza"}}
#Nota: qualunque riferimento a farsi del male, autolesionismo, pensieri di morte o suicidio
#DEVE sempre risultare in stato_psicologico = "Comportamenti autolesivi" o "Ideazione suicidaria",
#mai in etichette piu' generiche come "Disagio" o "Crisi emotiva".

def classify(user_message):
    llm = get_llm()
    grammar = _get_grammar()

    prompt = f"""Sei un sistema di classificazione per un servizio di supporto psicologico universitario.
Analizza il messaggio dell'utente e assegna le etichette PIU' PRECISE possibili, scegliendo
SOLO tra i valori ammessi dalla grammatica fornita.

Presta attenzione a questi segnali nel messaggio:
- tipo_utente: usa SOLO indizi espliciti nel testo. Regola di default OBBLIGATORIA: se
  l'utente non specifica ne' l'anno ne' il livello di laurea, e si presenta genericamente
  come "studente", classifica come "Studente triennale". Usa "Studente magistrale" SOLO se
  il messaggio menziona esplicitamente parole come "magistrale", "laurea magistrale",
  "secondo anno di magistrale", "specialistica".
  ATTENZIONE: "Dottorando", "Post-Doc", "Ricercatore" e "Professore" sono ruoli DISTINTI,
  non intercambiabili. Se l'utente scrive esplicitamente "sono un ricercatore", la risposta
  DEVE essere "Ricercatore", MAI "Dottorando" o altro ruolo simile. Usa sempre la parola
  esatta che l'utente ha usato per descrivere il proprio ruolo, senza sostituirla con un
  ruolo "vicino" o dedotto.
- stato_psicologico: l'emozione o il disagio ESPRESSO direttamente dall'utente, non dedotto
- causa: l'argomento/contesto CONCRETO citato nel messaggio - NON scegliere cause non
  menzionate nel testo
- tipologia: se non ci sono indicazioni esplicite, usa Presenza, Individuale, Italiano,
  Attivita di supporto/intervento guidato/Consulenza

{FEW_SHOT_EXAMPLES}

Ora classifica questo messaggio. Rispondi ESCLUSIVAMENTE con il JSON, nessun testo aggiuntivo.

Messaggio utente: "{user_message}"

Classificazione:"""

    output = llm(
        prompt,
        max_tokens=200,
        grammar=grammar,
        temperature=0.0,
    )

    raw_text = output["choices"][0]["text"]
    is_urgente_keyword = _controllo_keyword_urgenza(user_message)
    is_urgente_semantico = controllo_semantico_urgenza(user_message)

    try:
        labels = json.loads(raw_text)
    except json.JSONDecodeError:
        print(f"[classify.py] Errore parsing JSON. Output grezzo: {raw_text}")
        if is_urgente_keyword or is_urgente_semantico:
            return {"is_urgente": True, "motivo": "rilevato da controllo keyword/semantico (parsing LLM fallito)"}
        return None

    is_urgente_llm = labels.get("stato_psicologico") in SITUAZIONI_URGENTI
    labels["is_urgente"] = is_urgente_llm or is_urgente_keyword or is_urgente_semantico

    if (is_urgente_keyword or is_urgente_semantico) and not is_urgente_llm:
        print("[classify.py] il controllo keyword/semantico ha rilevato "
              "urgenza ma il LLM no. In uso l'override di sicurezza.")

    return labels


if __name__ == "__main__":
    msg = input("Messaggio utente di prova: ")
    result = classify(msg)
    print(json.dumps(result, indent=2, ensure_ascii=False))
