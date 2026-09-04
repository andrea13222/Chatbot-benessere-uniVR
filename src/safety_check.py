import re
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "multilingual-e5-small"
EMBEDDING_MODEL = str(MODELS_DIR)

FRASI_RIFERIMENTO_URGENZA = [
    "mi faccio del male",
    "mi taglio quando sto male",
    "mi ferisco di proposito",
    "penso di farmi del male",
    "ho impulsi di farmi male",
    "voglio impiccarmi",

    "voglio farla finita",
    "non voglio più vivere",
    "vorrei sparire per sempre",
    "penso spesso alla morte come una via d'uscita",
    "non vedo un futuro per me",
    "sarebbe meglio se non ci fossi più",
    "vorrei non svegliarmi più",
    "sto pensando al suicidio",
    "non ce la faccio più a vivere così",
    "vorrei che tutto finisse",

    "sento delle voci che mi parlano",
    "vedo cose che gli altri non vedono",
    "penso che qualcuno mi stia spiando o controllando",

    "voglio fare del male a qualcuno",
    "ho pensieri di violenza verso altre persone",
    "vorrei vendicarmi facendo del male a qualcuno",

    "come si realizza un cappio?",
]

FRASI_RIFERIMENTO_NEUTRE = [
    "oggi mi sento felice",
    "sono contento del mio esame",
    "va tutto bene, sono soddisfatto del mio percorso",
    "mi sono divertito con i miei amici",
    "sono entusiasta del nuovo semestre",
    "sono un po' stanco ma sto bene",
    "ho superato l'esame con un bel voto",
    "mi piace molto il mio corso di studi",
    "oggi è stata una bella giornata",
    "sono tranquillo riguardo al futuro",
    "vorrei parlare con uno psicologo",
    "voglio prenotare un colloquio psicologico",
    "vorrei un supporto per gestire lo stress da esami",
    "ho bisogno di parlare con qualcuno del servizio di ascolto",
    "vorrei fissare un appuntamento con un counselor",
    "mi piacerebbe avere un colloquio di supporto psicologico",
    "vorrei informazioni sul servizio di supporto psicologico",
    "sono un po' in ansia per gli esami e vorrei un consiglio",
]

PATTERN_ESPLICITI_URGENZA = [
    r"impiccar\w*",
    r"\bmi\s+ammazz\w*",
    r"ammazzarmi",
    r"suicid\w*",
    r"\bmi\s+tagli\w*",
    r"tagliarmi",
    r"tagliarsi",
    r"farla finita",
    r"\bmi\s+uccid\w*",
    r"uccidermi",
    r"uccidersi",
    r"non voglio più vivere",
    r"non ce la faccio più a vivere",
    r"voglio morire",
    r"voglio\s+la\s+morte",
    r"togliermi la vita",
    r"farla\s+finita",
    r"farmi del male",
    r"farmi male",
]

_PATTERN_ESPLICITI_COMPILATI = [re.compile(p, re.IGNORECASE) for p in PATTERN_ESPLICITI_URGENZA]

_embedder = None
_embeddings_riferimento = None
_embeddings_neutre = None

MARGINE_MINIMO = 0.02


def _controllo_keyword_esplicite(testo_utente):
    testo = testo_utente.strip()
    return any(pattern.search(testo) for pattern in _PATTERN_ESPLICITI_COMPILATI)


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_embeddings_riferimento():
    global _embeddings_riferimento
    if _embeddings_riferimento is None:
        embedder = _get_embedder()
        frasi_con_prefisso = [f"query: {frase}" for frase in FRASI_RIFERIMENTO_URGENZA]
        _embeddings_riferimento = embedder.encode(
            frasi_con_prefisso, normalize_embeddings=True, convert_to_tensor=True
        )
    return _embeddings_riferimento


def _get_embeddings_neutre():
    global _embeddings_neutre
    if _embeddings_neutre is None:
        embedder = _get_embedder()
        frasi_con_prefisso = [f"query: {frase}" for frase in FRASI_RIFERIMENTO_NEUTRE]
        _embeddings_neutre = embedder.encode(
            frasi_con_prefisso, normalize_embeddings=True, convert_to_tensor=True
        )
    return _embeddings_neutre


def controllo_semantico_urgenza(testo_utente, margine=MARGINE_MINIMO, verbose=False):
    testo = testo_utente.strip()

    if _controllo_keyword_esplicite(testo):
        if verbose:
            print(" Rilevato urgenza da parole chiave.")
        return True

    if len(testo.split()) < 3:
        return False

    embedder = _get_embedder()
    embeddings_rischio = _get_embeddings_riferimento()
    embeddings_neutre = _get_embeddings_neutre()

    embedding_utente = embedder.encode(
        f"query: {testo_utente}", normalize_embeddings=True, convert_to_tensor=True
    )

    sim_rischio = util.cos_sim(embedding_utente, embeddings_rischio)[0]
    sim_neutre = util.cos_sim(embedding_utente, embeddings_neutre)[0]

    max_sim_rischio = float(sim_rischio.max())
    max_sim_neutre = float(sim_neutre.max())
    differenza = max_sim_rischio - max_sim_neutre

    if verbose:
        idx_rischio = int(sim_rischio.argmax())
        idx_neutro = int(sim_neutre.argmax())
        print(f"  Similarità max RISCHIO: {max_sim_rischio:.4f} -> \"{FRASI_RIFERIMENTO_URGENZA[idx_rischio]}\"")
        print(f"  Similarità max NEUTRA:  {max_sim_neutre:.4f} -> \"{FRASI_RIFERIMENTO_NEUTRE[idx_neutro]}\"")
        print(f"  Differenza: {differenza:.4f} (soglia margine: {margine})")

    return differenza >= margine
