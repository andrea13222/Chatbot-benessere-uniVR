from llm_engine import get_llm
import unicodedata

#SYSTEM_INSTRUCTION = """
#Rispondi come un assistente universitario.

#Rispondi direttamente alla frase dell'utente.
#Usa il contesto fornito solo se contiene informazioni utili.
#Non inventare informazioni.

#Scrivi solo la risposta da mostrare all'utente.
#Non descrivere il tuo comportamento.
#Non scrivere regole, note o commenti sulla risposta.
#Evita domande generiche di apertura.
#Non elencare le informazioni: componi un discorso naturale.
#Scrivi la risposta tutta in una riga.
#"""
SYSTEM_INSTRUCTION = """
Rispondi come un assistente dell'universita di verona.

Rispondi alla frase dell'utente.
Usa il contesto fornito solo se contiene informazioni utili.
Non inventare informazioni.
NON scrivere nome di file.
Se small talk, rispondi in modo naturale e SENZA UTILIZZARE IL CONTESTO FORNITO.

Scrivi solo la risposta da mostrare all'utente.
Non descrivere il tuo comportamento.
Non scrivere regole, note o commenti sulla risposta.
Evita domande generiche di apertura.
NON FARE ELENCHI.
Scrivi la risposta IN ITALIANO tutta in una riga.
"""


def generate_answer(user_message, retrieved_chunks):
    llm = get_llm()

    MAX_CONTEXT_CHARS = 3000

    context_parts = []
    current_length = 0
    used_sources = set()

    for item in retrieved_chunks:
        chunk = item[0] if len(item) > 0 else ""
        src = item[1] if len(item) > 1 else ""

        piece = f"[Fonte: {src}]\n{chunk}\n\n"

        if current_length + len(piece) > MAX_CONTEXT_CHARS:
            break

        context_parts.append(piece)
        current_length += len(piece)

        if src:
            src_normalizzato = unicodedata.normalize("NFC", str(src).strip().lower())
            used_sources.add(src_normalizzato)

    context = "".join(context_parts)

    prompt = f"""{SYSTEM_INSTRUCTION}

Contesto disponibile:
{context}

Messaggio utente:
{user_message}

Rispondi:
"""

    output = llm(
        prompt,
        max_tokens=200,
        temperature=0.2,
        top_p=0.85,          #PROVA (RIMUOVERE SE NON FUNZIONA)
        top_k=40,           #PROVA (RIMUOVERE SE NON FUNZIONA)
        #repeat_penalty=1.1,
        stop=[
            "DOMANDA UTENTE:",
            "CONTESTO:",
            "RISPOSTA:",
            "Rispondi:",
            "Messaggio utente:",
            "Contesto disponibile:",
            "Rispondi solo alla frase dell'utente",
            "Usando il contesto fornito:",
            "#",
            "CSS",
            "🌟",
            "🔴",
            "Contesto fornito:",
            "Buona fortuna",
            #"157",
        ]
    )

    answer = output["choices"][0]["text"].strip()

    if answer and answer[-1] not in ['.', '!', '?']:
        last_dot = answer.rfind('.')
        last_exclamation = answer.rfind('!')
        last_question = answer.rfind('?')

        last_punct = max(last_dot, last_exclamation, last_question)

        if last_punct != -1:
            answer = answer[:last_punct + 1]
        else:
            answer += "..."

    if used_sources:
        answer += "\n\n<b>Fonti utilizzate:</b>\n"

        BASE_URL = "https://www.univr.it/"

        url_stampati = set()
        nomi_stampati = set()

        for source in used_sources:
            source = source.split("#")[0]
            percorso_base = source.replace(".txt", "")
            percorso_url = percorso_base.replace("_", "/")
            url_completo = f"{BASE_URL}{percorso_url}"
            nome_visibile = percorso_base.split("_")[-1].replace("-", " ").title()

            if url_completo in url_stampati or nome_visibile in nomi_stampati:
                continue

            url_stampati.add(url_completo)
            nomi_stampati.add(nome_visibile)

            answer += f'- <a href="{url_completo}" target="_blank" style="color: #3f6f8f; text-decoration: underline; font-weight: bold;">{nome_visibile}</a>\n'

    return answer.strip()
