import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
from retrieve import retrieve
from classify import classify, _controllo_keyword_urgenza
from generate import generate_answer

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR))
CORS(app)

MESSAGGIO_EMERGENZA = """Il messaggio che hai scritto sembra riguardare una situazione seria per la quale è importante parlare subito con una persona qualificata, non con un chatbot.
Se sei in pericolo immediato, contatta il 112 (numero unico di emergenza).
Non sei solo/a: parlare con qualcuno è il primo passo."""


@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Messaggio vuoto"}), 400

    richiedi_classificazione = data.get("classifica", True)
    if not isinstance(richiedi_classificazione, bool):
        richiedi_classificazione = True

    t0 = time.time()

    urgente_keyword = _controllo_keyword_urgenza(user_message)

    if richiedi_classificazione:
        try:
            labels = classify(user_message)
        except Exception as e:
            print(f"[app_web.py] Errore imprevisto in classify(): {e}")
            labels = {"is_urgente": urgente_keyword}
    else:
        labels = {"is_urgente": urgente_keyword}

    if urgente_keyword:
        labels["is_urgente"] = True

    t_classify = time.time() - t0

    print(f"[app_web.py] Messaggio: \"{user_message}\"")
    print(f"[app_web.py] Etichette: {labels} (classificazione richiesta: {richiedi_classificazione}, keyword: {urgente_keyword})")
    print(f"[app_web.py] Tempo classificazione: {t_classify:.1f}s")

    if labels and labels.get("is_urgente"):
        return jsonify({
            "risposta": MESSAGGIO_EMERGENZA,
            "etichette": labels if richiedi_classificazione else None,
            "urgente": True
        })

    if labels and labels.get("errore") == "contesto troppo lungo":
        return jsonify({"risposta": labels["messaggio"],
            "etichette": None,
            "urgente": False
        })

    t0 = time.time()
    chunks = retrieve(user_message, top_k=2)
    t_retrieve = time.time() - t0

    t0 = time.time()
    answer = generate_answer(user_message, chunks)
    t_generate = time.time() - t0

    print(f"[app_web.py] Tempo retrieval: {t_retrieve:.1f}s, generazione: {t_generate:.1f}s, "
          f"totale: {t_classify + t_retrieve + t_generate:.1f}s")

    return jsonify({
        "risposta": answer,
        "etichette": labels if richiedi_classificazione else None,
        "urgente": False
    })


if __name__ == "__main__":
    STATIC_DIR.mkdir(exist_ok=True)
    print("Server avviato su http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)
