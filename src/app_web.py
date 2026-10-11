import time
import sqlite3
from functools import wraps
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from retrieve import retrieve
from classify import classify, _controllo_keyword_urgenza
from generate import generate_answer

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DB_PATH = BASE_DIR / "feedback.db"

app = Flask(__name__, static_folder=str(STATIC_DIR))
CORS(app)

ADMIN_USER = "placeholder" #MODIFICABILE
ADMIN_PASS = "placeholder" #MODIFICABILE

def richiedi_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or auth.username != ADMIN_USER or auth.password != ADMIN_PASS:
            return (
                "Accesso non autorizzato",
                401,
                {"WWW-Authenticate": 'Basic realm="Area Riservata Feedback"'}
            )
        return f(*args, **kwargs)
    return decorated

@app.before_request
def blocca_url_statico():
    if request.path == "/static/feedback.html":
        abort(403)

MESSAGGIO_EMERGENZA = """Il messaggio che hai scritto sembra riguardare una situazione seria per la quale è importante parlare subito con una persona qualificata, non con un chatbot.
Se sei in pericolo immediato, contatta il 112 (numero unico di emergenza).
Non sei solo/a: parlare con qualcuno è il primo passo."""

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stelle INTEGER NOT NULL CHECK (stelle >= 1 AND stelle <= 4),
                commento TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")

@app.route("/feedback")
@richiedi_auth
def pagina_feedback():
    return send_from_directory(str(STATIC_DIR), "feedback.html")

@app.route("/api/feedback/stats", methods=["GET"])
@richiedi_auth
def statistiche_feedback():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT stelle, COUNT(*) FROM feedback GROUP BY stelle")
            conteggi = {1: 0, 2: 0, 3: 0, 4: 0}
            for stella, count in cursor.fetchall():
                if stella in conteggi:
                    conteggi[stella] = count

            cursor.execute("""
                SELECT stelle, commento, datetime(timestamp, 'localtime')
                FROM feedback
                ORDER BY timestamp DESC
            """)
            commenti = [
                {"stelle": row[0], "commento": row[1], "timestamp": row[2]}
                for row in cursor.fetchall()
            ]

        return jsonify({
            "conteggi": conteggi,
            "totale_valutazioni": sum(conteggi.values()),
            "commenti": commenti
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
        return jsonify({
            "risposta": labels["messaggio"],
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

@app.route("/api/feedback", methods=["POST"])
def salva_feedback():
    data = request.get_json(force=True)
    stelle = data.get("stelle")
    commento = (data.get("commento") or "").strip()

    if stelle is None or not isinstance(stelle, int) or not (1 <= stelle <= 4):
        return jsonify({"error": "La valutazione da 1 a 4 stelle è obbligatoria."}), 400

    commento_valido = commento[:2000] if commento else None

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO feedback (stelle, commento) VALUES (?, ?)",
                (stelle, commento_valido)
            )
            conn.commit()

        return jsonify({"success": True, "message": "Feedback registrato con successo."}), 201
    except Exception as e:
        print(f"[app_web.py] Errore salvataggio feedback: {e}")
        return jsonify({"error": "Errore durante il salvataggio del feedback."}), 500

if __name__ == "__main__":
    init_db()
    STATIC_DIR.mkdir(exist_ok=True)
    print("Server avviato su http://127.0.0.1:5000")
    print("Database in SQLite per i feedback attivo in:", DB_PATH)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)
