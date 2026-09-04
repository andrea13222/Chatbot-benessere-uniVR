# Chatbot-benessere-uniVR
Il chatbot funge da assistente virtuale per il benessere della comunità universitaria (sito web: https://www.univr.it/benessere). Classifica le richieste in base alle tassonomie date e genera la risposta da restituire all'utente rilevando situazioni di emergenza. Il sistema è interamente locale che utilizza Qwen2.5-3B-Instruct-Q4_K_M per la classificazione e la generazione della risposta. E' possibile richiedere o no la classificazione delle tassonomie, riducendo i tempi di risposta (i tempi di risposta variano in base alle risorse hardware disponibili).

Peso complessivo del sistema si attestano a poco meno di 6GB

## <ins>Il chatbot non e' uno psicologo, è opportuno verificare sempre le informazioni, il modello può commettere errori.</ins>

# SU MACOS/LINUX:
## 1. attivazione ambiente

python -m venv venv

source venv/bin/activate

## 2. installazione dipoendenze python in /venv

pip install -r requisiti.txt

## 3. scaricare i modelli in /models
Da Hugging Face (se non presente -> pip install -U huggingface_hub):

huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir ./models

huggingface-cli download intfloat/multilingual-e5-small --local-dir ./models/multilingual-e5-small

## 4. Popolare il /vectorstore

python src/ingest.py

## 5. Avviare il chatbot

python src/app_web.py


# AVVERTENZA
Il chatbot è stato progettato tramite l'hardware disponibile della seguente macchina:

Modello: Macbook Pro 2019
Specifiche:

2,4 GHz Intel Core i5 quad-core

8 GB 2133 MHz LPDDR3

Intel Iris Plus Graphics 655 1536 MB
