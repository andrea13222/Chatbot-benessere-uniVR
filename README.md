# Chatbot-benessere-uniVR

# SU MACOS/LINUX:
## 1. attivazione ambiente

python -m venv venv
source venv/bin/activate

## 2. installazione dipoendenze python presente in /venv

pip install -r requisiti.txt

## 3. scaricare i modelli
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
