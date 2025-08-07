import json
import torch
import openai
import os
from sentence_transformers import SentenceTransformer, util
import streamlit as st

# === CONFIG ===
openai.api_key = os.getenv("OPENAI_API_KEY")
REMEDI_PATH = "ReMeDi-base.json"

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

@st.cache_resource
def load_data():
    with open(REMEDI_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    dialogue_pairs = []
    for conversation in data:
        turns = conversation["information"]
        for i in range(len(turns)-1):
            if turns[i]["role"] == "patient" and turns[i+1]["role"] == "doctor":
                dialogue_pairs.append({
                    "patient": turns[i]["sentence"],
                    "doctor": turns[i+1]["sentence"]
                })
    return dialogue_pairs

@st.cache_data
def build_embeddings(dialogue_pairs, model):
    patient_sentences = [pair["patient"] for pair in dialogue_pairs]
    embeddings = model.encode(patient_sentences, convert_to_tensor=True)
    return embeddings

# === TRANSLATE USING GPT ===
def translate_to_english(chinese_text):
    prompt = f"Translate the following Chinese medical response to English:\n\n{chinese_text}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"Translation failed: {str(e)}"

# === CHATBOT FUNCTION ===
def chatbot_response(user_input, model, dialogue_pairs, patient_embeddings, top_k=1):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, patient_embeddings)[0]
    top_idx = torch.topk(similarities, k=top_k).indices[0].item()

    match = dialogue_pairs[top_idx]
    translated = translate_to_english(match["doctor"])

    return {
        "matched_question": match["patient"],
        "original_response": match["doctor"],
        "translated_response": translated
    }

# === MAIN APP ===
st.set_page_config(page_title="Dr_Q_bot", layout="centered")
st.title("🩺 Dr_Q_bot - Medical Chatbot")
st.write("Ask about a symptom and get an example doctor response (translated from Chinese).")

# Load resources
model = load_model()
dialogue_pairs = load_data()
patient_embeddings = build_embeddings(dialogue_pairs, model)

# Chat UI
user_input = st.text_input("Describe your symptom:")

if st.button("Submit") and user_input:
    with st.spinner("Thinking..."):
        result = chatbot_response(user_input, model, dialogue_pairs, patient_embeddings)
        st.markdown("### 🧑‍⚕️ Closest Patient Question")
        st.write(result["matched_question"])

        st.markdown("### 🇨🇳 Original Doctor Response (Chinese)")
        st.write(result["original_response"])

        st.markdown("### 🌐 Translated Doctor Response (English)")
        st.success(result["translated_response"])

        st.markdown("---")
        st.warning("This chatbot uses real dialogue data for research and educational use only. Not a substitute for professional medical advice.")
