import os
import comet_llm
from opik import track

# ✅ Set all relevant cache directories to a writable location
os.environ["HF_HOME"] = "/tmp/cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/cache/transformers"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/cache/sentence_transformers"
os.environ["HF_DATASETS_CACHE"] = "/tmp/cache/hf_datasets"
os.environ["TORCH_HOME"] = "/tmp/cache/torch"

# ✅ Create the directories if they don't exist
for path in [
    "/tmp/cache",
    "/tmp/cache/transformers",
    "/tmp/cache/sentence_transformers",
    "/tmp/cache/hf_datasets",
    "/tmp/cache/torch"
]:
    os.makedirs(path, exist_ok=True)
import json
import torch
import openai
import os
from sentence_transformers import SentenceTransformer, util
import streamlit as st
from pathlib import Path

# === CONFIG ===
# Set the API key
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#openai.api_key = os.getenv("OPENAI_API_KEY")
# REMEDI_PATH = "ReMeDi-base.json"
BASE_DIR = Path(__file__).parent
REMEDI_PATH = BASE_DIR / "ReMeDi-base.json"

# Check if file exists
if not REMEDI_PATH.exists():
    raise FileNotFoundError(f"❌ File not found: {REMEDI_PATH}")

# Load the file
with open(REMEDI_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
    #return model

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
def build_embeddings(dialogue_pairs, _model):
    patient_sentences = [pair["patient"] for pair in dialogue_pairs]
    embeddings = _model.encode(patient_sentences, convert_to_tensor=True)
    return embeddings

# === TRANSLATE USING GPT ===
@track
def translate_to_english(chinese_text):
    prompt = f"Translate the following Chinese medical response to English:\n\n{chinese_text}"
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content

        #return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"Translation failed: {str(e)}"
@track
def gpt_direct_response(user_input):
    prompt = f"You are a knowledgeable and compassionate medical assistant. Answer the following patient question clearly and concisely:\n\n{user_input}"
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo" to save credits
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GPT response failed: {str(e)}"


# === CHATBOT FUNCTION ===

@track
def chatbot_response(user_input, _model, dialogue_pairs, patient_embeddings, top_k=1):
    user_embedding = _model.encode(user_input, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, patient_embeddings)[0]
    top_score, top_idx = torch.topk(similarities, k=1)
    top_score = top_score.item()
    top_idx = torch.topk(similarities, k=top_k).indices[0].item()

    match = dialogue_pairs[top_idx]
    translated = translate_to_english(match["doctor"])

    return {
        "matched_question": match["patient"],
        "original_response": match["doctor"],
        "translated_response": translated
        #"similarity_score": top_score
    }

# === MAIN APP ===
st.set_page_config(page_title="Dr_Q_bot", layout="centered")
st.title("🩺 Dr_Q_bot - Medical Chatbot")
st.write("Ask about a symptom and get an example doctor response and enhanced GPT-4 LLM Doctor's response")

# Add author info in the sidebar
with st.sidebar:
    st.markdown("## 👤👤Authors")
    st.markdown("**Vasan Iyer**")
    st.markdown("**Eric J Giacomucci**")
    st.markdown("[GitHub](https://github.com/Vaiy108)")
    st.markdown("[LinkedIn](https://linkedin.com/in/vasan-iyer)")         
# Load resources
model = load_model()
dialogue_pairs = load_data()
patient_embeddings = build_embeddings(dialogue_pairs, model)

# Chat UI
user_input = st.text_input("Describe your symptom:")

if st.button("Submit") and user_input:
    with st.spinner("Thinking..."):
        result = chatbot_response(user_input, model, dialogue_pairs, patient_embeddings)
        gpt_response = gpt_direct_response(user_input)

        st.markdown("## ✅ GPT-4 Doctor's Response")
        st.success(gpt_response)

        #if torch.max(similarities).item() < 0.4:
        
        st.markdown("## 🔁 Example Historical Dialogue")      
        st.markdown("### 🧑‍⚕️ Closest Patient Question")
        st.write(result["matched_question"])

        st.markdown("### 🇨🇳 Original Doctor Response (Chinese)")
        st.write(result["original_response"])

        st.markdown("### 🌐 Translated Doctor Response (English)")
        st.success(result["translated_response"])
        #else:
        #    st.warning("No close match found in dataset. Using GPT response only.")
            

        #st.markdown("### 💬 GPT Doctor Response (AI-generated)")
        #st.info(gpt_response)
        
           
           # Skip dataset result

        st.markdown("---")
        st.warning("This chatbot uses real dialogue data for research and educational use only. Not a substitute for professional medical advice.")

