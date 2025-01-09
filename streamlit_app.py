import os

os.environ["OPENAI_API_KEY"] = "3G9udyTZD3sbH6f" 

import streamlit as st
from llama_index.core import VectorStoreIndex, Document
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
#from OpenAILike import OpenAILike
import openai
from llama_index.core import SimpleDirectoryReader

# Lokale Definitionen
api_base = "https://kaih2md.streamlit.app/"
api_key = "3G9udyTZD3sbH6f"
model_name = "KaiH2md"
openai.api_key = api_key

# Initialisiere Sprachmodell
Settings.llm = OpenAI(api_base=api_base, api_key=api_key, model=model_name, max_tokens=256, temperature=0.8, system_prompt="Du bist Experte für den Studiengang AI.Engineering. Das Gespräch soll sich um die Studienordnung und den Studiengang drehen. Deine Antworten entsprechen den Fakten entsprechend der verfügbaren Dokumente - halluziniere keine Fakten.")

### Initialisiere Embeddings
from llama_index.embeddings.openai import OpenAIEmbedding

embed_model = OpenAIEmbedding(api_key="3G9udyTZD3sbH6f")

# Initialisiere den Chat:
st.header("Fragen zum Studiengang AI.Engineering 💬")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ich beantworte dir Fragen zur Studienordnung des Studiengangs AI.Engineering an der Otto-von-Guericke Universität Magdeburg sowie an den Hochschulen Anhalt, Harz, Magdeburg-Stendal und Merseburg"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Ich lese die abgelegten Dokumente. Moment..."):
        # PDF-Datei lesen
        with open("/1_SPO_BA_MB_AB_3_2019.pdf", "rb") as pdf_file:
            pdf_reader = pypdf.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

        # Dokument erstellen
        documents = [Document(text=text)]

        # ServiceContext initialisieren (mit deinem OpenAI-Modell)
        service_context = ServiceContext.from_defaults(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo"))

        # Index erstellen
        index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True) 
        return index
