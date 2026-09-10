import os
import io
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
from gtts import gTTS
from duckduckgo_search import DDGS
from dotenv import load_dotenv

# OCR Imports for Scanned/Image-based PDFs
import pytesseract
from pdf2image import convert_from_bytes

# Safe import for Ollama
try:
    import ollama
except ImportError:
    ollama = None

load_dotenv()

# 1. Page Configuration
st.set_page_config(
    page_title="S I Y A | AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Refined UI Custom CSS
st.markdown("""
<style>
    .stApp {
        background: #0B0E14;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .brand-logo {
        font-family: 'Courier New', monospace;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.25em !important;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-align: center !important;
        margin: 0 auto 20px auto !important;
        width: 100% !important;
        display: block !important;
        box-sizing: border-box !important;
    }

    [data-testid="stSidebar"] {
        background-color: #121721 !important;
        border-right: 1px solid #1E2638 !important;
    }

    [data-testid="stChatMessage"] {
        background-color: #161C28 !important;
        border: 1px solid #222B3D !important;
        border-radius: 14px !important;
        padding: 12px 18px !important;
        margin-bottom: 12px !important;
    }

    /* Embedded Form Chat Bar Styling */
    .stForm {
        background-color: #161C28 !important;
        border: 1px solid #2E384E !important;
        border-radius: 28px !important;
        padding: 4px 12px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }

    .stForm:focus-within {
        border-color: #00F2FE !important;
        box-shadow: 0 0 12px rgba(0, 242, 254, 0.25) !important;
    }

    div[data-testid="stTextInput"] > label {
        display: none !important;
    }

    div[data-testid="stTextInput"] input {
        background-color: transparent !important;
        color: #E2E8F0 !important;
        border: none !important;
        font-size: 0.95rem !important;
        box-shadow: none !important;
    }

    div[data-testid="stPopover"] {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    div[data-testid="stPopover"] > button {
        background-color: transparent !important;
        border: none !important;
        color: #00F2FE !important;
        font-size: 1.2rem !important;
        border-radius: 50% !important;
        width: 36px !important;
        height: 36px !important;
        padding: 0 !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        min-height: 0px !important;
    }

    div[data-testid="stPopover"] > button:hover {
        background-color: #222B3D !important;
    }

    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%) !important;
        color: #0B0E14 !important;
        border: none !important;
        font-weight: bold !important;
        border-radius: 50% !important;
        width: 36px !important;
        height: 36px !important;
        padding: 0 !important;
        margin: 0 !important;
        min-height: 0px !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def extract_pdf_text(uploaded_file):
    pdf_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
            
    if not text.strip():
        try:
            images = convert_from_bytes(pdf_bytes)
            ocr_text = [pytesseract.image_to_string(img) for img in images]
            text = "\n".join(ocr_text)
        except Exception as e:
            text = f"[OCR Extraction Error: {str(e)}]"
            
    return text if text.strip() else "The document contains no readable text."

def perform_web_search(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            return "\n".join([f"- {r['title']}: {r['body']} ({r['href']})" for r in results])
    except Exception as e:
        return f"Web search error: {str(e)}"

def speak_text(text):
    tts = gTTS(text=text[:300], lang='en')
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

# 3. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_context" not in st.session_state:
    st.session_state.pdf_context = ""
if "pdf_file_name" not in st.session_state:
    st.session_state.pdf_file_name = ""

# 4. Sidebar UI Configuration
with st.sidebar:
    st.markdown("<h2 style='text-align: center; letter-spacing: 0.3em; color: #E2E8F0;'>S I Y A</h2>", unsafe_allow_html=True)
    st.caption("<div style='text-align: center; color: #64748B;'>v1.0 • Hybrid Intelligence</div>", unsafe_allow_html=True)
    st.divider()

    st.markdown("##### ⚙️ Engine Settings")
    use_gemini = st.toggle("Use Gemini Cloud API", value=True)
    
    if use_gemini:
        secret_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
        gemini_api_key = st.text_input("Gemini API Key", type="password", value=secret_key, help="Key loaded automatically from secrets if set.")
        gemini_model = st.selectbox(
            "Model", 
            ["gemini-2.5-flash", "gemini-2.5-pro"]
        )
    else:
        st.info("⚡ **Mode:** Local (`llama3`) via Ollama")

    st.divider()
    st.markdown("##### 🛠️ Capabilities")
    enable_web_search = st.checkbox("Live Web Search", value=False)
    enable_voice = st.checkbox("Voice Output (TTS)", value=False)

    st.divider()
    if st.button("🗑️ Clear Chat & PDF", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pdf_context = ""
        st.session_state.pdf_file_name = ""
        st.rerun()

# 5. Header & Messages
st.markdown('<div class="brand-logo">S I Y A</div>', unsafe_allow_html=True)

USER_AVATAR = "👤"
SIYA_AVATAR = "⚡"

for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else SIYA_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Display active PDF chip above the input box if uploaded
if st.session_state.pdf_file_name:
    st.info(f"📎 Attached PDF: **{st.session_state.pdf_file_name}**")

# 6. Integrated Chat Input Bar with '+' Popover
with st.form(key="chat_form", clear_on_submit=True):
    col_plus, col_input, col_submit = st.columns([0.06, 0.88, 0.06], vertical_alignment="center")
    
    with col_plus:
        with st.popover("➕", help="Attach PDF"):
            uploaded_pdf = st.file_uploader("Upload Context PDF", type=["pdf"], key="inline_pdf")
            if uploaded_pdf:
                st.session_state.pdf_context = extract_pdf_text(uploaded_pdf)
                st.session_state.pdf_file_name = uploaded_pdf.name
                st.success(f"Attached: {uploaded_pdf.name}")

    with col_input:
        prompt_text = st.text_input(
            "Message S I Y A...",
            placeholder="Ask S I Y A anything...",
            label_visibility="collapsed"
        )

    with col_submit:
        submit_button = st.form_submit_button("➔")

# 7. Processing Form Submission
if submit_button and prompt_text:
    prompt = prompt_text
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    web_context = ""
    if enable_web_search:
        with st.status("Searching live web...", expanded=False):
            web_context = perform_web_search(prompt)

    system_instruction = "You are S I Y A, a helpful and intelligent AI assistant."
    if st.session_state.pdf_context:
        system_instruction += f"\n\nContext from attached document ({st.session_state.pdf_file_name}):\n{st.session_state.pdf_context[:4000]}"
    if web_context:
        system_instruction += f"\n\nContext from web search:\n{web_context}"

    with st.chat_message("assistant", avatar=SIYA_AVATAR):
        message_placeholder = st.empty()
        full_response = ""

        try:
            if use_gemini:
                active_api_key = gemini_api_key.strip() if gemini_api_key.strip() else st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
                
                if not active_api_key:
                    st.error("Please enter a valid Gemini API Key or configure GEMINI_API_KEY in Streamlit Secrets.")
                else:
                    client = genai.Client(api_key=active_api_key)
                    
                    gemini_contents = []
                    for m in st.session_state.messages:
                        role = "user" if m["role"] == "user" else "model"
                        gemini_contents.append(
                            types.Content(
                                role=role,
                                parts=[types.Part.from_text(text=m["content"])]
                            )
                        )

                    response_stream = client.models.generate_content_stream(
                        model=gemini_model,
                        contents=gemini_contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction
                        )
                    )

                    for chunk in response_stream:
                        if chunk.text:
                            full_response += chunk.text
                            message_placeholder.markdown(full_response + "▌")

            else:
                if ollama is None:
                    st.error("Ollama is not available in this environment.")
                else:
                    formatted_messages = [{"role": "system", "content": system_instruction}] + [
                        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
                    ]
                    stream = ollama.chat(model="llama3", messages=formatted_messages, stream=True)
                    for chunk in stream:
                        full_response += chunk['message']['content']
                        message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
            
            if enable_voice and full_response:
                with st.spinner("Generating audio..."):
                    audio_fp = speak_text(full_response)
                    st.audio(audio_fp, format="audio/mp3")

        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            full_response = "An error occurred while processing your request."

    if full_response and not full_response.startswith("An error occurred"):
        st.session_state.messages.append({"role": "assistant", "content": full_response})
