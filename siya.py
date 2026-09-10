import os
import io
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
from gtts import gTTS
from duckduckgo_search import DDGS
from PIL import Image

# Safe import for python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# OCR Imports for Scanned/Image-based PDFs
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# Safe import for Ollama
try:
    import ollama
except ImportError:
    ollama = None

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
        font-size: 3.2rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.3em !important;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-align: center !important;
        margin: 10px auto 30px auto !important;
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

    .stForm {
        background-color: #161C28 !important;
        border: 1px solid #2E384E !important;
        border-radius: 28px !important;
        padding: 4px 12px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        margin-top: 20px !important;
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
            
    if not text.strip() and HAS_OCR:
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
    phonetic_text = text.replace("S I Y A", "Sih-yah").replace("SIYA", "Sih-yah").replace("Siya", "Sih-yah")
    
    tts = gTTS(text=phonetic_text[:300], lang='en')
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

def is_image_generation_request(prompt):
    triggers = ["generate an image", "create an image", "draw an image", "generate image", "create image", "draw ", "make an image", "picture of"]
    return any(trigger in prompt.lower() for trigger in triggers)

# 3. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "attached_file" not in st.session_state:
    st.session_state.attached_file = None
if "pdf_context" not in st.session_state:
    st.session_state.pdf_context = ""
if "insights" not in st.session_state:
    st.session_state.insights = {
        "thumbs_up": 0,
        "thumbs_down": 0,
        "total_queries": 0,
        "history": []
    }

# 4. Sidebar UI Configuration
with st.sidebar:
    st.markdown("<h2 style='text-align: center; letter-spacing: 0.3em; color: #E2E8F0;'>S I Y A</h2>", unsafe_allow_html=True)
    st.caption("<div style='text-align: center; color: #64748B;'>v1.0 • Hybrid Intelligence</div>", unsafe_allow_html=True)
    st.divider()

    st.markdown("##### ⚙️ Engine Settings")
    use_gemini = st.toggle("Use Gemini Cloud API", value=True)
    
    if use_gemini:
        gemini_model = st.selectbox(
            "Model", 
            ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        )
        st.caption("🟢 Connected via Server Key")
    else:
        st.info("⚡ **Mode:** Local (`llama3`) via Ollama")

    st.divider()
    st.markdown("##### 🛠️ Capabilities")
    enable_web_search = st.checkbox("Live Web Search", value=False)
    enable_voice = st.checkbox("Voice Output (TTS)", value=False)

    st.divider()
    st.markdown("##### 📊 SIYA Insights & Analytics")
    st.metric("Total Prompts Processed", st.session_state.insights["total_queries"])
    
    # Feedback Ratio Bar Chart
    feedback_data = {
        "Positive (👍)": st.session_state.insights["thumbs_up"],
        "Negative (👎)": st.session_state.insights["thumbs_down"]
    }
    st.caption("Feedback Distribution")
    st.bar_chart(feedback_data, height=160)

    # Query Growth Line Chart
    if st.session_state.insights["history"]:
        st.caption("Prompts Processed Over Time")
        st.line_chart(st.session_state.insights["history"], height=160)
    else:
        st.info("Ask SIYA questions to populate the usage graph!")

    st.divider()
    if st.button("🗑️ Clear Chat & Files", use_container_width=True):
        st.session_state.messages = []
        st.session_state.attached_file = None
        st.session_state.pdf_context = ""
        st.rerun()

# 5. Header Title
st.markdown('<div class="brand-logo">S I Y A</div>', unsafe_allow_html=True)

USER_AVATAR = "👤"
SIYA_AVATAR = "⚡"

# 6. Render Active Chat History & Reviews Widget
for idx, message in enumerate(st.session_state.messages):
    avatar = USER_AVATAR if message["role"] == "user" else SIYA_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        if message.get("type") == "image":
            st.image(message["content"], caption=message.get("caption", "Generated Image"), use_container_width=True)
        else:
            st.markdown(message["content"])
            if message.get("audio"):
                st.audio(message["audio"], format="audio/mp3", autoplay=False)

        # Review/Feedback Buttons for Assistant Responses
        if message["role"] == "assistant" and message.get("type") != "image":
            col_like, col_dislike, _ = st.columns([0.08, 0.08, 0.84])
            if col_like.button("👍", key=f"like_{idx}"):
                st.session_state.insights["thumbs_up"] += 1
                st.toast("Thanks for your feedback!")
                st.rerun()
            if col_dislike.button("👎", key=f"dislike_{idx}"):
                st.session_state.insights["thumbs_down"] += 1
                st.toast("Thanks! We'll work on improving SIYA.")
                st.rerun()

# Display attached file indicator if active
if st.session_state.attached_file:
    st.info(f"📎 Attached File: **{st.session_state.attached_file['name']}** ({st.session_state.attached_file['type']})")

# 7. Render Multimodal Input Form at the BOTTOM
with st.form(key="chat_form", clear_on_submit=True):
    col_plus, col_input, col_submit = st.columns([0.06, 0.88, 0.06], vertical_alignment="center")
    
    with col_plus:
        with st.popover("➕", help="Attach Images, Videos, Audio, or PDFs"):
            uploaded_file = st.file_uploader(
                "Upload Media or File", 
                type=["pdf", "png", "jpg", "jpeg", "webp", "mp4", "mov", "mp3", "wav"], 
                key="inline_file"
            )
            if uploaded_file:
                file_bytes = uploaded_file.read()
                uploaded_file.seek(0)
                
                st.session_state.attached_file = {
                    "name": uploaded_file.name,
                    "type": uploaded_file.type,
                    "bytes": file_bytes
                }
                
                if uploaded_file.type == "application/pdf":
                    st.session_state.pdf_context = extract_pdf_text(uploaded_file)
                st.success(f"Attached: {uploaded_file.name}")

    with col_input:
        prompt_text = st.text_input(
            "Message S I Y A...",
            placeholder="Ask S I Y A anything or request an image (e.g., 'generate an image of a cat')...",
            label_visibility="collapsed"
        )

    with col_submit:
        submit_button = st.form_submit_button("➔")

# 8. Process Submission After Input Form
if submit_button and prompt_text.strip():
    user_prompt = prompt_text.strip()
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    # Update Analytics Data
    st.session_state.insights["total_queries"] += 1
    st.session_state.insights["history"].append(st.session_state.insights["total_queries"])

    # Retrieve API key safely across Streamlit Cloud, .env, and OS environment
    active_api_key = None
    try:
        active_api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        active_api_key = os.environ.get("GEMINI_API_KEY", "")

    # --- IMAGE GENERATION BRANCH ---
    if is_image_generation_request(user_prompt) and use_gemini:
        if not active_api_key:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "⚠️ **API Key Missing:** Please add `GEMINI_API_KEY` to Streamlit Cloud Secrets or your `.env` file."
            })
        else:
            try:
                client = genai.Client(api_key=active_api_key)
                result = client.models.generate_images(
                    model="imagen-3.0-generate-002",
                    prompt=user_prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        aspect_ratio="1:1"
                    )
                )
                
                generated_image = result.generated_images[0]
                image_bytes = generated_image.image.image_bytes

                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "image",
                    "content": image_bytes,
                    "caption": user_prompt
                })
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    error_msg = "⚠️ Image Generation Rate Limit Exceeded. Please try again in a minute."
                else:
                    error_msg = f"Failed to generate image: {error_str}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # --- TEXT / MULTIMODAL CHAT BRANCH ---
    else:
        web_context = ""
        if enable_web_search:
            with st.status("Searching live web...", expanded=False):
                web_context = perform_web_search(user_prompt)

        system_instruction = "You are S I Y A, a helpful and intelligent AI assistant."
        if st.session_state.pdf_context:
            system_instruction += f"\n\nContext from document ({st.session_state.attached_file['name']}):\n{st.session_state.pdf_context[:4000]}"
        if web_context:
            system_instruction += f"\n\nContext from web search:\n{web_context}"

        try:
            if use_gemini:
                if not active_api_key:
                    full_response = "⚠️ **API Key Missing:** Please add `GEMINI_API_KEY` to Streamlit Cloud Secrets or your `.env` file."
                else:
                    client = genai.Client(api_key=active_api_key)
                    
                    gemini_contents = []
                    for m in st.session_state.messages:
                        if m.get("type") == "image":
                            continue
                        role = "user" if m["role"] == "user" else "model"
                        gemini_contents.append(
                            types.Content(
                                role=role,
                                parts=[types.Part.from_text(text=m["content"])]
                            )
                        )

                    # Append raw bytes for media files
                    if st.session_state.attached_file and st.session_state.attached_file["type"] != "application/pdf":
                        file_data = st.session_state.attached_file
                        media_part = types.Part.from_bytes(
                            data=file_data["bytes"],
                            mime_type=file_data["type"]
                        )
                        gemini_contents[-1].parts.append(media_part)

                    # Direct call with safe standard model fallback
                    try:
                        response = client.models.generate_content(
                            model=gemini_model,
                            contents=gemini_contents,
                            config=types.GenerateContentConfig(
                                system_instruction=system_instruction
                            )
                        )
                        full_response = response.text if response.text else "No response generated."
                    except Exception as primary_err:
                        err_text = str(primary_err)
                        if "404" in err_text or "NOT_FOUND" in err_text or "429" in err_text or "RESOURCE_EXHAUSTED" in err_text:
                            st.toast("Primary model unavailable or quota reached. Switching to gemini-1.5-flash...")
                            fallback_resp = client.models.generate_content(
                                model="gemini-1.5-flash",
                                contents=gemini_contents,
                                config=types.GenerateContentConfig(
                                    system_instruction=system_instruction
                                )
                            )
                            full_response = fallback_resp.text if fallback_resp.text else "No response generated."
                        else:
                            raise primary_err

            else:
                if ollama is None:
                    full_response = "Ollama module is not installed locally."
                else:
                    formatted_messages = [{"role": "system", "content": system_instruction}] + [
                        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages if m.get("type") != "image"
                    ]
                    response = ollama.chat(model="llama3", messages=formatted_messages)
                    full_response = response['message']['content']

        except Exception as e:
            full_response = f"Error generating response: {str(e)}"

        assistant_msg = {"role": "assistant", "content": full_response}

        # Generate Audio if Voice Output is enabled in sidebar
        if enable_voice and full_response and not full_response.startswith("Error") and not full_response.startswith("⚠️"):
            try:
                audio_fp = speak_text(full_response)
                assistant_msg["audio"] = audio_fp.getvalue()
            except Exception as e:
                st.warning(f"Voice generation failed: {str(e)}")

        st.session_state.messages.append(assistant_msg)

    st.rerun()