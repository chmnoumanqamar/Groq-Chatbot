import streamlit as st
from dotenv import load_dotenv
import os

from openai import OpenAI  # Groq's API is OpenAI-compatible

# ---------------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Groq Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom CSS - Dark navy + glowing cyan theme
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 85% 10%, rgba(34, 211, 238, 0.08) 0%, transparent 45%),
                radial-gradient(circle at 10% 90%, rgba(34, 211, 238, 0.05) 0%, transparent 40%),
                #060b14;
        }

        .header-card {
            padding: 26px 30px;
            margin-bottom: 24px;
            border-radius: 16px;
            background: #0b1220;
            border: 1px solid #16233a;
            display: flex;
            align-items: center;
            gap: 18px;
            box-shadow: 0 0 40px rgba(34, 211, 238, 0.06);
        }
        .header-icon {
            font-size: 1.9rem;
            width: 56px;
            height: 56px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #0d1a2e;
            border: 1px solid #22d3ee;
            box-shadow: 0 0 18px rgba(34, 211, 238, 0.35);
        }
        .header-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.7rem;
            font-weight: 700;
            color: #e8f6fb;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .header-sub {
            color: #5f7a92;
            font-size: 0.9rem;
            margin-top: 2px;
        }

        .tag-row {
            display: flex;
            gap: 8px;
            margin-top: 12px;
            flex-wrap: wrap;
        }
        .tag {
            background: #0d1a2e;
            border: 1px solid #1c3350;
            color: #7fd8ec;
            padding: 4px 12px;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }
        .tag-dot {
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #22d3ee;
            margin-right: 6px;
            box-shadow: 0 0 6px rgba(34, 211, 238, 0.8);
        }

        .bubble-row {
            display: flex;
            margin-bottom: 14px;
        }
        .bubble-row.user { justify-content: flex-end; }
        .bubble-row.assistant { justify-content: flex-start; }
        .bubble {
            max-width: 70%;
            padding: 14px 18px;
            border-radius: 12px;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        .bubble.user {
            background: #e8f6fb;
            color: #071019;
            font-weight: 500;
        }
        .bubble.assistant {
            background: #0b1220;
            color: #d7ecf4;
            border: 1px solid #16233a;
            border-left: 2px solid #22d3ee;
        }
        .bubble-tag {
            font-size: 0.65rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 6px;
            opacity: 0.55;
            font-family: 'JetBrains Mono', monospace;
        }

        section[data-testid="stSidebar"] {
            background: #04070d;
            border-right: 1px solid #16233a;
        }
        section[data-testid="stSidebar"] * { color: #b8cdd9; }
        section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
            font-family: 'JetBrains Mono', monospace;
            color: #e8f6fb;
            font-weight: 700;
        }

        .stButton>button {
            border-radius: 8px;
            border: 1px solid #1c3350;
            background: #0d1a2e;
            color: #e8f6fb;
            font-weight: 600;
        }
        .stButton>button:hover {
            background: #22d3ee;
            color: #071019;
            border-color: #22d3ee;
        }

        .stSelectbox>div>div, .stTextInput>div>div>input {
            border-radius: 8px !important;
            background: #0d1a2e !important;
            border: 1px solid #1c3350 !important;
            color: #e8f6fb !important;
        }

        [data-testid="stChatInput"] {
            border-radius: 10px;
            border: 1px solid #1c3350 !important;
            background: #0b1220;
        }

        .stSlider [data-baseweb="slider"] > div > div {
            background: #22d3ee !important;
        }

        .empty-card {
            text-align: center;
            padding: 64px 20px;
            border-radius: 14px;
            background: #0b1220;
            border: 1px dashed #1c3350;
            color: #5f7a92;
        }
        .empty-card h3 {
            color: #e8f6fb;
            font-family: 'JetBrains Mono', monospace;
        }

        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Model registry - real GroqCloud model ids
# NOTE: llama-3.3-70b-versatile is on Groq's deprecation list
# (announced June 2026). If it stops working, switch the
# default to "openai/gpt-oss-120b" or "qwen/qwen3.6-27b".
# ---------------------------------------------------------
MODEL_OPTIONS = {
    "Llama 3.3 70B": "llama-3.3-70b-versatile",
    "GPT OSS 120B": "openai/gpt-oss-120b",
    "GPT OSS 20B": "openai/gpt-oss-20b",
    "Qwen 3.6 27B": "qwen/qwen3.6-27b",
}


@st.cache_resource(show_spinner=False)
def get_client(api_key: str):
    # Groq's API speaks the same protocol as OpenAI, just a different base_url
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def stream_response(client, model_id, history, temperature):
    stream = client.chat.completions.create(
        model=model_id,
        messages=history,
        temperature=temperature,
        max_tokens=1024,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------------------------------------------------------
# API key comes only from environment - never shown in the UI.
# Get yours at: https://console.groq.com/keys
# ---------------------------------------------------------
groq_key = os.getenv("GROQ_API_KEY", "")

# ---------------------------------------------------------
# Sidebar - Settings
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## Settings")

    model_label = st.selectbox(
        "Model",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
        help="Llama 3.3 70B is a strong general-purpose text model on GroqCloud.",
    )
    model_id = MODEL_OPTIONS[model_label]

    temperature = st.slider(
        "Creativity (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.1,
    )

    st.markdown("---")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Built with GroqCloud API + Streamlit")

# ---------------------------------------------------------
# Session State - Chat History
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="header-card">
        <div class="header-icon">🤖</div>
        <div>
            <p class="header-title">Groq Chatbot</p>
            <p class="header-sub">Powered by GroqCloud</p>
            <div class="tag-row">
                <span class="tag">{model_label}</span>
                <span class="tag">Temp {temperature}</span>
                <span class="tag"><span class="tag-dot"></span>Online</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Render existing chat history as custom bubbles
# ---------------------------------------------------------
for msg in st.session_state.messages:
    role = msg["role"]
    tag = "You" if role == "user" else "Bot"
    st.markdown(
        f"""
        <div class="bubble-row {role}">
            <div class="bubble {role}">
                <div class="bubble-tag">{tag}</div>
                {msg["content"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Empty state
# ---------------------------------------------------------
if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-card">
            <h3>🤖 Say hello</h3>
            <p>Type your message below to start chatting.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Chat Input
# ---------------------------------------------------------
user_prompt = st.chat_input("Type your message here...")

if user_prompt:
    if not groq_key:
        st.error("GROQ_API_KEY not found. Add it to your .env file (or platform secrets) and restart the app.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_prompt})

    history = [{"role": "system", "content": "You are a witty and helpful AI assistant."}]
    for msg in st.session_state.messages:
        history.append({"role": msg["role"], "content": msg["content"]})

    try:
        client = get_client(groq_key)
        response_text = "".join(list(stream_response(client, model_id, history, temperature)))
    except Exception as e:
        response_text = f"Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.rerun()