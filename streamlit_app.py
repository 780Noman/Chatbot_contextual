import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os

# --- Environment Detection and Model Setup ---

# Load environment variables from a .env file if it exists (for local development)
load_dotenv()

# This is the key: Check for a specific environment variable set by Hugging Face Spaces
IS_ON_HUGGINGFACE = os.environ.get("SYSTEM") == "spaces"

# Import the correct libraries based on the environment
if IS_ON_HUGGINGFACE:
    from transformers import pipeline
    # --- THIS IS THE FIX ---
    # The Conversation object was moved to a more specific path.
    from transformers.pipelines.conversational import Conversation
    # --------------------
    print("Running on Hugging Face Space. Using Transformers pipeline.")
else:
    from langchain_community.llms import Ollama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    print("Running locally. Using Ollama.")

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="AI Chatbot", layout="centered")
st.title('AI Chatbot 🤖')

# --- Session State Initialization ---
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""
# For the Hugging Face pipeline, we need a conversation object
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
# For Ollama, we build a string history
if 'conversation_history_str' not in st.session_state:
    st.session_state.conversation_history_str = ""

# --- Model Loading and Chaining ---
@st.cache_resource
def load_model():
    """Loads the appropriate model based on the environment."""
    if IS_ON_HUGGINGFACE:
        # Load a stable conversational pipeline for Hugging Face Spaces
        return pipeline("conversational", model="microsoft/DialoGPT-medium")
    else:
        # Load the local Ollama model
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. Please respond to the user queries."),
                ("user", "Question: {question}")
            ]
        )
        llm = Ollama(model="phi3")
        output_parser = StrOutputParser()
        return prompt | llm | output_parser

try:
    chain_or_pipeline = load_model()
except Exception as e:
    st.error(f"Failed to load the AI model. Error: {e}")
    st.stop()

# --- Core Application Logic ---
def generate_response(user_query):
    """Generates a response using the appropriate model."""
    if IS_ON_HUGGINGFACE:
        # Logic for the Hugging Face conversational pipeline
        if st.session_state.conversation is None:
            st.session_state.conversation = Conversation()
            
        st.session_state.conversation.add_user_input(user_query)
        response_conversation = chain_or_pipeline(st.session_state.conversation, pad_token_id=50256)
        return response_conversation.generated_responses[-1]
    else:
        # Logic for the local Ollama LangChain chain
        complete_prompt = st.session_state.conversation_history_str + f"\nUser: {user_query}"
        response = chain_or_pipeline.invoke({"question": complete_prompt}).strip()
        st.session_state.conversation_history_str += f"\nUser: {user_query}\nAI: {response}"
        return response

# --- Streamlit UI Components ---
def submit():
    st.session_state.user_input = st.session_state.prompt_input
    st.session_state.prompt_input = ""

st.text_input('YOU: ', key='prompt_input', on_change=submit)

if st.session_state.user_input:
    user_query = st.session_state.user_input
    st.session_state.past.append(user_query)
    with st.spinner("Thinking..."):
        output = generate_response(user_query)
        st.session_state.generated.append(output)
    st.session_state.user_input = ""

# --- Display Chat History ---
chat_container = st.container()
with chat_container:
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="identicon", seed="User123")
            message(st.session_state["generated"][i], key=str(i), avatar_style="micah", seed="AI-Bot")
    else:
        st.info("Hello! I'm your helpful AI assistant. How can I help you today?")
