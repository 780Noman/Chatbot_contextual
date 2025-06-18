import streamlit as st
from streamlit_chat import message
from transformers import pipeline
# --- THIS IS THE FIX ---
# The Conversation object was moved to a more specific path in newer versions of the transformers library.
# We are updating the import statement to its new, correct location.
from transformers.pipelines.conversational import Conversation
# --------------------
from dotenv import load_dotenv
import os

# Load environment variables from a .env file if it exists
load_dotenv()

# This will automatically use the HUGGINGFACEHUB_API_TOKEN from your Space secrets.
# We no longer need a separate LangChain API key.

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="AI Chatbot", layout="centered")
st.title('AI Chatbot - 🤖')

# --- Session State Initialization ---
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""

# Initialize the conversation object in session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = None

# --- Hugging Face Pipeline Setup ---
try:
    # We use st.cache_resource to load the model only once
    @st.cache_resource
    def load_chatbot_pipeline():
        """Loads the conversational pipeline from Hugging Face."""
        return pipeline("conversational", model="microsoft/DialoGPT-medium")
    
    chatbot = load_chatbot_pipeline()
    
    # We only create a new conversation object if one doesn't already exist
    if st.session_state.conversation is None:
        st.session_state.conversation = Conversation()
    
    print("Conversational pipeline loaded successfully!")
except Exception as e:
    st.error(f"Failed to initialize the Hugging Face pipeline. Error: {e}")
    st.stop()

# --- Core Application Logic ---
def generate_response(user_query):
    """
    Generates a response using the conversational pipeline.
    """
    # Add the user's new message to our conversation object
    st.session_state.conversation.add_user_input(user_query)
    
    # Get the model's response by passing the entire conversation
    # The `pad_token_id` is set to prevent a common warning with this model
    response_conversation = chatbot(st.session_state.conversation, pad_token_id=50256)
    
    # The model's reply is the last generated response in the conversation
    ai_response = response_conversation.generated_responses[-1]
    
    return ai_response

# --- Streamlit UI Components ---
def submit():
    st.session_state.user_input = st.session_state.prompt_input
    st.session_state.prompt_input = ""

# User input text box
st.text_input('YOU: ', key='prompt_input', on_change=submit)

# Process user input and generate response
if st.session_state.user_input:
    user_query = st.session_state.user_input
    st.session_state.past.append(user_query)
    
    with st.spinner("Thinking..."):
        output = generate_response(user_query)
        st.session_state.generated.append(output)
    
    st.session_state.user_input = "" # Clear input to prevent re-running

# --- Display Chat History ---
chat_container = st.container()

with chat_container:
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="identicon", seed="User123")
            message(st.session_state["generated"][i], key=str(i), avatar_style="micah", seed="AI-Bot")
    else:
        st.info("Hello! I'm your helpful AI assistant. How can I help you today?")
