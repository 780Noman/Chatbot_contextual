import streamlit as st
from streamlit_chat import message
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Load environment variables from a .env file if it exists
load_dotenv()

# This will automatically use the HUGGINGFACEHUB_API_TOKEN from your Space secrets
# We do not need a separate LangChain API key for this setup.

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="AI Chatbot", layout="centered")
st.title('AI Chatbot - 🤖 ')

# --- Session State Initialization ---
# Ensures variables persist across user interactions
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""

# --- Hugging Face Inference Client Setup ---
try:
    # Initialize the client. It will use the HF_TOKEN from secrets.
    client = InferenceClient()
    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
except Exception as e:
    st.error(f"Failed to initialize the Hugging Face Inference Client. Error: {e}")
    st.stop()

# --- Core Application Logic ---
def generate_response(user_query, chat_history):
    """
    Generates a response using the hosted Hugging Face model directly.
    """
    # Build the prompt with conversation history using the model's required format
    conversation = ""
    for user_msg, ai_msg in zip(chat_history['past'], chat_history['generated']):
        conversation += f"[INST] {user_msg} [/INST]{ai_msg} "
    
    # Add the latest user query
    prompt = f"{conversation}[INST] {user_query} [/INST]"

    try:
        # Call the text_generation endpoint directly
        response_stream = client.text_generation(
            model=MODEL_ID,
            prompt=prompt,
            max_new_tokens=1024,
            temperature=0.7,
            stream=True # Use streaming for better user experience
        )
        
        # Stream the response back to the UI
        full_response = ""
        for token in response_stream:
            full_response += token
        return full_response.strip()

    except Exception as e:
        st.error(f"Sorry, I encountered an error: {e}")
        return "I am unable to respond at the moment. Please try again later."


# --- Streamlit UI Components ---
# This function is called when the user presses Enter in the text input
def submit():
    st.session_state.user_input = st.session_state.prompt_input
    st.session_state.prompt_input = ""

# User input text box at the bottom
st.text_input('YOU: ', key='prompt_input', on_change=submit)

# Process user input and generate response
if st.session_state.user_input:
    # Get the user's query
    user_query = st.session_state.user_input
    
    # Store the user's query and generate the AI's response
    st.session_state.past.append(user_query)
    with st.spinner("Thinking..."):
        # Pass the current chat history for context
        chat_history = {
            'past': st.session_state.past,
            'generated': st.session_state.generated
        }
        output = generate_response(user_query, chat_history)
        st.session_state.generated.append(output)
    
    # Clear the input so it doesn't re-run
    st.session_state.user_input = ""

# --- Display Chat History with Avatars ---
chat_container = st.container()

with chat_container:
    if st.session_state['generated']:
        # Display the chat history in chronological order (oldest to newest)
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="identicon", seed="User123")
            message(st.session_state["generated"][i], key=str(i), avatar_style="micah", seed="AI-Bot")
    else:
        # Display a welcome message and image when the chat is empty
        st.info("Hello! I'm your helpful AI assistant. How can I help you today?")
