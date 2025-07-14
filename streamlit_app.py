import streamlit as st
from streamlit_chat import message
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# --- 1. SETUP ---

# Get the Hugging Face API token from the Space's secrets
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Streamlit page configuration
st.set_page_config(page_title="AI Mentor")
st.title("AI Mentor")

# Check for the token and stop if it's not found
if not hf_token:
    st.error("Hugging Face API token not set. Please add it to your Space's secrets.")
    st.info("Go to your Space's 'Settings' > 'Secrets' and add a secret named 'HUGGINGFACEHUB_API_TOKEN'.")
    st.stop()

# --- 2. SESSION STATE INITIALIZATION ---

if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = ""

# --- 3. MODEL AND CHAIN SETUP ---

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant named AI Mentor. Be polite and concise in your responses."),
        ("user", "{question}")
    ]
)

try:
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-1.1-7b-it",
        huggingfacehub_api_token=hf_token,
        temperature=0.7,
        max_new_tokens=512
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
except Exception as e:
    st.error(f"Failed to initialize the AI model. Error: {e}")
    st.stop()


# --- 4. FUNCTIONS ---

def submit():
    st.session_state.entered_prompt = st.session_state.prompt_input
    st.session_state.prompt_input = ""

def generate_response(user_query):
    """
    Generate a response using the Hugging Face model with robust error handling.
    """
    try:
        complete_prompt = st.session_state.conversation_history + f"\nUser: {user_query}\nAI:"
        
        # This is where the StopIteration error happens
        response = chain.invoke({"question": complete_prompt})

        # Check for an empty response, which can also cause issues
        if not response or not response.strip():
            return "I'm sorry, I received an empty response from the model. Please try again."

        response = response.strip()
        st.session_state.conversation_history += f"\nUser: {user_query}\nAI: {response}"
        return response

    except Exception as e:
        # Catch the StopIteration and other errors and provide a helpful message
        error_message = f"An API error occurred: {e}. This might be due to an invalid API token or a model cold start. Please check your token and try again in a moment."
        st.error(error_message)
        return "Sorry, I couldn't get a response. Please see the error above."

# --- 5. APP LAYOUT AND LOGIC ---

st.text_input('YOU: ', key='prompt_input', on_change=submit)

if st.session_state.entered_prompt:
    user_query = st.session_state.entered_prompt
    st.session_state.past.append(user_query)

    with st.spinner("Thinking..."):
        output = generate_response(user_query)
        st.session_state.generated.append(output)

# This is your original display loop, it was not removed.
if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')