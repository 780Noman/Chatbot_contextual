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

# Initialize session state variables
if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""  # Store the latest user input

if 'generated' not in st.session_state:
    st.session_state['generated'] = []  # Store AI generated responses

if 'past' not in st.session_state:
    st.session_state['past'] = []  # Store past user inputs

if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = ""  # Store the entire conversation history

# --- 3. MODEL AND CHAIN SETUP ---

# Prompt template (the 'system' part acts as a base instruction)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant named AI Mentor. Be polite and concise in your responses."),
        ("user", "{question}")
    ]
)

# Initialize the HuggingFaceEndpoint
try:
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
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

# Define function to submit user input
def submit():
    st.session_state.entered_prompt = st.session_state.prompt_input
    st.session_state.prompt_input = ""

def generate_response(user_query):
    """
    Generate a response using the Hugging Face model.
    """
    # Use the conversation history to maintain context
    complete_prompt = st.session_state.conversation_history + f"\nUser: {user_query}\nAI:"
    response = chain.invoke({"question": complete_prompt}).strip()

    # Update conversation history with the new exchange
    st.session_state.conversation_history += f"\nUser: {user_query}\nAI: {response}"

    return response

# --- 5. APP LAYOUT AND LOGIC ---

# Create a text input for the user
st.text_input('YOU: ', key='prompt_input', on_change=submit)

if st.session_state.entered_prompt:
    user_query = st.session_state.entered_prompt
    st.session_state.past.append(user_query)

    # Generate response
    with st.spinner("Thinking..."):
        output = generate_response(user_query)
        st.session_state.generated.append(output)

# Display the chat history in reverse order
if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')