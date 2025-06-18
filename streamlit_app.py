import streamlit as st
from streamlit_chat import message
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables from a .env file if it exists
load_dotenv()

# Set your LangChain API key if you have one (optional)
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Langchain with Hosted LLM", layout="centered")
st.title('AI Chatbot - 🤖 ')

# --- Session State Initialization ---
# Ensures variables persist across user interactions
if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = ""

# --- LangChain and Hugging Face Setup ---
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

try:
    # This connects to a free, hosted model on Hugging Face
    # We are using a highly compatible and powerful instruction-tuned model.
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.7,
        max_new_tokens=1024
    )
    
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
except Exception as e:
    st.error(f"Failed to initialize the language model. Error: {e}")
    st.stop()

# --- Core Application Logic ---
def generate_response(user_query):
    """
    Generates a response using the hosted Hugging Face model.
    """
    complete_prompt = st.session_state.conversation_history + f"\nUser: {user_query}"
    response = chain.invoke({"question": complete_prompt}).strip()
    st.session_state.conversation_history += f"\nUser: {user_query}\nAI: {response}"
    return response

# --- Streamlit UI Components ---
def submit():
    """
    Callback function to handle user input submission.
    """
    st.session_state.entered_prompt = st.session_state.prompt_input
    st.session_state.prompt_input = ""

# User input text box at the bottom
st.text_input('YOU: ', key='prompt_input', on_change=submit)

# Process user input and generate response
if st.session_state.entered_prompt:
    user_query = st.session_state.entered_prompt
    st.session_state.past.append(user_query)
    with st.spinner("Thinking..."):
        output = generate_response(user_query)
        st.session_state.generated.append(output)

# --- Display Chat History with Avatars ---
chat_container = st.container()

with chat_container:
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            # Display AI response with a robot avatar
            message(
                st.session_state["generated"][i],
                key=str(i),
                avatar_style="bottts-neutral",
                seed="AI-Bot"
            )
            # Display user's prompt with a human avatar
            message(
                st.session_state['past'][i],
                is_user=True,
                key=str(i) + '_user',
                avatar_style="adventurer-neutral",
                seed="User123"
            )
    else:
        # Display a welcome message and image when the chat is empty
        st.info("Hello! I'm your helpful AI assistant. How can I help you today?")
