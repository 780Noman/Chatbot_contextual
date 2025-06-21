import streamlit as st
from streamlit_chat import message
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set your LangChain API key if you have one
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="AI Chatbot", layout="wide")
st.title('AI Chatbot - 🤖 Hosted Model')

# --- THIS IS THE FIX: CSS Injection ---
# This CSS will stick the text input area to the bottom of the screen
# and add padding to the main chat area to prevent the last message
# from being hidden behind the input bar.
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 5rem; /* Space for the input bar */
}
/* Selects the container of the st.text_input */
[data-testid="stTextInput"] {
  position: fixed;
  bottom: 0;
  width: 100%;
  left: 0;
  background-color: #0E1117; /* Match Streamlit's dark theme */
  padding: 1rem 1rem 1rem 1rem;
  border-top: 1px solid #31333F;
  z-index: 999;
}
</style>
""", unsafe_allow_html=True)
# --------------------


# --- Session State Initialization ---
if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

# --- LangChain and Model Setup ---
template = """
You are a helpful and friendly AI assistant.
Here is the chat history so far:
{history}

Now, please respond to the user's latest question.
User: {user_input}
AI:"""
prompt = ChatPromptTemplate.from_template(template)

try:
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        temperature=0.7, max_new_tokens=512,
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
except Exception as e:
    st.error(f"Failed to load the AI model. Error: {e}")
    st.stop()

# --- Core Application Logic ---
def generate_response(user_query):
    history = ""
    for i in range(len(st.session_state['generated'])):
        history += f"User: {st.session_state['past'][i]}\n"
        history += f"AI: {st.session_state['generated'][i]}\n"
    raw_response = chain.invoke({"history": history, "user_input": user_query}).strip()
    if "\nUser:" in raw_response:
        response = raw_response.split("\nUser:")[0].strip()
    else:
        response = raw_response
    return response

# --- UI Layout ---
# First, display the chat history
chat_container = st.container()
with chat_container:
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="identicon", seed="User123")
            message(st.session_state["generated"][i], key=str(i), avatar_style="micah", seed="AI-Bot")

# Then, create the text input bar which will be fixed to the bottom by our CSS
def submit():
    st.session_state.entered_prompt = st.session_state.prompt_input
    st.session_state.prompt_input = ""

st.text_input('YOU: ', key='prompt_input', on_change=submit)

# Finally, process the input if there is any
if st.session_state.entered_prompt != "":
    user_query = st.session_state.entered_prompt
    st.session_state.past.append(user_query)
    with st.spinner("Thinking..."):
        output = generate_response(user_query)
        st.session_state.generated.append(output)
    # Rerun the script to immediately display the new messages
    st.rerun()
