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

# Streamlit page configuration
st.set_page_config(page_title="AI Chatbot")
st.title('AI Chatbot - 🤖 Hosted Model')

# Initialize session state variables
if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = ""

# Define function to submit user input
def submit():
    st.session_state.entered_prompt = st.session_state.prompt_input
    st.session_state.prompt_input = ""

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries in a friendly manner."),
        ("user", "Question: {question}")
    ]
)

try:
    # --- THIS IS THE FIX ---
    # We are using a different, highly compatible model that is very reliable on Hugging Face Spaces.
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        temperature=0.7,
        max_new_tokens=512,
        top_k=50,
        top_p=0.95,
    )
    # --------------------
    
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
except Exception as e:
    st.error(f"Failed to load the AI model. Error: {e}")
    st.stop()


def generate_response(user_query):
    """
    Generate a response using the hosted Hugging Face model.
    """
    # Include conversation history in the input to maintain context
    complete_prompt = st.session_state.conversation_history + f"\nUser: {user_query}"
    response = chain.invoke({"question": complete_prompt}).strip()
    
    # Update conversation history with the new user query and AI response
    st.session_state.conversation_history += f"\nUser: {user_query}\nAI: {response}"
    
    return response

# Create a text input for the user
st.text_input('YOU: ', key='prompt_input', on_change=submit)

if st.session_state.entered_prompt != "":
    user_query = st.session_state.entered_prompt
    st.session_state.past.append(user_query)

    # Generate response with context
    with st.spinner("Thinking..."):
        output = generate_response(user_query)
        st.session_state.generated.append(output)

# Display the chat history
if st.session_state['generated']:
    chat_container = st.container()
    with chat_container:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i), avatar_style="micah", seed="AI-Bot")
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="identicon", seed="User123")

