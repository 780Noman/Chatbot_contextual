import streamlit as st
from streamlit_chat import message
# IMPORTANT: Use HuggingFaceChat for conversational models
from langchain_huggingface import HuggingFaceChat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set your Hugging Face API token if you have one
# Ensure HUGGINGFACEHUB_API_TOKEN is set in your Space's secrets
if os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Streamlit page configuration
st.set_page_config(page_title="AI Chatbot")
st.title('AI Assistant - 🤖')

# Initialize session state variables
if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

# A more structured prompt template to guide the model better.
template = """You are a helpful and friendly AI assistant.
Here is the chat history so far:
{history}
Now, please respond to the user's latest question.
User: {user_input}
AI:"""
prompt = ChatPromptTemplate.from_template(template)

try:
    # THE FIX IS HERE: Use HuggingFaceChat instead of HuggingFaceEndpoint
    llm = HuggingFaceChat(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        temperature=0.7,
        max_new_tokens=512,
    )

    output_parser = StrOutputParser()
    # Chain the components together
    chain = prompt | llm | output_parser

except Exception as e:
    st.error(f"Failed to load the AI model. Error: {e}")
    st.stop()

def generate_response(user_query):
    """
    Generate a response using the hosted Hugging Face model, including conversation history.
    """
    # Build the history string from session state
    history = ""
    for i in range(len(st.session_state['generated'])):
        history += f"User: {st.session_state['past'][i]}\n"
        history += f"AI: {st.session_state['generated'][i]}\n"

    # Invoke the chain with the structured history and new input
    response = chain.invoke({"history": history, "user_input": user_query}).strip()
    return response

# Define function to submit user input
def submit():
    st.session_state.entered_prompt = st.session_state.prompt_input
    st.session_state.prompt_input = ""

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
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))