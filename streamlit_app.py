import streamlit as st
from streamlit_chat import message
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set your LangChain API key
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Streamlit page configuration
st.set_page_config(page_title="Langchain with Ollama")
st.title('Langchain with Ollama - phi3 Model')

# Initialize session state variables
if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""  # Store the latest user input

if 'generated' not in st.session_state:
    st.session_state['generated'] = []  # Store AI generated responses

if 'past' not in st.session_state:
    st.session_state['past'] = []  # Store past user inputs

if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = ""  # Store the entire conversation history

# Define function to submit user input
def submit():
    st.session_state.entered_prompt = st.session_state.prompt_input
    st.session_state.prompt_input = ""

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

# Initialize the Ollama model (phi3)
# llm = Ollama(model="phi3")
from langchain_huggingface import HuggingFaceEndpoint

# This connects to a free, hosted model on Hugging Face
# It will automatically use your HF_TOKEN from the secrets
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.7,
    max_new_tokens=1024
)
output_parser = StrOutputParser()

# Chain the prompt, model, and output parser
chain = prompt | llm | output_parser

def generate_response(user_query):
    """
    Generate a response using the Ollama model.
    Include the entire conversation history in the prompt.
    """
    # Include conversation history in the input to maintain context
    complete_prompt = st.session_state.conversation_history + f"\nUser: {user_query}\nAI:"
    response = chain.invoke({"question": complete_prompt}).strip()
    
    # Update conversation history with the new user query and AI response
    st.session_state.conversation_history += f"\nUser: {user_query}\nAI: {response}\n"
    
    return response

# Create a text input for the user
st.text_input('YOU: ', key='prompt_input', on_change=submit)

if st.session_state.entered_prompt != "":
    user_query = st.session_state.entered_prompt
    st.session_state.past.append(user_query)

    # Generate response with context
    output = generate_response(user_query)
    st.session_state.generated.append(output)

# Display the chat history
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
