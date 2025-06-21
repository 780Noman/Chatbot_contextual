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


# A more structured prompt template to guide the model better.
template = """
You are a helpful and friendly AI assistant.
Here is the chat history so far:
{history}

Now, please respond to the user's latest question.

User: {user_input}
AI:"""

prompt = ChatPromptTemplate.from_template(template)


try:
    # Connect to a reliable hosted model
    llm = HuggingFaceEndpoint(
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
    # We loop through the number of generated responses, which corresponds to completed turns.
    for i in range(len(st.session_state['generated'])):
        history += f"User: {st.session_state['past'][i]}\n"
        history += f"AI: {st.session_state['generated'][i]}\n"

    # Invoke the chain with the structured history and new input
    raw_response = chain.invoke({"history": history, "user_input": user_query}).strip()
    
    # --- THIS IS THE FIX ---
    # Some models continue generating text past their own turn.
    # We will find the next "User:" marker and split the response there,
    # ensuring we only get the AI's intended reply.
    # We search for "\nUser:" to avoid splitting on the word "user" in the middle of a sentence.
    if "\nUser:" in raw_response:
        response = raw_response.split("\nUser:")[0].strip()
    else:
        response = raw_response
    # --------------------
    
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
    chat_container = st.container()
    with chat_container:
        # We display in reverse order so newest messages are at the top
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i), avatar_style="micah", seed="AI-Bot")
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="identicon", seed="User123")
