import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# --- 1. SETUP ---

# Set the title for your app
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title('AI Assistant')

# Check for the Hugging Face API token in Streamlit's secrets management
# This is the standard way for apps deployed on Hugging Face Spaces
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    st.error("Hugging Face API token is not set. Please add it to your Space's secrets.")
    st.info("Go to your Space's 'Settings' > 'Secrets' and add a secret named 'HUGGINGFACEHUB_API_TOKEN'.")
    st.stop()

# --- 2. MODEL AND CHAIN INITIALIZATION ---

# A structured prompt template to guide the model
template = """
You are a helpful and friendly AI assistant.

Current conversation:
{history}

User: {user_input}
AI:"""
prompt_template = ChatPromptTemplate.from_template(template)

try:
    # Connect to a reliable hosted model from Hugging Face
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        huggingfacehub_api_token=hf_token, # Explicitly pass the token
        temperature=0.7,
        max_new_tokens=512,
    )

    # Define the output parser and chain the components together
    output_parser = StrOutputParser()
    chain = prompt_template | llm | output_parser

except Exception as e:
    st.error(f"Failed to load the AI model. Error: {e}")
    st.stop()


# --- 3. CHAT HISTORY MANAGEMENT ---

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior messages on rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- 4. USER INPUT AND RESPONSE GENERATION ---

# Accept user input using the modern chat input widget
if user_prompt := st.chat_input("What can I help you with?"):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Build the history string from session state
                history = "\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages]
                )

                # Invoke the chain to get a response
                response = chain.invoke({"history": history, "user_input": user_prompt})
                st.markdown(response)

                # THE FIX IS HERE: Add AI response to session state ONLY if successful
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                # Display a user-friendly error message in the chat
                error_message = f"Sorry, I ran into a problem: {e}"
                st.error(error_message)