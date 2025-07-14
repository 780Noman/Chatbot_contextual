import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# --- 1. SETUP ---

# Set the title for your app
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title('AI Assistant')

# Check for the Hugging Face API token in the Space's secrets
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
    # Connect to a Hugging Face hosted model using its repository ID
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        huggingfacehub_api_token=hf_token, # Pass the token
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

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- 4. USER INPUT AND RESPONSE GENERATION ---

# Accept user input
if user_prompt := st.chat_input("What can I help you with?"):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Build the history string
                history = "\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages]
                )

                # Get the AI response
                response = chain.invoke({"history": history, "user_input": user_prompt})
                st.markdown(response)

                # Add AI response to session state
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"An error occurred: {e}")