import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os

# --- Environment Detection and Model Setup ---

# Load environment variables from a .env file if it exists (for local development)
load_dotenv()

# This is the key: Check for a specific environment variable set by Hugging Face Spaces
IS_ON_HUGGINGFACE = os.environ.get("SYSTEM") == "spaces"

# Import the correct libraries based on the environment
if IS_ON_HUGGINGFACE:
    from transformers import pipeline
    print("Running on Hugging Face Space. Using Transformers text-generation pipeline.")
else:
    from langchain_community.llms import Ollama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    print("Running locally. Using Ollama.")

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="AI Chatbot", layout="centered")
st.title('AI Chatbot 🤖')

# --- Session State Initialization ---
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""

# --- Model Loading and Chaining ---
@st.cache_resource
def load_model():
    """Loads the appropriate model based on the environment."""
    if IS_ON_HUGGINGFACE:
        # Load a stable text-generation pipeline. This is more reliable.
        # "HuggingFaceH4/zephyr-7b-beta" is a powerful model fine-tuned for chat.
        return pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta")
    else:
        # Load the local Ollama model using LangChain
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. Please respond to the user queries."),
                ("user", "Question: {question}")
            ]
        )
        llm = Ollama(model="phi3")
        output_parser = StrOutputParser()
        return prompt | llm | output_parser

try:
    model_pipeline = load_model()
except Exception as e:
    st.error(f"Failed to load the AI model. Error: {e}")
    st.stop()

# --- Core Application Logic ---
def generate_response(user_query, chat_history):
    """Generates a response using the appropriate model."""
    if IS_ON_HUGGINGFACE:
        # Build a prompt string with history formatted for the Zephyr model
        # The format is <|system|>\n...\n<|user|>\n...\n<|assistant|>\n...
        messages = [{"role": "system", "content": "You are a friendly chatbot who always gives helpful answers."}]
        for user_msg, ai_msg in zip(chat_history['past'], chat_history['generated']):
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": ai_msg})
        messages.append({"role": "user", "content": user_query})
        
        # Use the pipeline's chat template to format the prompt
        prompt = model_pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate the response
        outputs = model_pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        
        # Extract the generated text from the pipeline's output
        generated_text = outputs[0]["generated_text"]
        
        # The response includes the prompt, so we extract only the new part.
        response = generated_text.split("<|assistant|>")[-1].strip()
        return response
    else:
        # Logic for the local Ollama LangChain chain
        return model_pipeline.invoke({"question": user_query}).strip()

# --- Streamlit UI Components ---
def submit():
    st.session_state.user_input = st.session_state.prompt_input
    st.session_state.prompt_input = ""

st.text_input('YOU: ', key='prompt_input', on_change=submit)

if st.session_state.user_input:
    user_query = st.session_state.user_input
    st.session_state.past.append(user_query)
    with st.spinner("Thinking..."):
        chat_history = {
            'past': st.session_state.past,
            'generated': st.session_state.generated
        }
        output = generate_response(user_query, chat_history)
        st.session_state.generated.append(output)
    st.session_state.user_input = ""

# --- Display Chat History ---
chat_container = st.container()
with chat_container:
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="identicon", seed="User123")
            message(st.session_state["generated"][i], key=str(i), avatar_style="micah", seed="AI-Bot")
    else:
        st.info("Hello! I'm your helpful AI assistant. How can I help you today?")