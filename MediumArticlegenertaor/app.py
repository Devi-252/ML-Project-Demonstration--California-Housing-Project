import os
import streamlit as st
from langchain.llms import OpenAI
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("OPENAI_API_KEY")

st.title("Medicum Article Generator")

# Check if the API key was successfully loaded
if api_key:
    # Now you can proceed with the rest of your app, as the key is available
    topic = st.text_input("Input your topic of interest")

    # Example: Initializing the LangChain LLM with the API key
    llm = OpenAI(openai_api_key=api_key, temperature=0.9)

    # Add a button or check for the topic to trigger the generation
    if topic:
        st.write(f"Generating article for: {topic}")

        # --- CODE TO CALL THE LLM AND GET THE RESPONSE ---
        with st.spinner("Generating..."):
            try:
                # Use the __call__ method to invoke the LLM with the topic
                response = llm(f"Write a comprehensive article about {topic}")
                # Display the generated text in the Streamlit app
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")

else:
    # Display the error message in the Streamlit app
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

