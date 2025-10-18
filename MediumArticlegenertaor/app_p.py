import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAI

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("OPENAI_API_KEY")

st.title("📝 Medium Article Generator")

# Check if API key exists
if not api_key:
    st.error("❌ OpenAI API key not found. Please add it to your .env file as OPENAI_API_KEY.")
else:
    # Inputs
    topic = st.text_input("Enter your topic of interest:")
    language = st.selectbox("Select language:", ["English", "French", "Spanish", "German", "Hindi"])
    mode = st.radio("Choose what to generate:", ["Article", "Title Only"])

    if st.button("Generate"):
        if not topic.strip():
            st.warning("⚠️ Please enter a topic first.")
        else:
            try:
                # Initialize model
                llm = OpenAI(openai_api_key=api_key, temperature=0.9)

                # Create prompt based on mode
                if mode == "Article":
                    prompt = f"Write a detailed Medium-style article about '{topic}' in {language} language."
                else:
                    prompt = f"Give me a catchy Medium article title about '{topic}' in {language} language."

                # Generate response
                with st.spinner("✨ Generating..."):
                    response = llm(prompt)

                # Display result
                if mode == "Article":
                    st.subheader("📰 Generated Article")
                else:
                    st.subheader("🎯 Suggested Title")

                st.write(response.strip())

            except Exception as e:
                st.error(f"An error occurred: {e}")
