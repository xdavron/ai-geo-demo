import streamlit as st
# from sidebar import display_sidebar
from chat_interface import display_chat_interface
import os

logo_path = os.path.join(os.path.dirname(__file__), "static", "img", "LogoGFU.png")

print("Logo path:", logo_path)
print("Exists:", os.path.exists(logo_path))


st.image(logo_path, width=120)
st.title("AI-GEO")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Display the sidebar
# display_sidebar()

# Display the chat interface
display_chat_interface()
