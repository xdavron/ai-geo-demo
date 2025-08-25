import streamlit as st
from api_utils import get_api_response, get_streaming_response


def display_chat_interface():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if prompt := st.chat_input("Ask me about meals..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response_text = st.write_stream(
                    get_streaming_response(prompt, st.session_state.session_id)
                )

            # Save assistant response in chat history
            st.session_state.messages.append({"role": "assistant", "content": response_text})

        with st.expander("Details"):
            st.subheader("Generated Answer")
            st.code(response_text)
            st.subheader("Session ID")
            st.code(st.session_state.session_id)

    # # Display chat history
    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])
    #
    # # Handle new user input
    # if prompt := st.chat_input("Query:"):
    #     st.session_state.messages.append({"role": "user", "content": prompt})
    #     with st.chat_message("user"):
    #         st.markdown(prompt)
    #
    #     # Get API response
    #     with st.spinner("Generating response..."):
    #         response = get_api_response(prompt, st.session_state.session_id)
    #
    #         if response:
    #             st.session_state.session_id = response.get('session_id')
    #             st.session_state.messages.append({"role": "assistant", "content": response['answer']})
    #
    #             with st.chat_message("assistant"):
    #                 st.markdown(response['answer'])
    #
    #             with st.expander("Details"):
    #                 st.subheader("Generated Answer")
    #                 st.code(response['answer'])
    #                 st.subheader("Session ID")
    #                 st.code(response['session_id'])
    #         else:
    #             st.error("Failed to get a response from the API. Please try again.")
