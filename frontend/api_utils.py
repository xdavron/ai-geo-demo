import json

import requests
import streamlit as st


def get_api_response(question, session_id):
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    data = {"question": question}
    if session_id:
        data["session_id"] = session_id

    try:
        response = requests.post("http://127.0.0.1:8000/api/v1/chat", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API request failed with status code {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None


def get_streaming_response(question, session_id):
    headers = {"accept": "text/event-stream"}
    data = {"question": question}

    # Assign session_id if not provided
    if session_id:
        data["session_id"] = session_id

    try:
        with requests.post('http://127.0.0.1:8000/api/v1/chat/stream', json=data, stream=True, headers=headers) as response:
            if response.status_code != 200:
                st.error(f"API request failed with {response.status_code}: {response.text}")
                return

            full_text = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8").strip()
                    if decoded_line.startswith("data:"):
                        data = decoded_line.replace("data:", "", 1).strip()
                        payload = json.loads(data)

                        if payload.get("done"):
                            break

                        chunk = payload.get("content", "")
                        full_text += chunk
                        yield chunk  # yield partial chunk for real-time UI

            return full_text

    #     # Use stream=True to process the response incrementally
    #     response = requests.post('http://127.0.0.1:8000/api/v1/chat/stream', json=data, stream=True, headers=headers)
    #     if response.status_code != 200:
    #         st.error(f"API request failed with {response.status_code}: {response.text}")
    #         return
    #
    #     client = sseclient.SSEClient(response)
    #     full_text = ""
    #     for event in client.events():
    #         data = event.data.strip()
    #         if not data:
    #             continue
    #         payload = requests.utils.json.loads(data)
    #         if payload.get("done"):
    #             break
    #         chunk = payload.get("content", "")
    #         full_text += chunk
    #         yield chunk  # Stream partial content back to Streamlit
    #
    #     return full_text
    #
    except Exception as e:
        st.error(f"Error while streaming: {str(e)}")
        return None


def upload_document(file):
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post("http://localhost:8000/upload-doc", files=files)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to upload file. Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while uploading the file: {str(e)}")
        return None

def list_documents():
    try:
        response = requests.get("http://localhost:8000/list-docs")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch document list. Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"An error occurred while fetching the document list: {str(e)}")
        return []

def delete_document(file_id):
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    data = {"file_id": file_id}

    try:
        response = requests.post("http://localhost:8000/delete-doc", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to delete document. Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while deleting the document: {str(e)}")
        return None
