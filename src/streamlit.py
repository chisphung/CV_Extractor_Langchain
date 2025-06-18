import streamlit as st
import requests
import json
import os
import chat.output_parser as output_parser
API_URL = "http://localhost:5000"  # Change if deployed

st.set_page_config(page_title="CV Analyzer", layout="centered")
st.title("\U0001F4DA CV Upload and Candidate Search")

# --- Upload Section ---
st.header("Upload a CV or Google Drive link")
uploaded_files = st.file_uploader(
    "Upload PDF CV(s) or a folder", 
    type=["pdf"], 
    accept_multiple_files=True
)
drive_link = st.text_input("or Paste a Google Drive link")

if st.button("Upload & Extract"):
    with st.spinner("Uploading and processing..."):
        data = {"drive_link": drive_link} if drive_link else {}

        files = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                files.append(
                    ("file", (uploaded_file.name, uploaded_file.getvalue(), "application/pdf"))
                )

        try:
            if files:
                response = requests.post(f"{API_URL}/upload_cv", files=files, data=data)
            else:
                response = requests.post(f"{API_URL}/upload_cv", data=data)

            result = response.json()
            extracted = result.get("extracted", [])
            st.session_state["extracted_candidates"] = extracted
            st.session_state["last_export_path"] = None  # Reset previous export path

            st.success("CV processed successfully!")
            st.json(result)

        except Exception as e:
            st.error(f"Failed to upload: {e}")

# --- Export Section ---
st.header("Export Candidates")
export_dir = st.text_input("Export directory", "./exported_candidates")

if st.button("Export Candidates"):
    if "extracted_candidates" not in st.session_state or not st.session_state["extracted_candidates"]:
        st.error("Please upload and extract CVs first.")
    else:
        with st.spinner("Exporting candidates..."):
            try:
                export_payload = {
                    "data": output_parser.json_clean(st.session_state["extracted_candidates"]),
                    "outdir": export_dir
                }
                response = requests.post(f"{API_URL}/export_candidates", json=export_payload)
                export_result = response.json()

                st.session_state["last_export_path"] = export_result["file_path"]

                st.success(f"Candidates exported successfully to {export_result['file_path']}")
                st.json(export_result)

            except Exception as e:
                st.error(f"Export failed: {e}")


# --- Search Section ---
st.header("Search Candidates")
search_query = st.text_input("Search by skills, job description, etc.")
if st.button("Search") and search_query:
    with st.spinner("Searching candidates..."):
        try:
            response = requests.post(f"{API_URL}/search_candidates", json={"query": search_query})
            result = response.json()
            st.success(f"Found {len(result['matches'])} match(es)")
            for i, match in enumerate(result['matches'], 1):
                st.subheader(f"Match #{i}")
                st.write(match['text'])
                st.caption(f"Metadata: {match['metadata']}")
        except Exception as e:
            st.error(f"Search failed: {e}")

# --- Q&A Section ---
st.header("Ask a Question")
question = st.text_input("Enter your question")
if st.button("Ask") and question:
    with st.spinner("Getting answer..."):
        try:
            response = requests.post(f"{API_URL}/generative_ai", json={"question": question})
            answer = response.json()["answer"]
            st.success("Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"Question failed: {e}")
