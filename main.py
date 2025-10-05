import os

import streamlit as st

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pypdf import PdfReader

os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HF_TOKEN"]


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a given PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += (
            page.extract_text() + "\n"
        )  # Add a newline for readability between pages
    return text


st.title("Ruibin Ma's Resume Assistant")

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V3.2-Exp",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        provider="auto",  # let Hugging Face choose the best provider for you
    )
)

if "messages" not in st.session_state:
    st.session_state.messages = []
    # Read the CV and add it to the messages
    CV_PATH = "ruibinma-cv.pdf"
    cv_content = extract_text_from_pdf(CV_PATH)
    print(f"CV content loaded. Length: {len(cv_content)} characters.")
    st.session_state.messages.append(
        SystemMessage(
            "You are an assistant for potential recruiters to"
            f" help understand the following CV (resume):\n\n{cv_content}\n\n"
            "Please answer questions in a honest, polite manner."
            "Do not make up facts that are not in the CV."
            "Try your best to help the candidate get a job.",
        )
    )


for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        with st.chat_message("assistant"):
            st.markdown(
                "Ruibin Ma's Resume is loaded in the background. Ask me anything!"
            )
            continue
    with st.chat_message(message.type):
        st.markdown(message.content)

if query := st.chat_input("What do you want to know about Ruibin Ma?"):
    human_msg = HumanMessage(query)
    st.session_state.messages.append(human_msg)
    with st.chat_message(human_msg.type):
        st.markdown(human_msg.content)

    response_msg = llm.invoke(st.session_state.messages)
    with st.chat_message(response_msg.type):
        response = st.write(response_msg.content)
    st.session_state.messages.append(response_msg)
