import os

import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pypdf import PdfReader

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


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
        text += page.extract_text() + "\n"
    return text


st.title("Ruibin Ma's Resume Assistant (马睿斌的AI简历助手)")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=512,
    timeout=None,
    max_retries=2,
    api_key=st.secrets["OPENAI_API_KEY"],
)

if "messages" not in st.session_state:
    st.session_state.messages = []
    # Read the CV and add it to the messages
    CV_PATH = "maruibin-cv-chinese.pdf"
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
                "Ruibin Ma's Resume is loaded in the background. Ask me anything! 简历已读取，问我问题吧！"
            )
            continue
    with st.chat_message(message.type):
        st.markdown(message.content)

if query := st.chat_input("What do you want to know about Ruibin Ma?"):
    human_msg = HumanMessage(query)
    st.session_state.messages.append(human_msg)
    with st.chat_message(human_msg.type):
        st.markdown(human_msg.content)

    msg_stream = llm.stream(st.session_state.messages)
    full_response_content = []

    def convert_to_str_stream(msg_stream):
        for msg in msg_stream:
            full_response_content.append(msg.content)
            yield msg.content

    stream = convert_to_str_stream(msg_stream)

    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    if full_response_content:
        response_msg = AIMessage("".join(full_response_content))
        st.session_state.messages.append(response_msg)
