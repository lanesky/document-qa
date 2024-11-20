__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_community.document_loaders.csv_loader import CSVLoader
from pathlib import Path
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
import os
# from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# Show title and description.
st.title("📄 OMOP RxNorm Mapping")
st.write(
    "薬剤名を入力すると、その薬剤に対応するRxNormコードおよび名前を取得できます。ただし、このアプリはデモ用途のみで、サポート対象の薬剤は限られています（「ベンゾイル酸」「Hyoscyamine Sulfate」などは使用可能です）。"

    # "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

openai_api_key = st.secrets["OpenAI_key"] 
# openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    st.stop()

# Ask the user for a question via `st.text_area`.
drug_name = st.text_input(
    "Now input a drug name you want to map to RxNorm",
    value="Hyoscyamine Sulfate"
)
if not drug_name:
    st.info("Please input a drug name to continue.")
    st.stop()


if st.button("Ask AI"):

    llm = ChatOpenAI(model="gpt-3.5-turbo",api_key=openai_api_key,temperature=0)
    # Translation prompt for drug name
    translation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a translation assistant. Translate the following drug name to English."),
        ("human", "{input}")
    ])
    translation_chain = translation_prompt.format_prompt(input=drug_name)
    translated_drug_name = llm(translation_chain.to_messages()).content

    db=Chroma(persist_directory="./vector_db_dir",embedding_function=OpenAIEmbeddings(api_key=openai_api_key),collection_name="local-rag" )
    retriever = db.as_retriever()

    # Set up system prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        
    ])

    # Create the question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)


    answer= rag_chain.invoke({"input": f"what are the possible three concept codes and names and ids for the drug {translated_drug_name}?"})
    # Stream the response to the app using `st.write_stream`.
    st.write(f"The drug name to map: {translated_drug_name} \n {answer['answer']}")
