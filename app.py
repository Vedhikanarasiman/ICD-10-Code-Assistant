import streamlit as st
from PyPDF2 import PdfReader #library to read pdf files
from langchain.text_splitter import RecursiveCharacterTextSplitter#library to split pdf files
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings #to embed the text
import google.generativeai as genai

from langchain.vectorstores import FAISS #for vector embeddings
from langchain_google_genai import ChatGoogleGenerativeAI #
from langchain.chains.question_answering import load_qa_chain #to chain the prompts
from langchain.prompts import PromptTemplate #to create prompt templates
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    # iterate over all pdf files uploaded
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        # iterate over all pages in a pdf
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    # create an object of RecursiveCharacterTextSplitter with specific chunk size and overlap size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
    # now split the text we have using object created
    chunks = text_splitter.split_text(text)

    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001") # google embeddings
    vector_store = FAISS.from_texts(text_chunks,embeddings) # use the embedding object on the splitted text of pdf docs
    vector_store.save_local("faiss_index") # save the embeddings in local

def get_conversation_chain():

    # define the prompt
    prompt_template = """
    You are provided with the context from the ICD-10 Volume 3 Alphabetical Index. The context includes terms and their corresponding 
    ICD-10 codes. Your task is to find the most relevant ICD-10 code(s) for the given diagnosis from this context.\n\n
    Context:\n {context}?\n
    Given the diagnosis or condition: \n{question}\n

    Please find and provide the most accurate ICD-10 code(s) that match this diagnosis or condition from the context. If the exact diagnosis is 
    not available, provide the closest possible code(s) related to the diagnosis.

    Answer:
    """

    model = ChatGoogleGenerativeAI(model = "gemini-1.0-pro", temperatue = 0.2) # create object of gemini-pro

    prompt = PromptTemplate(template = prompt_template, input_variables= ["context","question"])

    chain = load_qa_chain(model,chain_type="stuff",prompt = prompt)

    return chain

def user_input(user_question):
    # user_question is the input question
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # load the local faiss db
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # using similarity search, get the answer based on the input
    docs = new_db.similarity_search(user_question)

    chain = get_conversation_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Answer: ", response["output_text"])

def main():
    st.set_page_config(
    page_title="ICD-10 Code Chat Assistant",  
    page_icon="🔍")
    st.header("ICD-10 Code Assistant")

    user_question = st.text_input("Ask a Question:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()