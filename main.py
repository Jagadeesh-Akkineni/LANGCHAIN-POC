# from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# from langchain.chains import SequentialChain
from langchain.agents import AgentType, initialize_agent, load_tools
import os
#----------------------------------------------------------------------
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI
import streamlit as st


def get_serp_response(question):
    tools = load_tools(["wikipedia", "llm-math"], llm = llm)
    agent = initialize_agent(tools, llm, agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    response = agent.run(question)
    
    return response

def get_bs4_response(question):
    embeddings = OpenAIEmbeddings(openai_api_key=globe_key)
    loader = WebBaseLoader("https://www.jagranjosh.com/general-knowledge/list-of-double-centuries-in-odis-1576060960-1")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)

    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
    ])

    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
 
    <context>
    {context}
    </context>
 
    Question: {input}""")
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": question})

    return response['answer']

#-----------------------------------------------------------------------------

# Configuring basic site settings
st.set_page_config(page_title="LangChain App", page_icon=":fork_and_knife:")
 
st.title("LangChain App")
 
# Tabs for navigation
tabs = ["API INPUT", "WIKI MATH", "RAG USING BEAUTIFUL SOUP"]
page = st.sidebar.selectbox("Navigation", tabs)

# Define a global variable for the API key
globe_key = None

# Check if the page is "API INPUT"
if page == "API INPUT":
    # Select box to enter the API key
    with st.form(key="api_key_form"):
        st.write("Enter API Key:")
        api_key = st.text_input(label="", key="api_key_input")
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        # Assign the API key to the global variable
        globe_key = api_key

        print(f"Type of API key: {type(globe_key)}")
        print(f"API key: {globe_key}")

if globe_key != None:
    # Set the OpenAI API key in the environment variable
    os.environ['OPENAI_API_KEY'] = globe_key
    # os.environ["SERPAPI_API_KEY"] = k.secret_key_serp


llm = OpenAI(temperature=0.7)

if page == "WIKI MATH":
    st.title("WIKI MATH")
 
    input = st.text_input("Input : ", key="input")
 
    response = get_serp_response(input)
 
    submit = st.button("ASK")
 
    if submit:
        st.subheader("The response is: ")
        st.write(response)
 
if page == "RAG USING BEAUTIFUL SOUP":
    st.title("BS4")
 
    input = st.text_input("Input : ", key="input")
 
    response = get_bs4_response(input)
 
    submit = st.button("ASK")
 
    if submit:
        st.subheader("The response is: ")
        st.write(response)

