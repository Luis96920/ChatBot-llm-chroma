from prompt_templates import memory_prompt_template, pdf_chat_prompt
from langchain.chains import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms import HuggingFaceHub
from langchain.llms import Replicate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from operator import itemgetter
from utils import load_config
import chromadb
from dotenv import load_dotenv 
import os
config = load_config()
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

def replicate_Llama_model():
    llm = Replicate(
        streaming = True,
        model = "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781", 
        callbacks=[StreamingStdOutCallbackHandler()],
        input = {"temperature": 0.5, "max_length" :1024,"top_p":1})
    return llm

def load_google_model():
    return ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3,api_key = os.getenv("GOOGLE_API_KEY"))

def create_llm(model_path = config["ctransformers"]["model_path"]["large"], model_type = config["ctransformers"]["model_type"], model_config = config["ctransformers"]["model_config"]):
    llm = CTransformers(model=model_path, model_type=model_type, config=model_config)
    return llm

def load_llm():
    return  HuggingFaceHub(repo_id = "microsoft/Phi-3-vision-128k-instruct", model_kwargs={"temperature":0.5, "max_length":512})

def create_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

def create_chat_memory(chat_history):
    return ConversationBufferWindowMemory(memory_key="history", chat_memory=chat_history, k=3)

def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)

def create_llm_chain(llm, chat_prompt):
    return LLMChain(llm=llm, prompt=chat_prompt)
    
def load_normal_chain():
    return chatChain()

def load_vectordb(embeddings):
    persistent_client = chromadb.PersistentClient(config["chromadb"]["chromadb_path"])

    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=config["chromadb"]["collection_name"],
        embedding_function=embeddings,
    )

    return langchain_chroma

def load_pdf_chat_chain():
    return pdfChatChain()

def load_retrieval_chain(llm, vector_db):
    return RetrievalQA.from_llm(llm=llm, retriever=vector_db.as_retriever(search_kwargs={"k": config["chat_config"]["number_of_retrieved_documents"]}), verbose=True)

def create_pdf_chat_runnable(llm, vector_db, prompt):
    runnable = (
        {
        "context": itemgetter("human_input") | vector_db.as_retriever(search_kwargs={"k": config["chat_config"]["number_of_retrieved_documents"]}),
        "human_input": itemgetter("human_input"),
        "history" : itemgetter("history"),
        }
    | prompt | llm.bind(stop=["Human:"]) 
    )
    return runnable

class pdfChatChain:

    def __init__(self):
        vector_db = load_vectordb(create_embeddings())
        #llm = load_llm()
        llm = replicate_Llama_model()
        #llm = create_llm()
        prompt = create_prompt_from_template(pdf_chat_prompt)
        self.llm_chain = create_pdf_chat_runnable(llm, vector_db, prompt)

    def run(self, user_input, chat_history):
        print("File Chat chain is running...")
        return self.llm_chain.invoke(input={"human_input" : user_input, "history" : chat_history})

class chatChain:

    def __init__(self):
        #llm = create_llm()
        llm = load_google_model()
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = create_llm_chain(llm, chat_prompt)

    def run(self, user_input, chat_history):
        print("Chat chain is running...")
        return self.llm_chain.invoke(input={"human_input" : user_input, "history" : chat_history} ,stop=["Human:"])["text"]