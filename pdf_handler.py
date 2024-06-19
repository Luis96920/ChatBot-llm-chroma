from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from llm_chains import load_vectordb, create_embeddings
from utils import load_config
import pypdfium2
import pytesseract
from PIL import Image
import requests
from urllib.parse import urlparse
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
import io

config = load_config()

def get_pdf_texts(pdfs_bytes_list):
    return [extract_text_from_pdf(pdf_bytes.getvalue()) for pdf_bytes in pdfs_bytes_list]

def extract_text_from_pdf(pdf_bytes):
    pdf_file = pypdfium2.PdfDocument(pdf_bytes)
    return "\n".join(pdf_file.get_page(page_number).get_textpage().get_text_range() for page_number in range(len(pdf_file)))
    

def extract_text_from_image(img):
    extracted_text = pytesseract.image_to_string(img)
    return extracted_text

def extract_text_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    if isinstance(document, tuple):
        text = document[1]  
    elif hasattr(document, 'content'):
        text = document.content  
    else:
        text = str(document)  
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=config["pdf_text_splitter"]["chunk_size"], 
                                              chunk_overlap=config["pdf_text_splitter"]["overlap"],
                                                separators=config["pdf_text_splitter"]["separators"])
    return splitter.split_text(text)

def get_document_chunks(text_list):
    documents = []
    for text in text_list:
        for chunk in get_text_chunks(text):
            documents.append(Document(page_content = chunk))
    return documents

def add_documents_to_db(pdfs_bytes):
    texts = get_pdf_texts(pdfs_bytes)
    documents = get_document_chunks(texts)
    vector_db = load_vectordb(create_embeddings())
    vector_db.add_documents(documents)
    print("Documents added to db.")

def url_documents_to_db(url):
    texts = extract_text_from_url(url)
    documents = get_document_chunks(texts)
    vector_db = load_vectordb(create_embeddings())
    vector_db.add_documents(documents)
    print("Documents added to db.")