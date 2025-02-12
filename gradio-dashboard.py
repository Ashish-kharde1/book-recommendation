#%%
import os
import numpy as np
import pandas as pd
from  dotenv import  load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

import gradio as gr

load_dotenv()
os.environ["GOOGLE_API_KEY"]

books=pd.read_csv("books_with_emotion.csv")
books["large_thumbnail"]=books["thumbnail"] + "&fife=w800"
books["large_thumbnail"]=np.where(
    books["large_thumbnail"].isna(),
    "cover_not_found.jpg",
    books["large_thumbnail"]
)

loader = TextLoader("tagged_description.txt",encoding="utf-8")
doc=loader.load()
text_splitter=CharacterTextSplitter(chunk_size=0,chunk_overlap=0,separator="\n")
documents=text_splitter.split_documents(doc)
embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db_books=Chroma.from_documents(documents,embeddings)
