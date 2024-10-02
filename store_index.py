from src.helper import load_pdf,text_split,download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')



extracted_data=load_pdf("data/")
text_chunks=text_split(extracted_data)
embedding_model = download_hugging_face_embeddings()


# 
pc = Pinecone(api_key=PINECONE_API_KEY)  
index_name="medical-chatbot"                          ##data is already stored in my API key so you can comment on this code between # to #
index = pc.Index(index_name)  
for i, t in zip(range(len(text_chunks)), text_chunks):
   query_result = embedding_model.embed_query(t.page_content)
   index.upsert(
   vectors=[
        {
            "id": str(i),  # Convert i to a string
            "values": query_result, 
            "metadata": {"text":str(text_chunks[i].page_content)} # meta data as dic
        }
    ],
    namespace="real" 
)
#
