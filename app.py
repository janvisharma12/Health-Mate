from flask import Flask,render_template,jsonify,request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone as PineconeV
import pinecone
from pinecone import Pinecone
from langchain.schema import Document
from typing import List, Dict
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')


# Define the PromptTemplate class
class PromptTemplate:
    def __init__(self, template: str, input_variables: List[str]):
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs):
        return self.template.format(**kwargs)

# Define the RetrievalQAChain class
class RetrievalQAChain:
    def __init__(self, pinecone_api_key: str, index_name: str, model_path: str, model_type: str, prompt_template: PromptTemplate):
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
        
        # Initialize language model
        self.llm = CTransformers(
            model=model_path,
            model_type=model_type,
            config={'max_new_tokens': 512, 'temperature': 0.8}
        )
        
        # Initialize prompt template
        self.prompt_template = prompt_template
        
        # Define your embedding model (you should define or load your embedding model)
        self.embedding_model = download_hugging_face_embeddings()

    def perform_semantic_search(self, query: str, top_k: int = 2) -> List[Dict]:
        # Embed the query
        query_embedding = self.embedding_model.embed_query(query)
        
        # Perform semantic search
        result = self.index.query(
            vector=query_embedding,
            namespace="real",
            top_k=top_k,
            include_values=True,
            include_metadata=True
        )
        
        return result['matches']

    def generate_prompt(self, context: str, question: str) -> str:
        return self.prompt_template.format(context=context, question=question)
    
    def retrieve_answers(self, query: str, top_k: int = 2) -> Dict:
        # Perform semantic search to get relevant documents
        search_results = self.perform_semantic_search(query, top_k)
        
        answers = []
        source_documents = []
        
        for match in search_results:
            context = match['metadata'].get('text', '')
            prompt = self.generate_prompt(context, query)
            
            # Get answer from language model
            answer = self.llm(prompt)
            
            answers.append(answer)
            source_documents.append({
                'id': match['id'],
                'score': match['score'],
                'context': context
            })
        
        # Return the first answer and source documents
        return {
            'result': answers[0] if answers else "I don't know.",
            'source_documents': source_documents
        }

# Define your embedding model (replace this with your actual embedding model)
class YourEmbeddingModel:
    def embed_query(self, query):
        # This should return the embedding of the query
        # Replace with your actual embedding implementation
        pass

# Define the prompt template

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Initialize the retrieval QA chain
qa_chain = RetrievalQAChain(
    pinecone_api_key="867f0699-efba-4d89-8733-bde228aca31b",
    index_name="medical-chatbot",
    model_path="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    prompt_template=PROMPT
)

# Interactive loop
# while True:
#     user_input = input("Input Prompt: ")
#     result = qa_chain.retrieve_answers(user_input)
#     print("Response:", result["result"])
#     for doc in result["source_documents"]:
#         print(f"Source Document ID: {doc['id']}, Score: {doc['score']}")
#         print(f"Context: {doc['context']}\n")






@app.route("/")
def index():
    return render_template('chat.html')




@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa_chain.retrieve_answers(input)
    print("Response : ", result["result"])
    return str(result["result"])




if __name__ == '__main__':
    app.run(debug=True)