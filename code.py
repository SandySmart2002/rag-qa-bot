# prompt: Problem Statement:
# Develop a Retrieval-Augmented Generation (RAG) model for a Question Answering (QA)
# bot for a business. Use a vector database like Pinecone DB and a generative model like
# Cohere API (or any other available alternative). The QA bot should be able to retrieve
# relevant information from a dataset and generate coherent answers.

import cohere
import pinecone
import os
from sentence_transformers import SentenceTransformer
import json
from pinecone import Pinecone
# Replace with your API keys
PINECONE_API_KEY = "07bda632-c51f-4966-8a32-d5e0d3b354d9"
COHERE_API_KEY = "6wvwdutBMrbEeNvVNFQyLHAWWKxi9smp5dS14TTo"


# Initialize Pinecone
pinecone = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1")

# Create or connect to an index (adjust dimension based on your embedding model)
index_name = "business-qa"
if index_name not in pinecone.list_indexes():
  from pinecone import ServerlessSpec
  spec = ServerlessSpec(
        cloud="aws",  # Choose your cloud provider
        region="us-east-1"  # Choose the region
    )
  pinecone.create_index(index_name, dimension=384, spec=spec)  # Example dimension size
index = pinecone.Index(index_name)

# Load the embedding model (Sentence-BERT is used here for embeddings)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Assume we have a large document split into chunks
document_chunks = [
    "Section 1: Details about resetting passwords...",
    "Section 2: Information on account recovery...",
    "Section 3: Create a new account...",
    # Add more chunks from your dataset
]

# Generate embeddings for the document chunks
chunk_embeddings = embedding_model.encode(document_chunks)

# Store each chunk's embedding in Pinecone along with its metadata
for i, embedding in enumerate(chunk_embeddings):
    index.upsert([(f"doc_chunk_{i}", embedding.tolist(), {"text": document_chunks[i]})])

# Function to retrieve relevant documents
def retrieve_relevant_documents(query, top_k=3):
    # Convert the query to an embedding
    query_embedding = embedding_model.encode([query])[0]
    
    # Query Pinecone for the most relevant documents
    index.query(vector=[0.8,0.05,0.59,0.42,0.13,0.27,0.94,0.15,0.38,0.49,0.29,0.56,0.01,0.33,0.46,0.65,0.91,0.04,0.69,0.26,0.43,0.56,1,0.68,0.15,0.57,0.51,0.66,0.25,0.81,0.07,0.63,0.03,0.7,0.1,0.7,0.2,0.2,0.14,0.38,0.59,0.01,0.7,0.06,0.23,0.3,0.52,0.68,0.4,0.4,0.44,0.89,0.78,0.79,0.6,0.73,0.98,0.31,0.9,0.42,0.75,0,0.53,0.77,0.1,0.97,0.08,0.77,0.27,0.79,0.89,0.2,0.51,0.76,0.57,0.55,0.97,0.73,0.92,0.88,0.07,0.33,0.94,0.62,0.69,0.24,0.12,0.76,0.04,0.77,0.84,0.43,0.45,0.92,0.43,0.53,0.47,0.39,1,0.59,0.73,0.76,0.81,0.27,0.36,0.18,0.6,0.24,0.1,0.72,0.03,0.42,0.58,0.98,0.53,0.74,0.43,0.01,0.27,0.07,0.44,0.88,0.21,0.76,0.72,0.11,0.6,0.37,0.03,0.42,0.77,0.43,0,0.68,0.9,0.99,0.6,0.18,0.11,0.29,0.33,0.78,0.71,0.24,0.63,0.21,0.88,0.39,0.98,0.68,0.24,0.89,0.83,0.19,0.7,0.74,0.76,0.33,0.55,0.97,0.86,0.7,0.61,0.79,0.19,0.47,0.83,0.43,0.89,0.71,0.12,0.55,0.07,0.3,0.51,0.97,0.85,0.19,0.12,0.6,0.58,0.39,0.25,0.51,0.52,0.09,0.89,0.1,0.59,0.27,0.55,0.95,0.09,0.06,0.2,0.18,0.13,0.26,0.01,0.18,0.33,0.61,0.67,0.44,0.9,0.03,0.49,0.22,0.48,0.77,0.13,0.47,0.11,0.22,0.48,0.46,0.89,0.66,0.44,0.47,0.59,0.68,0.42,0.33,0.65,0.41,0.75,0.78,0.26,0.12,0.47,0.75,0.36,0.92,0.1,0.86,0.37,0.14,0.82,0.27,0.24,0.77,0.81,0.47,0.05,0.87,0.82,0.44,0.87,0.59,0.17,0.1,0.06,0.93,0.07,0.66,0.98,0.78,0.04,0.35,0.03,0.2,0.52,0.12,0.75,0.46,0.14,0.03,0.28,0.33,0.84,0.54,0.71,0.13,0.71,0.96,0.01,0.02,0.08,0.79,0.63,0.75,0.42,0.03,0.3,0.43,0.62,0.73,0.13,0.49,0.78,0.61,0.55,0.47,0.23,0.6,0.68,0.05,0.7,0.34,0.91,0.76,0.32,0.37,0.26,0.46,0.68,0.19,0.45,0.86,0.91,0.13,0.57,0.37,0.18,0.09,0.05,0.78,0.76,0.84,0.34,0.95,0.82,0.84,0.99,0.8,0.06,0.95,0.79,0.38,0.5,0.01,0.91,0.05,0.15,0.31,0.34,0.4,0.75,0.63,0.22,0.9,0.23,0.19,0.89,0.74,0.79,0.49,0.79,0.81,0.13,0.3,0.31,0.69,0.74,0.24,0.95,0.2,0.16,0.62,0.94,0.19,0.54,0.68,0.39,0.02,0.71,0.81,0.35,0.58,0.61,0.09,0.38,0.34,0.22,0.86,0.62,0.23,0.08,0.44,0.52,0.01,0.32,0.93], top_k=10, namespace='my_namespace')
    # Function to retrieve relevant documents
def retrieve_relevant_documents(query, top_k=3):
    # Convert the query to an embedding
    query_embedding = embedding_model.encode([query])[0]
    
    # Query Pinecone for the most relevant documents
    response = index.query(vector=query_embedding.tolist(), top_k=top_k, namespace='my_namespace') # assign result of index.query to response
    # Extract the text from the retrieved documents
    retrieved_docs = [match['metadata']['text'] for match in response['matches']]
    
    return retrieved_docs
# Initialize the Cohere client
co = cohere.Client(COHERE_API_KEY)

def generate_answer_with_context(query, retrieved_docs):
    # Combine the retrieved docs to form the context for the answer
    context = " ".join(retrieved_docs)
    
    # Prompt the generative model with the query and context
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    
    response = co.generate(
        model="command-xlarge-nightly",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    
    return response.generations[0].text.strip()

# Main function to answer query
def answer_query(query):
    # Step 1: Retrieve relevant documents using Pinecone
    relevant_docs = retrieve_relevant_documents(query)
    
    # Step 2: Generate a coherent answer using a generative model (Cohere)
    final_answer = generate_answer_with_context(query, relevant_docs)
    
    return final_answer

# Example usage
query = "How to create a new account?"
answer = answer_query(query)
print("Answer:", answer)

#