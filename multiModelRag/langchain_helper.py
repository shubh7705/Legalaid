import os
import asyncio
from uuid import uuid4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cerebras import ChatCerebras
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from dotenv import load_dotenv
import faiss

load_dotenv()

# -----------------------------
# Initialize models
# -----------------------------
models = {
    "Gemini": ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.8,
        verbose=True,
        api_key=os.getenv("GOOGLE_API_KEY")
    ),
    "Deepseek-R1-distill-llama-70b": ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0.8,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    ),
    "Mistral": ChatMistralAI(
        model_name="open-mistral-nemo",
        temperature=0.8,
        verbose=True
    ),
    # "Llama": ChatCerebras(
    #     model="llama-3.3-70b",
    #     temperature=0.8,
    #     verbose=True,
    #     api_key=os.getenv("CEREBRAS_API_KEY")
    # )
}

# -----------------------------
# Initialize embeddings
# -----------------------------
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
embedding_model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=embedding_model_kwargs,
    encode_kwargs=encode_kwargs
)

# -----------------------------
# Async helper to invoke chains
# -----------------------------
async def async_invoke_chain(chain, input_data):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, chain.invoke, input_data)

# -----------------------------
# Process PDF / Document
# -----------------------------
def process_pdf(file_path):
    """
    Load PDF, split into chunks, and create FAISS vector store
    Returns: doc_retriever
    """
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n"],
        chunk_size=1200,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    # Create FAISS index
    embedding_dim = len(embeddings.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    ids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(chunks, ids=ids)
    for idx, doc_id in enumerate(ids):
        vector_store.index_to_docstore_id[idx] = doc_id

    # Return retriever
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# -----------------------------
# Get context from retriever
# -----------------------------
def get_retrieved_context(retriever, query):
    retrieved_documents = retriever.get_relevant_documents(query)
    return "\n".join(doc.page_content for doc in retrieved_documents)

# -----------------------------
# Generate response
# -----------------------------
def generate_response(model_key, retriever, user_query):
    """
    Generate response from a specific model using the retriever context
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
            You are an expert document analyst with the ability to process large volumes of text efficiently. 
            Your task is to extract key insights and answer questions based on the content of the provided document : {context}
            When asked a question, you should provide a direct, detailed, and concise response, only using the information available from the document. 
            If the answer cannot be found directly, you should clarify this and provide relevant context or related information if applicable.
            Focus on uncovering critical information, whether it's specific facts, summaries, or hidden insights within the document.
        """),
        ("human", "{question}")
    ])

    context = get_retrieved_context(retriever, user_query)
    chain = prompt_template | models[model_key] | StrOutputParser()

    # Run async chain
    return asyncio.run(async_invoke_chain(chain, {"question": user_query, "context": context}))
