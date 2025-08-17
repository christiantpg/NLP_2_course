# -----------------------------
# IMPORTS
# -----------------------------
import os

from flask import Flask, render_template, request, jsonify
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone.vectorstores import PineconeVectorStore as LC_Pinecone
from pinecone import Pinecone as PineconeSDK, ServerlessSpec
import dotenv

dotenv.load_dotenv('../.env')
INDEX_NAME = os.getenv("PINECONE_INDEX")

file_loader = PyPDFDirectoryLoader("../docs/")
docs = file_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=int(os.getenv("DOCS_CHUNK_SIZE")),
    chunk_overlap=int(os.getenv("DOCS_CHUNK_OVERLAP"))
)

documents = text_splitter.split_documents(docs)

texts = [doc.page_content for doc in documents]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectors = embeddings.embed_documents(texts)

pinecone = PineconeSDK(api_key=os.getenv("PINECONE_API_KEY"))
spec = ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD"), region=os.getenv("PINECONE_REGION"))

if INDEX_NAME not in pinecone.list_indexes().names():
    pinecone.create_index(
        INDEX_NAME,
        dimension=384,
        metric='cosine',
        spec=spec
    )
    pinecone.create_index(INDEX_NAME, dimension=len(vectors[0]), metric="cosine")

index = pinecone.Index(INDEX_NAME)

for i, (doc, vector) in enumerate(zip(documents, vectors)):
    index.upsert([
        (f"chunk-{i}", vector, {"text": doc.page_content})
    ])

vectorstore = LC_Pinecone(index, text_key="text", embedding=embeddings)

llm = ChatGroq(model=os.getenv("GROQ_MODEL"), temperature=0)

retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
    return_source_documents=True
)

# -----------------------------
# FLASK APP
# -----------------------------
app = Flask(__name__)
conversation_history = []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")

    # Llamada a la chain con historial
    result = retrieval_chain({"question": user_input, "chat_history": conversation_history})

    # Actualizamos el historial
    conversation_history.append((user_input, result["answer"]))

    return jsonify({
        "answer": result["answer"],
        "sources": [doc.page_content for doc in result["source_documents"]]
    })


if __name__ == "__main__":
    app.run(debug=True)
