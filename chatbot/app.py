import os
import re

import dotenv
from flask import Flask, render_template, request, jsonify
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

dotenv.load_dotenv()

DOCS_CHUNK_SIZE = int(os.getenv("DOCS_CHUNK_SIZE"))
DOCS_CHUNK_OVERLAP = int(os.getenv("DOCS_CHUNK_OVERLAP"))
ALUMNO = os.getenv("ALUMNO")

dotenv.load_dotenv('../.env')
INDEX_NAME = os.getenv("PINECONE_INDEX")
DOCS_PATH = "../docs/"

file_loader = PyPDFDirectoryLoader(DOCS_PATH)
docs = file_loader.load()


class Agent:
    def __init__(self, filename: str, filepath: str, embeddings, llm):
        name = os.path.splitext(filename)[0].lower()
        self.name = name
        self.cv_filepath = filepath
        self.embeddings = embeddings
        self.index = name
        self.llm = llm
        self.default = name == ALUMNO

        self.cv = None
        self.splits = None
        self.vectorstore = None
        self.retriever = None
        self.chain = None

        self._load_cv()
        self._create_vectorstore()
        self._create_chain()

    def _load_cv(self):
        loader = PyPDFLoader(self.cv_filepath)
        self.cv = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=DOCS_CHUNK_SIZE, chunk_overlap=DOCS_CHUNK_OVERLAP)
        self.chunks = splitter.split_documents(self.cv)
        self.vectors = embeddings.embed_documents([doc.page_content for doc in self.chunks])

    def _create_vectorstore(self):
        self.vectorstore = PineconeVectorStore.from_documents(
            self.chunks,
            self.embeddings,
            index_name=self.index
        )
        self.retriever = self.vectorstore.as_retriever()

    def _create_chain(self):
        prompt = ChatPromptTemplate.from_template(
            "Usa el siguiente CV de {name} para responder la pregunta.\n"
            "Contexto:\n{context}\n\n"
            "Pregunta: {input}"
        )

        doc_chain = create_stuff_documents_chain(self.llm, prompt)
        self.chain = create_retrieval_chain(self.retriever, doc_chain)

    def answer(self, question: str, history) -> str:
        result = self.chain.invoke({"input": question, "name": self.name, "chat_history": history})
        return result["answer"]


class MultiAgent:
    def __init__(self, agents, llm, name):
        self.agents = agents
        self.llm = llm
        self.name = name

    def answer(self, mentioned, question: str):
        answers = []
        for agent in mentioned:
            result = agent.answer(question, conversation_history)
            answers.append({
                "agent": agent.name,
                "answer": result,
            })

        context = "\n\n".join(
            [f"[{a['agent'].capitalize()}]: {a['answer']}" for a in answers]
        )

        prompt = f"""La consulta fue: {question}

        Aquí tienes información de varios CVs:
        {context}

        Responde de forma integrada y comparativa, mencionando a cada persona según corresponda.
        """

        return self.llm.invoke(prompt).content


embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDINGS_MODEL"))
groq = ChatGroq(model=os.getenv("GROQ_MODEL"), temperature=0)


def create_agents_from_cvs(directory=DOCS_PATH):
    agents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            agent = Agent(filename, filepath, embeddings=embeddings, llm=groq)
            agents.append(agent)

    return agents


agents = create_agents_from_cvs(DOCS_PATH)
multiagent = MultiAgent(agents, groq, "multiagent")

pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
spec = ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD"), region=os.getenv("PINECONE_REGION"))


def recreate_index(index_name, pinecone, spec):
    if index_name in pinecone.list_indexes().names():
        pinecone.delete_index(index_name)
    print("index {} borrado".format(index_name))

    if index_name not in pinecone.list_indexes().names():
        print("index creado con el nombre: {}".format(index_name))
        pinecone.create_index(
            index_name,
            dimension=384,
            metric='cosine',
            spec=spec
        )
    else:
        print("el index con el nombre {} ya estaba creado".format(index_name))


def upload_to_pinecone(index, documents, vectors):
    for i, (doc, vector) in enumerate(zip(documents, vectors)):
        pinecone_index = pinecone.Index(index)
        pinecone_index.upsert([
            (
                f"chunk-{i}",
                vector,
                {"text": doc.page_content}
            )
        ])

    print(f"vectores cargados en {index}")


for agent in agents:
    recreate_index(agent.index, pinecone, spec)
    upload_to_pinecone(agent.index, agent.chunks, agent.vectors)


def ask(question: str, agents, conversation_history=[]):
    mentioned = [
        agent for agent in agents
        if re.search(agent.name, question, re.IGNORECASE)
    ]

    if len(mentioned) == 0:
        queried_agent = next(agent for agent in agents if agent.default)
        answer = queried_agent.answer(question, conversation_history)

    elif len(mentioned) == 1:
        queried_agent = mentioned[0]
        answer = queried_agent.answer(question, conversation_history)

    else:
        queried_agent = multiagent
        answer = queried_agent.answer(mentioned, question)

    return f"[{queried_agent.name.capitalize()}]: {answer}"


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

    ans = ask(user_input, agents, conversation_history)
    conversation_history.append((user_input, ans))

    return jsonify({
        "answer": ans
    })


if __name__ == "__main__":
    app.run(debug=True)
