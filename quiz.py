import os

import requests
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import CharacterTextSplitter

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def web_search(query):
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    headers = {"User-Agent": os.environ.get("USER_AGENT", "Mozilla/5.0")}
    r = requests.post(url, data=params, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    for a in soup.find_all("a", class_="result__a"):
        title = a.get_text()
        href = a.get("href")
        results.append(f"{title} - {href}")
    return "\n".join(results)


def scholar_search(query):
    return web_search(f"{query} research paper filetype:pdf OR journal")


web_tool = Tool(
    name="Web Search",
    func=web_search,
    description="Search the web for general information",
)

scholar_tool = Tool(
    name="Scholar Search",
    func=scholar_search,
    description="Search for academic papers and research",
)


def load_chunk_persist_pdf(
    pdf_folder="data/pdfs", db_path="./chroma_pdf_db", collection_name="pdf_docs"
) -> Chroma:
    documents = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())
    web_loader = WebBaseLoader(
        "https://alfredportfolio.vercel.app",
        headers={"User-Agent": os.environ.get("USER_AGENT", "Mozilla/5.0")},
    )
    web_docs = web_loader.load()
    for doc in web_docs:
        doc.metadata["source"] = "portfolio_site"
        documents.append(doc)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=db_path,
    )


db_path = "./chroma_pdf_db"
collection_name = "pdf_docs"

vector_store = Chroma(
    collection_name=collection_name,
    persist_directory=db_path,
    embedding_function=OllamaEmbeddings(model="mxbai-embed-large"),
)

if not os.path.exists(db_path):
    vector_store = load_chunk_persist_pdf()

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

model = OllamaLLM(model="llama3.2:latest")

template = """
Context:
{context}

Question:
{question}

If the context is not enough, say INSUFFICIENT_CONTEXT.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


def stream_output(generator):
    for chunk in generator:
        print(chunk, end="", flush=True)
    print()


def query_pdf(question):
    docs = retriever.invoke(question)
    stream = chain.stream({"context": docs, "question": question})
    response = ""
    for chunk in stream:
        print(chunk, end="", flush=True)
        response += str(chunk)
    print()
    return response, docs


def query_web(question):
    results = web_tool.run(question)
    stream = model.stream(f"Answer based on these results:\n{results}")
    stream_output(stream)
    return results


def query_scholar(question):
    results = scholar_tool.run(question)
    stream = model.stream(f"Answer based on these academic results:\n{results}")
    stream_output(stream)
    return results


def query_all(question):
    pdf_response, pdf_docs = query_pdf(question)
    web_results = web_tool.run(question)
    scholar_results = scholar_tool.run(question)
    combined_context = ""
    for doc in pdf_docs:
        combined_context += doc.page_content + "\n"
    combined_context += "\nWeb Results:\n" + web_results
    combined_context += "\nScholar Results:\n" + scholar_results
    stream = chain.stream({"context": combined_context, "question": question})
    stream_output(stream)


while True:
    question = input("Ask your question (q to quit): ")
    if question.lower() == "q":
        break
    mode = input("Choose mode (pdf / web / scholar / all): ").lower()
    if mode == "pdf":
        response = query_pdf(question)
        if "INSUFFICIENT_CONTEXT" in response[0]:
            fallback = input(
                "No sufficient context found. Search online? (web / scholar / no): "
            ).lower()
            if fallback == "web":
                query_web(question)
            elif fallback == "scholar":
                query_scholar(question)
    elif mode == "web":
        query_web(question)
    elif mode == "scholar":
        query_scholar(question)
    elif mode == "all":
        query_all(question)
    else:
        print("Invalid mode")
