import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import CharacterTextSplitter

os.environ["USER_AGENT"] = "Mozilla/5.0"

search = DuckDuckGoSearchRun()


def web_search(query):
    return search.run(query)


def load_chunk_persist_pdf(
    pdf_folder="data/pdfs",
    db_path="./chroma_pdf_db",
    collection_name="pdf_docs",
) -> Chroma:
    documents = []

    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())

    web_loader = WebBaseLoader("https://alfredportfolio.vercel.app")
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


def extract_job_context(profile_docs):
    context = "\n".join([doc.page_content for doc in profile_docs])

    response = model.invoke(f"""
You are an AI job agent.

From this profile, extract:
- best matching job roles
- key skills

Then create an INTERNAL job search query.

Do not explain anything. Just return the query.

Profile:
{context}
""")

    return str(response).strip()


def get_jobs_from_results(results):
    return model.stream(f"""
From the following search results, extract real job opportunities.

For each job provide:
- Job Title
- Company
- Location
- Short description

Only include relevant jobs.

Results:
{results}
""")


def stream_output(generator):
    for chunk in generator:
        print(chunk, end="", flush=True)
    print()


db_path = "./chroma_pdf_db"
collection_name = "pdf_docs"

vector_store = Chroma(
    collection_name=collection_name,
    persist_directory=db_path,
    embedding_function=OllamaEmbeddings(model="mxbai-embed-large"),
)

if not os.path.exists(db_path):
    vector_store = load_chunk_persist_pdf()

retriever = vector_store.as_retriever(search_kwargs={"k": 20})

model = OllamaLLM(model="llama3.2:latest")


def get_profile_docs():
    return retriever.invoke("")


def run_agent():
    print("Scanning your profile...\n")

    docs = get_profile_docs()

    internal_query = extract_job_context(docs)

    if not internal_query or len(internal_query) < 5:
        internal_query = "software developer jobs Kenya OR remote"

    results = web_search(internal_query)

    print("Matching jobs found:\n")

    stream = get_jobs_from_results(results)
    stream_output(stream)


if __name__ == "__main__":
    run_agent()
