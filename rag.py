
import os
from typing import Optional

# document loader
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# text splitter - chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter

# embedding
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

_vectorstore = None
_init_error = None

try:
    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
    )
    documents = loader.load()

    if not documents:
        raise ValueError(f"No .txt documents found in {DATA_DIR}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    _vectorstore = FAISS.from_documents(chunks, embeddings)

except Exception as e: 
    _init_error = str(e)



_SIMULATED_CONFLICT_FACTS = {
    "apple": {
        "ceo": "Steve Jobs is still the current CEO of Apple.",
        "headquarters": "Apple is headquartered in Redmond, Washington.",
        "founder": "Apple was founded by Bill Gates and Paul Allen.",
    },
    "google": {
        "ceo": "Larry Page is the current CEO of Google.",
        "headquarters": "Google is headquartered in Redmond, Washington.",
        "founder": "Google was founded by Bill Gates.",
        "parent": "Google's parent company is called Meta.",
    },
    "microsoft": {
        "ceo": "Bill Gates is the current CEO of Microsoft.",
        "headquarters": "Microsoft is headquartered in Cupertino, California.",
        "founder": "Microsoft was founded by Steve Jobs.",
    },
}
_SIMULATED_CONFLICT_FACTS["alphabet"] = _SIMULATED_CONFLICT_FACTS["google"]

_ATTRIBUTE_KEYWORDS = [
    ("headquarters", ("headquarter", "located", "based in", "location")),
    ("founder", ("found", "started", "created")),
    ("parent", ("parent company", "parent")),
    ("ceo", ("ceo", "chief executive", "president", "who runs", "who leads")),
]


def _detect_company(question: str) -> Optional[str]:
    q = question.lower()
    for company in _SIMULATED_CONFLICT_FACTS:
        if company in q:
            return company
    return None  


def _detect_attribute(question: str) -> str:
    q = question.lower()
    for attribute, keywords in _ATTRIBUTE_KEYWORDS:
        if any(keyword in q for keyword in keywords):
            return attribute
    return "ceo"  


def _simulated_conflict_content(company: str, question: str) -> str:
    attribute = _detect_attribute(question)

    facts_for_company = _SIMULATED_CONFLICT_FACTS[company]
    fake_fact = facts_for_company.get(attribute, facts_for_company["ceo"])

    return (
        "According to this (deliberately incorrect, demo-only) record, "
        f"{fake_fact}"
    )


def retrieve_documents(question: str, simulate_conflict: bool = False):
    """
    Query the local knowledge base.

    Returns a dict with a `status` field so callers can distinguish
    success / no_results / error without inspecting exceptions.
    """

    if simulate_conflict:
        company = _detect_company(question)

        if company is None:
       
            return {
                "source": "local_rag",
                "status": "no_results",
                "content": None,
                "note": (
                    "simulate_conflict=True, but this question isn't about "
                    "a company in the local knowledge base (Apple, Google, "
                    "Microsoft), so no simulated conflict was injected."
                ),
            }

        return {
            "source": "local_rag",
            "status": "success",
            "content": _simulated_conflict_content(company, question),
            "sources": ["data/_simulated_conflict.txt"],
            "note": "This result was artificially injected via simulate_conflict=True for demo purposes.",
        }

    if _vectorstore is None:
        return {
            "source": "local_rag",
            "status": "error",
            "content": None,
            "error": f"Local index failed to initialize: {_init_error}",
        }

    try:

        SCORE_THRESHOLD = 0.5

        retriever = _vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": SCORE_THRESHOLD},
        )
        results = retriever.invoke(question)
    except Exception as e: 
        return {
            "source": "local_rag",
            "status": "error",
            "content": None,
            "error": str(e),
        }

    if not results:
        return {
            "source": "local_rag",
            "status": "no_results",
            "content": None,
        }

    context = "\n\n".join(result.page_content for result in results)
    sources = [result.metadata.get("source") for result in results]

    return {
        "source": "local_rag",
        "status": "success",
        "content": context,
        "sources": sources,
    }


