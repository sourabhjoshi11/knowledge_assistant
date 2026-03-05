from langchain_community.document_loaders import PyPDFLoader


def load_documents():
    loader=PyPDFLoader("ingestion/policy.pdf")
    return loader.load()

docs=load_documents()





