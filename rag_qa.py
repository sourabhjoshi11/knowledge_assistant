from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from model import llm
from ingestion.vecdb import vectorstore
from guardrails import pii
from langsmith import traceable
system_prompt = (
    "You are a helpful assistant. Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. \n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])


prompt_search_query = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("human", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])        

retriever = vectorstore.as_retriever()


retriever_chain = create_history_aware_retriever(llm, retriever, prompt_search_query)

document_chain = create_stuff_documents_chain(llm, prompt)


@traceable
def rag():
    return create_retrieval_chain(retriever_chain, document_chain)













