import streamlit as st
from rag_qa import rag
from langchain_core.messages import HumanMessage, AIMessage

from guardrails import validate_input,validate_output,pii


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

rag=rag()

question = st.text_input("Ask a question from your documents")

if question:
    if validate_input(question):
        response = ""
        
        with st.chat_message("AI"):
            answer = st.empty() 
            
            for chunk in rag.stream({
                    "input": question, 
                    "chat_history": st.session_state.chat_history
                }):
                if "answer" in chunk:
                    response += chunk["answer"]
                    answer.write(response) 

        if validate_output(response):
               st.session_state.chat_history.append(HumanMessage(content=question))
               st.session_state.chat_history.append(AIMessage(content=response))
              
        else:
            st.error("output validation error")

