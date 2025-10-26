from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

INDEX_DIR = "faiss.index"
DEVICE = "cpu"
MODEL_NAME = "BAAI/bge-base-en-v1.5"

def retrieve(llm, query, retriever, prompt):
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}  # context window
            | prompt
            | llm
            | StrOutputParser()
    )
    result = rag_chain.invoke(query)
    return result

llm = ChatOllama(model="llama3.1", temperature=0.2)

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True}
)
index = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = index.as_retriever()
# Chatbot

instruction_prompt = """Ты помощник, который отвечает на вопросы ТОЛЬКО на основе предоставленного контекста. ВСЕГДА отвечай на русском языке. Если в базе знаний нет ответа на вопрос, то отвечай 'Я не знаю'
Вопрос: {question} 
Контекст: {context} 
Ответ:
"""

while True:
    try:
        input_query = input('Задай мне вопрос: ')
        if input_query.lower() in ("exit", "quit"):
            break
        if input_query.strip():
            answer = retrieve(llm, input_query, retriever, ChatPromptTemplate.from_template(instruction_prompt))
            print("Ответ:")
            print(answer)
            print("—" * 10)
    except KeyboardInterrupt:
        print("До свидания!")
        break