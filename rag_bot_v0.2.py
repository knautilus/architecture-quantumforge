from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from sentence_transformers import SentenceTransformer

INDEX_DIR = "faiss.index"
DEVICE = "cpu"
MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Поиск релевантного контекста
def search_context(query, index, embedder, top_k=5):
    docs = index.similarity_search(query, k=top_k)
    context_chunks = []
    for doc in docs:
        context_chunks.append(doc.page_content)
    # Объединяем документы в один текст
    if context_chunks:
        return "\n\n".join(context_chunks)
    else:
        return None

def retrieve(llm, query, context, prompt):
    rag_chain = prompt | llm | StrOutputParser()
    result = rag_chain.invoke({"question": query, "context": context})
    return result

llm = ChatOllama(model="llama3.1", temperature=0.2)

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True}
)
embedder = SentenceTransformer(MODEL_NAME)
index = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
# Chatbot

instruction_prompt = PromptTemplate(
    template="""Ты помощник, который отвечает на вопросы ТОЛЬКО на основе предоставленного контекста. ВСЕГДА отвечай на русском языке. Если в базе знаний нет ответа на вопрос, то отвечай 'Я не знаю'
Вопрос: {question} 
Контекст: {context} 
Ответ:
""",
    input_variables=["question", "context"],
)

while True:
    try:
        input_query = input('Задай мне вопрос: ')
        if input_query.lower() in ("exit", "quit"):
            break
        if input_query.strip():
            context = search_context(input_query, index, embedder)
            answer = retrieve(llm, input_query, context, instruction_prompt)
            print("Ответ:")
            print(answer)
            print("—" * 10)
    except KeyboardInterrupt:
        print("До свидания!")
        break
