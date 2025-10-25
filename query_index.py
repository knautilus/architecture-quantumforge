from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_DIR = "faiss.index"

device = "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)
index = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

query_text = "Саша Скил"
print(f"Поиск фразы '{query_text}'")

# ищем 5 документов
docs = index.similarity_search(query_text, k=5)

for number, doc in enumerate(docs, 1):
    print(f"Результат {number}:")
    print(
        f"Источник: {doc.metadata['source']}, "
        f"чанк: {doc.metadata.get('chunk_id', 'N/A')}."
    )
    print("Текст результата:")
    print(doc.page_content)
    print("-" * 50)
print(f"Общее количество результатов: {len(docs)}")