from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_DIR = "faiss.index"

DEVICE = "cpu"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
#MODEL_NAME = "cointegrated/rubert-tiny2"
#MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
#MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
#MODEL_NAME = "BAAI/bge-m3"
#MODEL_NAME = "ai-forever/sbert_large_mt_nlu_ru"
#MODEL_NAME = "DeepPavlov/rubert-base-cased-sentence"

#QUERY_TEXT = "Саша Скил"
QUERY_TEXT = "кризис подливиона"
#QUERY_TEXT = "древний манускрипт"

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True}
)
index = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

print(f"Поиск фразы '{QUERY_TEXT}'")

# ищем 5 документов
docs = index.similarity_search(QUERY_TEXT, k=5)

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