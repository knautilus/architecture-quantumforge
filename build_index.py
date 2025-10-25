from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
import os
import glob

def create_chunks(file_path):
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=[
            "\n\n",            # логические разделы
            "\n",              # абзацы
            ". ", "! ", "? ",  # предложения
            " ",               # слова
            ""
        ]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Создано {len(chunks)} чанков")
    return chunks

def create_vectorstore(chunks):
    device = "cpu"
    print("Добавление метаданных...")
    for i, doc in enumerate(chunks):
        source = doc.metadata['source']
        title = os.path.splitext(os.path.basename(source))[0].title()
        doc.metadata['title'] = title
        doc.metadata['chunk_id'] = f"chunk_{i:04d}"
    print("Создание векторного хранилища...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

print("Загрузка документов...")
chunks = []
for file in glob.glob("./knowledge_base/processed/*.txt"):
    chunks.extend(create_chunks(file))
print(f"Общее количество чанков: {len(chunks)}")
index = create_vectorstore(chunks)
print("Сохранение faiss.index...")
index.save_local("faiss.index")