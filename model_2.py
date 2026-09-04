import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 1. CSV file load karein
df = pd.read_csv('/content/IPC.csv')

# 2. Dataset ko LangChain Documents mein convert karein
legal_docs = []
for index, row in df.iterrows():
    content = f"Section: {row['Section']} | Offense: {row['Offense']} | Description: {row['Description']} | Punishment: {row['Punishment']}"

    doc = Document(
        page_content=content,
        metadata={"section": str(row['Section'])}
    )
    legal_docs.append(doc)

# 3. Embeddings model load karein aur FAISS database banayein
print("Building FAISS Vector Database... Please wait.")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(legal_docs, embeddings)
print("Vector Database Successfully Created!")

# 4. Test Query karke check karein
query = "Online shopping mein fraud hua, cheating ki"
docs = db.similarity_search(query, k=1)

print("\n--- Test Result ---")
print(f"User Query: {query}")
print(f"Matched Legal Section: {docs[0].metadata['section']}")
print(f"Details: {docs[0].page_content[:300]}...")