from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Splits documents into smaller chunks for vectorization.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    return chunks
