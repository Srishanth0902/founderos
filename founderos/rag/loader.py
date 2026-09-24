import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader

def load_documents(directory_path: str):
    """
    Loads all markdown and json files from the knowledge directory.
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory {directory_path} not found.")

    # Load Markdown files
    md_loader = DirectoryLoader(directory_path, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    md_docs = md_loader.load()

    # Load JSON files
    json_loader = DirectoryLoader(directory_path, glob="**/*.json", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    json_docs = json_loader.load()

    all_docs = md_docs + json_docs
    print(f"Loaded {len(all_docs)} documents from {directory_path}")
    return all_docs
