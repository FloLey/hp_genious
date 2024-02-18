import os
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.embeddings import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseIndexParams, SparseVectorParams
from tqdm import tqdm

# Initialize embedding model
embed_model = HuggingFaceEmbedding(model_name="WhereIsAI/UAE-Large-V1")

# Base directory containing all books
base_directory = "/home/florent/Desktop/AI_Projects/HP_Genious/books"

# Collection name for Qdrant
collection_name = "hp_genious"
vector_name = "hp_genious_vector"


# Initialize Qdrant client
qdrant_client = QdrantClient(url="http://localhost:6333") # Started using docker

# Recreate collection in Qdrant for new data
qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    sparse_vectors_config={
        vector_name: SparseVectorParams(
            index=SparseIndexParams(
                on_disk=True,
            )
        )
    },
)

# Create entry points for Qdrant for a file
def process_file(file_path, chunk_size, chunk_overlap, start_id=0):
    book_name = os.path.basename(os.path.dirname(file_path))
    chapter_name = os.path.splitext(os.path.basename(file_path))[0]

    with open(file_path, 'r', encoding='utf-8') as file:
        documents = file.read()

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(documents)

    # Prepare data for insertion into Qdrant
    data_to_insert = []
    current_id = start_id
    for chunk in chunks:
        # Add the introductory sentence to each chunk
        intro_sentence = f"This is an extract of chapter {chapter_name}: \n"
        modified_chunk = intro_sentence + chunk

        embedding = embed_model.get_text_embedding(modified_chunk)
        point_dict = {
            "id": current_id,
            "vector": embedding,
            "payload": {
                "text": modified_chunk,
                "book": book_name,
                "chapter": chapter_name,
            }
        }
        data_to_insert.append(point_dict)
        current_id += 1

    for point in data_to_insert:
        qdrant_client.upsert(collection_name=collection_name, points=[point])

    return current_id

# Iterate over all .txt files in the directory and its subdirectories with progress bar
last_id = 0
chunks =[(300,50),(600,100)]
file_paths = glob.glob(f"{base_directory}/**/*.txt", recursive=True)
for chunk_size, chunk_overlap in tqdm(chunks, desc="Processing chunks"):
    for file_path in tqdm(file_paths, desc="Processing files"):
        last_id = process_file(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap, start_id=last_id)
