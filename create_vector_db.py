"""
Create ChromaDB vector database from hierarchical crawl JSON
"""
import json
import chromadb
from chromadb.config import Settings
from datetime import datetime

def create_chroma_db(json_file_path: str, collection_name: str = "pytorch_docs"):
    """
    Create a ChromaDB vector database from the filtered JSON file

    Args:
        json_file_path: Path to the filtered JSON file
        collection_name: Name for the ChromaDB collection
    """
    print(f"🚀 Starting ChromaDB creation from: {json_file_path}")

    # Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize ChromaDB client (persistent storage)
    client = chromadb.PersistentClient(path="./chroma_db")

    # Create or get collection
    try:
        # Delete existing collection if it exists
        client.delete_collection(name=collection_name)
        print(f"🗑️  Deleted existing collection: {collection_name}")
    except:
        pass

    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={
            "description": "PyTorch documentation semantic paths",
            "domain": data["crawl_metadata"]["domain"],
            "created_at": datetime.now().isoformat()
        }
    )

    print(f"✅ Created collection: {collection_name}")

    # Prepare data for insertion
    documents = []
    metadatas = []
    ids = []

    for idx, (key, element) in enumerate(data["semantic_elements"].items()):
        # Create a rich document text from all fields
        document_text = f"""
Semantic Path: {element['semantic_path']}
Original URL: {element['original_url']}
Source Type: {element['source_type']}
Element Type: {element['element_type']}
"""

        documents.append(document_text.strip())

        # Store metadata
        metadatas.append({
            "semantic_path": element['semantic_path'],
            "original_url": element['original_url'],
            "source_type": element['source_type'],
            "element_type": element['element_type']
        })

        # Create unique ID
        ids.append(f"element_{idx}")

    # Add documents to collection in batches
    batch_size = 100
    total_docs = len(documents)

    print(f"📊 Adding {total_docs} documents to ChromaDB...")

    for i in range(0, total_docs, batch_size):
        batch_end = min(i + batch_size, total_docs)
        collection.add(
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end],
            ids=ids[i:batch_end]
        )
        print(f"   ✓ Added batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} ({batch_end}/{total_docs} documents)")

    print(f"\n✅ Successfully created ChromaDB with {total_docs} documents!")
    print(f"📁 Database location: ./chroma_db")
    print(f"📦 Collection name: {collection_name}")

    # Test query
    print(f"\n🔍 Testing query: 'PyTorch installation'")
    results = collection.query(
        query_texts=["PyTorch installation"],
        n_results=3
    )

    print("\n📋 Top 3 results:")
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        print(f"\n{i}. Semantic Path: {metadata['semantic_path']}")
        print(f"   Original URL: {metadata['original_url']}")
        print(f"   Distance: {distance:.4f}")

    return client, collection


def query_chroma_db(collection_name: str = "pytorch_docs", query: str = None):
    """
    Query the ChromaDB database

    Args:
        collection_name: Name of the collection to query
        query: Query string
    """
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")

    # Get collection
    collection = client.get_collection(name=collection_name)

    if query:
        print(f"\n🔍 Querying: '{query}'")
        results = collection.query(
            query_texts=[query],
            n_results=5
        )

        print("\n📋 Results:")
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            print(f"\n{i}. Semantic Path: {metadata['semantic_path']}")
            print(f"   Original URL: {metadata['original_url']}")
            print(f"   Source Type: {metadata['source_type']}")
            print(f"   Element Type: {metadata['element_type']}")
            print(f"   Distance: {distance:.4f}")


if __name__ == "__main__":
    # Create ChromaDB from filtered JSON
    json_file = "output/hierarchical_crawl_pytorch_org_20250930_213016_filtered.json"

    client, collection = create_chroma_db(json_file, collection_name="pytorch_docs")

    # Example queries
    print("\n" + "="*60)
    print("Additional Query Examples")
    print("="*60)

    queries = [
        "tutorial deep learning",
        "contributor awards",
        "distributed training"
    ]

    for query in queries:
        query_chroma_db(collection_name="pytorch_docs", query=query)
        print("\n" + "-"*60)
