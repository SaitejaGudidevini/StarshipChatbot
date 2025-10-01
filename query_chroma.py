"""
Simple script to query ChromaDB
"""
import chromadb

def browse_collection():
    """Browse all documents in the collection"""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="pytorch_docs")

    total = collection.count()
    print(f"📊 Total documents in ChromaDB: {total}")
    print("\n" + "="*70)

    # Get first 10 documents
    results = collection.get(limit=10, include=['metadatas', 'documents'])

    print(f"\n📋 First 10 documents:\n")
    for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas']), 1):
        print(f"{i}. {meta['semantic_path']}")
        print(f"   URL: {meta['original_url']}")
        print(f"   Type: {meta['source_type']} / {meta['element_type']}")
        print()

def search_by_text(query):
    """Search ChromaDB by text query"""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="pytorch_docs")

    print(f"🔍 Searching for: '{query}'")
    print("="*70)

    results = collection.query(
        query_texts=[query],
        n_results=10
    )

    print(f"\n📋 Top 10 results:\n")
    for i, (meta, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0]), 1):
        print(f"{i}. {meta['semantic_path']}")
        print(f"   URL: {meta['original_url']}")
        print(f"   Type: {meta['source_type']} / {meta['element_type']}")
        print(f"   Similarity: {1 - distance:.4f}")
        print()

def filter_by_type(element_type):
    """Filter documents by element type"""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="pytorch_docs")

    print(f"📌 Filtering by element_type: '{element_type}'")
    print("="*70)

    results = collection.get(
        where={"element_type": element_type},
        limit=20,
        include=['metadatas']
    )

    print(f"\nFound {len(results['metadatas'])} documents:\n")
    for i, meta in enumerate(results['metadatas'], 1):
        print(f"{i}. {meta['semantic_path']}")
        print(f"   URL: {meta['original_url']}")
        print()

def filter_by_url(url_pattern):
    """Filter documents by original URL"""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="pytorch_docs")

    print(f"📌 Filtering by URL containing: '{url_pattern}'")
    print("="*70)

    results = collection.get(
        where={"original_url": {"$contains": url_pattern}},
        limit=20,
        include=['metadatas']
    )

    print(f"\nFound {len(results['metadatas'])} documents:\n")
    for i, meta in enumerate(results['metadatas'], 1):
        print(f"{i}. {meta['semantic_path']}")
        print(f"   URL: {meta['original_url']}")
        print()

if __name__ == "__main__":
    print("\n🗄️  ChromaDB Browser\n")

    while True:
        print("\nChoose an option:")
        print("1. Browse first 10 documents")
        print("2. Search by text")
        print("3. Filter by element type (heading/link/button)")
        print("4. Filter by URL pattern")
        print("5. Exit")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == "1":
            browse_collection()
        elif choice == "2":
            query = input("Enter search query: ").strip()
            if query:
                search_by_text(query)
        elif choice == "3":
            etype = input("Enter element type (heading/link/button): ").strip()
            if etype:
                filter_by_type(etype)
        elif choice == "4":
            url = input("Enter URL pattern (e.g., 'tutorials'): ").strip()
            if url:
                filter_by_url(url)
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Try again.")
