from scripts.llm import get_llm_response

KNOWLEDGE_BASE = {
    "doc1": {
        "title": "Project Chimera Overview",
        "content": (
            "Project Chimera is a research initiative focused on developing "
            "novel bio-integrated interfaces. It aims to merge biological "
            "systems with advanced computing technologies."
        )
    },
    "doc2": {
        "title": "Chimera's Neural Interface",
        "content": (
            "The core component of Project Chimera is a neural interface "
            "that allows for bidirectional communication between the brain "
            "and external devices. This interface uses biocompatible "
            "nanomaterials."
        )
    },
    "doc3": {
        "title": "Applications of Chimera",
        "content": (
            "Potential applications of Project Chimera include advanced "
            "prosthetics, treatment of neurological disorders, and enhanced "
            "human-computer interaction. Ethical considerations are paramount."
        )
    }
}

def naive_generation(query):
    prompt = f"Answer directly the following query: {query}"
    return get_llm_response(prompt)

def rag_retrieval(query, documents):
    query_words = set(query.lower().split())
    relevant = []

    for doc in documents.values():
        # tokenize and measure overlap on content
        content_words = set(doc["content"].lower().split())
        overlap = len(query_words & content_words)
        if overlap > 0:
            relevant.append(doc)

    # return all matching docs (could sort by overlap if you like)
    return relevant

def rag_generation(query, docs):
    if docs:
        # build one big snippet out of every relevant doc
        sections = []
        for d in docs:
            sections.append(f"{d['title']}:\n{d['content']}")
        snippet = "\n\n".join(sections)

        prompt = (
            f"Using the following information:\n{snippet}\n\n"
            f"Answer the query: {query}"
        )
    else:
        prompt = f"No relevant information found. Answer directly: {query}"

    return get_llm_response(prompt)

if __name__ == "__main__":
    query = "What are the applications of Project Chimera?"
    print("Naive approach:", naive_generation(query))

    retrieved_docs = rag_retrieval(query, KNOWLEDGE_BASE)
    print(f"Retrieved {len(retrieved_docs)} relevant document(s).")

    print("RAG approach:", rag_generation(query, retrieved_docs))
