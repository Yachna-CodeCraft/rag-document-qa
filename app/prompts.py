"""Small versioned prompt library.

Keeping prompts in one place (instead of scattered strings) makes them easy to
test, compare and improve. Add new versions instead of editing old ones.
"""
from langchain_core.prompts import ChatPromptTemplate

PROMPTS = {
    "qa_v1": ChatPromptTemplate.from_messages([
        ("system",
         "You are a careful assistant that answers questions about the user's documents.\n"
         "Rules:\n"
         "1. Use ONLY the context below. Do not use outside knowledge.\n"
         "2. If the context does not contain the answer, reply exactly: "
         "\"I couldn't find this in the uploaded documents.\"\n"
         "3. Cite sources inline like [filename, p.N] after the sentences they support.\n"
         "4. Be concise and factual.\n\n"
         "Context:\n{context}"),
        ("human", "{question}"),
    ]),
}

DEFAULT_PROMPT = "qa_v1"
NOT_FOUND_MESSAGE = "I couldn't find this in the uploaded documents."
