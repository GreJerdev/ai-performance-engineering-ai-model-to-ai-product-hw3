import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class Embedding:

    def __init__(self, docs:list[Document]):
            # 2. Create embeddings
        self.embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )

        # 1. Convert dataframe column to LangChain documents

        # 3. Create FAISS vector store from dataframe text
        self.vectorstore = FAISS.from_documents(
            documents=docs,
            embedding=self.embedding_model
        )


    # 4. Search similar rows
    
    def similarity_search(self, query: str, k: int = 2) -> list[Document]:
        results = self.vectorstore.similarity_search(
            query,
            k=2
        )

        return results


    # from langchain_openai import ChatOpenAI
    # from langchain_core.prompts import ChatPromptTemplate

    # retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # prompt = ChatPromptTemplate.from_template("""
    # Answer using only the context.

    # Context:
    # {context}

    # Question:
    # {question}
    # """)

    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # def rag_answer(question: str):
    #     docs = retriever.invoke(question)
    #     context = "\n\n".join(doc.page_content for doc in docs)

    #     response = llm.invoke(
    #         prompt.format_messages(
    #             context=context,
    #             question=question
    #         )
    #     )

    #     return response.content