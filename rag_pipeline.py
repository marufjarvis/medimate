import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import FakeListLLM
from langchain_core.prompts import ChatPromptTemplate

class MedicalRAG:
    def __init__(self, data_path="medical_knowledge.txt"):
        self.data_path = data_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self.llm = None
        self.setup()

    def setup(self):
        # 1. Load Data
        if not os.path.exists(self.data_path):
            with open(self.data_path, "w") as f:
                f.write("Aspirin is used to reduce fever and relieve mild to moderate pain.\n")
                f.write("Hypertension, or high blood pressure, is a common condition that can lead to heart disease.\n")
                f.write("Symptoms of COVID-19 include fever, cough, and loss of taste or smell.\n")
                f.write("Type 2 diabetes is a chronic condition that affects the way your body metabolizes sugar.\n")
        
        loader = TextLoader(self.data_path)
        docs = loader.load()

        # 2. Split Text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)

        # 3. Create Vector Store
        self.vector_store = Chroma.from_documents(splits, self.embeddings, persist_directory="./chroma_db")
        
        # 4. Define LLM
        responses = [
            "Based on the medical documents, here is what I found...", 
            "The clinical guidelines suggest...",
            "I retrieved this information from the knowledge base."
        ]
        self.llm = FakeListLLM(responses=responses * 100)
        
        if os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_community.chat_models import ChatOpenAI
                self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            except Exception:
                pass

    def query(self, user_query):
        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(user_query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Format prompt
        system_prompt = (
            "You are MediMate, a highly accurate and professional medical AI assistant. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know.\n\n"
            f"Context:\n{context}"
        )
        
        # Generate response using LLM
        try:
            # Try invoke for modern Langchain LLMs
            response = self.llm.invoke(f"{system_prompt}\n\nQuestion: {user_query}")
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = response
        except Exception as e:
            answer = f"Error generating response: {str(e)}"
            
        # Extract sources
        sources = [doc.page_content for doc in docs]
        
        return str(answer), sources
