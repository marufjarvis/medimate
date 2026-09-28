import gradio as gr
from rag_pipeline import MedicalRAG
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

# Initialize the RAG pipeline
print("Initializing MediMate RAG Pipeline...")
rag = MedicalRAG()
print("Pipeline initialized!")

def chat_interface(message, history):
    """
    Function to handle the Gradio Chatbot interface.
    """
    answer, sources = rag.query(message)
    
    # Format the sources for display
    source_text = "\n\n**Retrieved Sources:**\n"
    for i, source in enumerate(sources):
        source_text += f"- {source.strip()}\n"
        
    full_response = f"{answer}{source_text}"
    return full_response

# Define the Gradio UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal")) as demo:
    gr.Markdown(
        """
        # 🩺 MediMate
        ### End-to-End Medical Chatbot Data Science Project
        This chatbot uses a **Retrieval-Augmented Generation (RAG)** pipeline to answer health-related queries based on a medical knowledge base.
        """
    )
    
    chatbot = gr.ChatInterface(
        chat_interface,
        examples=[
            "What is Aspirin used for?",
            "What are the symptoms of COVID-19?",
            "Tell me about Type 2 diabetes."
        ],
        description="Ask a medical question, and MediMate will search its vector database for relevant clinical information."
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
