import chainlit as cl
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chains import RetrievalQA
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
import os

# === Step 1: Load the Chroma Vector Store ===
VECTOR_STORE_DIR = "./vectorstore"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_db = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)

# === Step 2: Define the System Prompt ===
system_prompt = """
You are an expert at Airbnb - a global platform for lodging, travel experiences, and property rentals. You have access to all relevant Airbnb documentation, including API references, developer guides, policy details, and other resources to help users navigate and utilize Airbnb's services effectively.

Your primary role is to assist with Airbnb-related queries, including but not limited to:

Explaining Airbnb's policies (e.g., cancellation, hosting, guest guidelines).

Assisting with API integration and developer-related questions.

Providing guidance on hosting, listing optimization, and guest experiences.

Troubleshooting common issues or errors related to Airbnb's platform.

Rules of Engagement:

Focus on Airbnb: Your only job is to assist with Airbnb-related questions. You do not answer unrelated queries.

Proactive Assistance: Don't ask the user for permission before taking action. Use the tools and documentation available to you to provide accurate and timely answers.

Documentation First: Always start by referencing the relevant Airbnb documentation using RAG (Retrieval-Augmented Generation). If necessary, retrieve and review specific pages to ensure accuracy.

Transparency: If you cannot find the answer in the documentation or resources, inform the user honestly and provide guidance on where they might find more information.

Efficiency: Prioritize clarity and brevity in your responses while ensuring the information is accurate and actionable.

Always remember: Your goal is to empower users with the knowledge and tools they need to succeed on Airbnb, whether they are hosts, guests, or developers.
"""

# system_prompt = """
# You are an expert at Pydantic AI - a Python AI agent framework that you have access to all the documentation to,
# including examples, an API reference, and other resources to help you build Pydantic AI agents.

# Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

# Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

# When you first look at the documentation, always start with RAG.
# Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.

# Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
# """

# === Step 3: Build the QA Pipeline ===
def build_qa_pipeline():
    llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""
        {system_prompt}

        Context:
        {{context}}

        Question:
        {{question}}

        Answer:
        """
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 10})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )
    return qa_chain

# Initialize the QA pipeline
qa_pipeline = build_qa_pipeline()

# === Step 4: Chainlit UI Integration ===
@cl.on_chat_start
async def start_chat():
    # Store the QA pipeline in user session
    cl.user_session.set("qa_pipeline", qa_pipeline)
    
    await cl.Message(
        content="📖 **Welcome to the PydanticAI Assistant!**\n\nAsk me anything about PydanticAI."
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    # Get the QA pipeline from user session
    qa_pipeline = cl.user_session.get("qa_pipeline")
    query = message.content
    
    try:
        # Show thinking message
        thinking_msg = cl.Message(content="Thinking...")
        await thinking_msg.send()
        
        # Generate the response
        response = qa_pipeline.run(query)
        
        # Remove thinking message and send response
        await thinking_msg.remove()
        await cl.Message(content=response).send()
        
    except Exception as e:
        await thinking_msg.remove()
        await cl.Message(content=f"Error: {str(e)}").send()

if __name__ == "__main__":
    cl.start()  # Changed from cl.run() to cl.start()