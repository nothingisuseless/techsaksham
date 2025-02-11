import streamlit as st
from ollama_project import load_pdf_data, split_docs, load_embedding_model, create_embeddings, load_qa_chain, get_response
from langchain.llms import Ollama
from langchain import PromptTemplate
import warnings
import tempfile

warnings.filterwarnings("ignore")

def healthcare_chatbot():
    # Streamlit UI
    st.title("Healthcare Assistant Chatbot")
    st.write("Upload a PDF document and ask questions related to the healthcare policy.")

    # File uploader for PDF file
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            pdf_path = tmp_file.name

        st.spinner("Processing the document...")

        # Loading orca-mini from Ollama
        llm = Ollama(model="orca-mini", temperature=0.7)

        # Loading the Embedding Model
        embed = load_embedding_model(model_path="all-MiniLM-L6-v2")

        # Loading and splitting the documents
        docs = load_pdf_data(file_path=pdf_path)
        documents = split_docs(documents=docs)

        # Creating the vectorstore and retriever
        vectorstore = create_embeddings(documents, embed)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        st.success("Document processed successfully!")

        # Prepare the prompt template
        template = """
        ### System:
        You are a healthcare assistant chatbot. You have to answer the user's questions using only the context \
        provided to you. If you don't know the answer, just say you don't know. Don't try to make up an answer.

        ### Context:
        {context}

        ### User:
        {question}

        ### Response:
        """

        # Creating the prompt from the template
        prompt = PromptTemplate.from_template(template)

        # Creating the QA chain
        chain = load_qa_chain(retriever, llm, prompt)

        # Get user question input
        question = st.text_input("Ask a question:")

        if question:
            with st.spinner("Generating response..."):
                try:
                    # Get the response for the user's question
                    result = get_response(question, chain)
                    
                    # Display the answer in the Streamlit frontend
                    if result:
                        st.write(f"**Answer:** {get_response(question, chain)}")
                    else:
                        st.write("Sorry, I couldn't find an answer.")
                except Exception as e:
                    # Display the error message in case of an issue
                    st.write(f"Error generating response: {e}")

# Run the app
if __name__ == "__main__":
    healthcare_chatbot()
