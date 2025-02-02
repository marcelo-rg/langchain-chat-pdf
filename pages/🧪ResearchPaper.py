import streamlit as st
import requests
import re
import os
from constants import research_prompts, similarity_search_queries, pdf_template
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from pypdf import PdfReader

persist_directory = os.environ.get('PERSIST_DIRECTORY')

def semantic_search(num_queries):
    unique_chunks = set()
    
    for key in list(similarity_search_queries)[:num_queries]:
        query = similarity_search_queries[key]
        results = st.session_state.knowledge_base.similarity_search(query, k=2)
        
        for chunk in results:
            chunk_str = str(chunk)
            if chunk_str not in unique_chunks:
                unique_chunks.add(chunk_str)

    return unique_chunks

def run_chain(option):
    # Perform semantic search
    context = semantic_search(3)
    prompt_template = ChatPromptTemplate.from_template(template= pdf_template)

    chain = LLMChain(
        llm= ChatOpenAI(),
        prompt=prompt_template,
    )

    output = chain.run(prompt = option, context=context)

    return output


def upload_pdf():

    # If a PDF exists in the session state, display its name and a "Remove PDF" button
    if 'uploaded_pdf' in st.session_state:
        col1, col2 = st.columns([15, 4])  # Creating two columns with different widths
        # col2.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)  # Add a CSS white space to align the button
        col1.write(f"Uploaded file: {st.session_state.uploaded_pdf.name}")

        if col2.button('Remove PDF'):
            del st.session_state.uploaded_pdf
            del st.session_state.knowledge_base
            return None

        return st.session_state.uploaded_pdf
    
    # Otherwise, display the file uploader
    else:
        uploaded_pdf = st.file_uploader("Upload your PDF", type="pdf")

        # If the user uploads a new PDF
        if uploaded_pdf:
            st.session_state.uploaded_pdf = uploaded_pdf
            return uploaded_pdf

        return None
    
def extract_text_from_pdf(pdf):
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    return None

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore directory exists and is not empty
    """
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        return True
    return False


def create_embeddings(chunks, embeddings):
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base

# def create_persistent_embeddings(chunks, embeddings):
#     # Check if vectorstore exists
#     if does_vectorstore_exist(persist_directory):
#         print(f"Embeddings already exist at {persist_directory}. No need to add anything.")
#         knowledge_base = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
#     else:
#         # Create and store locally vectorstore using Chroma
#         print("Creating new vectorstore")
#         knowledge_base = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=persist_directory)
#         knowledge_base.persist()
#         print(f"Ingestion complete! Embeddings created for uploaded file.")
#     return knowledge_base

    
def select_prompt(prompts):
    # st.header("Select a Prompt")
    col1, col2 = st.columns([10, 4])  # Creating two columns with different widths
    col2.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)	
    option = col1.selectbox('Select the research prompt to use',prompts)

    if col2.button('Run'):
        st.session_state['selected_prompt'] = option

    return option


def is_valid_api_key_format(secret_key):
    return re.match(r'^sk-[a-zA-Z0-9]{32,}$', secret_key)


def check_api_key_authorization(secret_key):
    headers = {
        'Authorization': f'Bearer {secret_key}'
    }

    response = requests.get("https://api.openai.com/v1/engines", headers=headers)
    return response.status_code == 200


def validate_api_key(secret_key):
    if secret_key:
        if not is_valid_api_key_format(secret_key):
            st.error("Invalid API key format.")
            return

        if not check_api_key_authorization(secret_key):
            st.error("Unauthorized API key.")
            return

        # st.success("Your API key is valid.")
        return True



def get_api_key():
    if 'api_key' not in st.session_state or not st.session_state.api_key:
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if api_key:
            is_valid = validate_api_key(api_key)
            if is_valid:
                st.success("API key is valid!")
                st.session_state.api_key = api_key
                return True  # Indicates the key is valid and you can proceed
            else:
                st.error("Invalid API key. Please try again.")
                return False  # Indicates the key is invalid
    else:
        return True  # If API key is already in session state



def main():
    st.set_page_config(
        page_title="Research Paper",
        # page_icon="🧪"
    )
    st.header("Fast Query Your Paper 🧪")

    if get_api_key():
        os.environ['OPENAI_API_KEY'] = st.session_state.api_key
        embeddings = OpenAIEmbeddings()

        # 1. Upload the PDF
        pdf = upload_pdf()
        
        # 2. Extract text from the PDF, if available
        if pdf:
            text = extract_text_from_pdf(pdf)

            # If the text is successfully extracted, proceed
            if text:
                # Split the text into chunks
                chunks = get_text_chunks(text)

                # Check if embeddings are already in the session state
                if 'knowledge_base' not in st.session_state:
                    st.session_state.knowledge_base = create_embeddings(chunks, embeddings)

        # 3. Get a list of research prompts
        prompts = research_prompts["Summarizing and Analysis"][:3]

        # 4. Display prompts and get the selected prompt option
        option = select_prompt(prompts)

        # 5. If 'Run' button is clicked in `select_prompt`, process the prompt
        if 'selected_prompt' in st.session_state:
            # Process the selected prompt using the run_chain function
            output = run_chain(option)

            # Display the output
            st.write("Output:", output)

            # Optionally clear the session state
            del st.session_state['selected_prompt']



if  __name__ == '__main__':
    main()