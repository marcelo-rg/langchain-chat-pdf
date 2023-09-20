# Langchain Ask PDF 

This is a Streamlit application that allows you to load a PDF and ask questions about it using natural language. Additionally, you can upload your research papers and receive summaries crafted through customized prompts, ensuring consistently optimal results. The application uses a LLM to generate a response about your PDF. The LLM will not answer questions unrelated to the document.

## How it works

The application reads the PDF and splits the text into smaller chunks that can be then fed into a LLM. It uses OpenAI embeddings to create vector representations of the chunks. The application then finds the chunks that are semantically similar to the question that the user asked and feeds those chunks to the LLM to generate a response.

The application uses Streamlit to create the GUI and Langchain to deal with the LLM.

---

## Deployed Application

The app is already deployed on a website for easy access. Visit [website-link](https://langchain-chat-pdf.streamlit.app/) to start using it immediately.

## Local Installation and Usage

For those who prefer to run the application locally, follow these steps:

### Installation

1. **Clone this repository**:

```bash
git clone https://github.com/marcelo-rg/langchain-chat-pdf/
```

2. **Install the requirements**:

```
pip install -r requirements.txt
```

### Usage

To use the application, run the `💬AskYourPDF.py` file with the streamlit CLI (after having installed streamlit): 

```
streamlit run 💬AskYourPDF.py
```

