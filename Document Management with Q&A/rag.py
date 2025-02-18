import streamlit as st
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack import Document
from transformers import BertTokenizer, BertModel, pipeline
from PIL import Image
import pytesseract
import torch

image_path_1 = r""
image_path_2 = r""

# Load the BERT model and tokenizer for text embeddings
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
bert_model = BertModel.from_pretrained('bert-base-uncased', local_files_only=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the question-answering pipeline
qa_pipeline = pipeline(task='question-answering', model='distilbert/distilbert-base-cased-distilled-squad', device=device)

# Define a fixed embedding dimension (768 for BERT)
EMBEDDING_DIM = 768

# Initialize the document store and retriever
document_store = InMemoryDocumentStore()
retriever = InMemoryEmbeddingRetriever(document_store=document_store)

# Function to convert tensor to list of floats
def tensor_to_list(tensor):
    return tensor.cpu().numpy().flatten().tolist()

# Function to pad or truncate embeddings to a fixed size
def pad_or_truncate(embedding, size):
    if len(embedding) > size:
        return embedding[:size]
    else:
        return embedding + [0] * (size - len(embedding))

# Function to extract text from an image using OCR
def extract_text_from_image(image=None, image_path=None):
    if image:
        text = pytesseract.image_to_string(image)
        return text
    else:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text

# Function to add documents to the store (text extracted from image or provided as text)
def add_document(doc_id, text=None, image=None, image_path=None):
    # Extract text from the image if provided
    if image:
        extracted_text = extract_text_from_image(image=image)
    elif image_path:
        extracted_text = extract_text_from_image(image_path=image_path)
    else:
        extracted_text = text

    if extracted_text:
        text_inputs = bert_tokenizer(extracted_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            text_embeddings = bert_model(**text_inputs).last_hidden_state.mean(dim=1).squeeze()
        text_embedding_list = tensor_to_list(text_embeddings)
    else:
        text_embedding_list = [0] * EMBEDDING_DIM

    # Store the document with text embeddings
    combined_embeddings = pad_or_truncate(text_embedding_list, EMBEDDING_DIM)
    document = Document(
        content=extracted_text,
        embedding=combined_embeddings,
        meta={"doc_id": doc_id}
    )
    document_store.write_documents([document])

# Function to retrieve the most relevant document based on the query
def retrieve_relevant_document(question, top_k=1):
    text_inputs = bert_tokenizer(question, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        text_embeddings = bert_model(**text_inputs).last_hidden_state.mean(dim=1).squeeze()
    query_embedding_list = tensor_to_list(text_embeddings)

    # Normalize embeddings for the query and find the top_k results
    combined_embeddings = pad_or_truncate(query_embedding_list, EMBEDDING_DIM)
    results = retriever.run(query_embedding=combined_embeddings, top_k=top_k)
    return results['documents'][0] if results['documents'] else None

# Function to answer question based on the extracted text from the image
def answer_question_from_text(question, document):
    # Extract the text content from the retrieved document
    extracted_text = document.content
    if not extracted_text:
        return "No text available in the document."

    # Use the QA pipeline to answer the question based on the text
    answer = qa_pipeline(question=question, context=extracted_text)
    return answer['answer']

add_document("doc1", image_path=image_path_1)
add_document("doc2", image_path=image_path_2)

# Streamlit UI
st.title("📑 Document Management and Question Answering")
st.markdown("""
<style>
    .reportview-container {
        background: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    h1 {
        color: #4e79a7;
    }
</style>
""", unsafe_allow_html=True)

# Upload file or provide text
with st.sidebar:
    st.header("Upload or Enter Document")
    uploaded_file = st.file_uploader("Choose an image file...", type=["png", "jpg", "jpeg"])
    text_input = st.text_area("Or enter text directly:")
    doc_id = st.text_input("Enter a document ID:")

    if st.button("Add Document"):
        if doc_id:
            image = Image.open(uploaded_file) if uploaded_file else None
            add_document(doc_id, text=text_input, image=image)
            st.success(f"✅ Document with ID {doc_id} added successfully!")
        else:
            st.error("❌ Please provide a document ID.")

# Query input
st.subheader("🔍 Ask a Question")
query_text = st.text_input("Enter your query here:")

if st.button("Get Answer"):
    if query_text:
        relevant_document = retrieve_relevant_document(query_text)
        if relevant_document:
            answer = answer_question_from_text(query_text, relevant_document)
            st.write(f"**Answer:** {answer}")
        else:
            st.warning("No relevant document found.")
    else:
        st.error("❌ Please enter a query.")
