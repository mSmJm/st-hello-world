import streamlit as st
import fitz  # PyMuPDF
from groq import Groq
import google.generativeai as genai
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fpdf import FPDF  # Importing the fpdf2 library for PDF generation
from PIL import Image
import spacy
import numpy as np

# Ground truth
meal1 = "Banana Oatmeal Cookies, Breakfast Banana Split, Popeye Power Smoothie"
# meal2 = 
# meal3 = 


# Initialize Groq client
groq_client = Groq(api_key="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")

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to preprocess the extracted PDF text
def preprocess_text(text):
    """Preprocess text: remove links, stopwords, punctuation, lemmatize, and lowercase."""
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", text)
    text = re.sub(r'\S+@\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    
    # Tokenize and remove stopwords, lemmatize
    words = text.split()
    processed_words = [
        lemmatizer.lemmatize(word) for word in words if word not in stop_words
    ]
    
    return " ".join(processed_words)

nlp = spacy.load("en_core_web_sm")

def VectrzTxt(text):
    doc = nlp(text)
    return doc.vector

def cos_sim (real_ans, ai_ans):
    # encode response into embeddings
    real_embedding = VectrzTxt(real_ans)
    ai_embedding = VectrzTxt(ai_ans)
    
    # calculate cosine similarity
    dot_product = np.dot(real_embedding, ai_embedding)
    magnitude = np.linalg.norm(real_embedding) * np.linalg.norm(ai_embedding)
    
    if magnitude == 0:
        return 0.0
    
    similarity_score = dot_product / magnitude
    return similarity_score


def calculate_containment_similarity(ground_truth, response):
    # Split ground truth into individual items
    ground_truth_items = [item.strip().lower() for item in ground_truth.split(",")]
    
    # Convert the chatbot response to lowercase for case-insensitive matching
    response_lower = response.lower()
    
    # Count matches of ground truth items in the response
    matches = sum(1 for item in ground_truth_items if item in response_lower)
    
    # Calculate containment similarity
    similarity = matches / len(ground_truth_items)
    return similarity


# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"


# Function to handle user input and generate a response
def process_user_input():
    user_input = st.session_state.user_input
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Choose model and generate response
        model_choice = st.session_state.get("model_choice", "Mixtral-8x7b-32768")
        if model_choice == "Mixtral-8x7b-32768":
            response = generate_groq_response(
                st.session_state.chat_history, model="mixtral-8x7b-32768")
        elif model_choice == "Gemini-1.5-flash-8b":
            response = generate_google_response(
                st.session_state.chat_history, context=st.session_state.pdf_text)
        elif model_choice == "Llama-3.1-8b-instant":
            response = generate_llama_response(
                st.session_state.chat_history, model="llama-3.1-8b-instant")
        else:
            response = "No valid model selected."
        
        cosine_similarity_value = cos_sim(meal1, preprocess_text(response))
        print("Cosine similarity for", model_choice, f"is {cosine_similarity_value:.5f}")
        
        # Calculate similarity
        containment_similarity = calculate_containment_similarity(meal1, response)
        print("Containment Similarity for", model_choice, f"is {containment_similarity:.5f}")
        
        # Append response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.session_state.user_input = ""

        
# Function to generate response using Google Generative AI
def generate_google_response(messages, context):
    try:
        genai.configure(api_key="AIzaSyCqUNbieAjm9bT90Z_I2qAN5UmbxE3xifg")
        model = genai.GenerativeModel("gemini-1.5-flash-8b")
        chat_history = "\n".join(
            [f"User: {m['content']}" if m['role'] == "user" else f"AI: {m['content']}" for m in messages]
        )
        context_with_history = f"{context}\n\nChat History:\n{chat_history}\n\n"
        response = model.generate_content(context_with_history)
        return response.text.strip()
    except Exception as e:
        return f"Error generating response: {e}"
    
# Function to generate response using Llama model via Groq
def generate_llama_response(messages, model):
    try:
        pdf_context = f"PDF Content:\n{st.session_state.pdf_text}\n\n"
        chat_history = "\n".join(
            [f"User: {m['content']}" if m['role'] == "user" else f"AI: {m['content']}" for m in messages]
        )
        prompt = (
            f"You are an assistant who strictly answers based only on the provided PDF content. "
            f"Do not speculate or provide additional information not explicitly mentioned in the PDF. "
            f"\n\n{pdf_context}\n\nChat History:\n{chat_history}\n\nUser Query: {messages[-1]['content']}"
        )
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=model
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"


# Function to generate response using Groq
def generate_groq_response(messages, model):
    try:
        response = groq_client.chat.completions.create(
            messages=messages, model=model
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"


# Function to export chat history as a PDF
def export_chat_to_pdf(chat_history, filename="Your_recipe.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    pdf.set_font("Times", size=12)
    
    # Add each chat entry to the PDF
    for chat in chat_history:
        role = "User" if chat["role"] == "user" else "AI"
        pdf.multi_cell(200, 10, txt=f"{role}: {chat['content']}", border=0, align='L')

    pdf.output(filename)
    
# Streamlit UI (including previous sections)
st.set_page_config(page_title="EVASH Meal Planner", layout="wide")

# Open the image file for the app logo (if you have one)
img = Image.open("image4.png")  # Replace with your image file path
img = img.resize((int(img.width * (300 / img.height)), 150))  # Resize to height=100px
st.image(img, use_column_width=True)

# Ensure session state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

# Layout for sections
col1, col2 = st.columns([2, 1])  # Center section (2) and right section (1)

# Center section: Chat interface
with col1:
    st.subheader("EVASH Meal Planner is ready to serve!")
    for chat in st.session_state.chat_history:
        if chat['role'] == 'user':
            st.markdown(
                f"""
                <div style="margin-bottom: 15px; padding: 15px; background-color: #e1d4ca; border: 1px solid #bbdefb; border-radius: 10px; display: flex; align-items: center;">
                    <div style="position: absolute; left: 10px; top: 10px;">
                        <img src="https://img.icons8.com/?size=100&id=BnpbsNUPf9Me&format=png&color=000000" alt="User" style="width: 40px; height: 40px;">
                    </div>
                    <div style="margin-left: 50px;">
                        <b>User:</b> {chat['content']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="margin-bottom: 15px; padding: 15px; background-color: #f8eee9; border: 1px solid #c8e6c9; border-radius: 10px; display: flex; align-items: center;">
                    <div style="position: absolute; left: 10px; top: 10px;">
                        <img src="https://img.icons8.com/?size=100&id=9Otd0Js4uSYi&format=png&color=000000" alt="AI" style="width: 40px; height: 40px;">
                    </div>
                    <div style="margin-left: 50px;">
                        <b>AI:</b> {chat['content']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    # Input field for user
    st.text_input(
        "What ingredient do you have?",
        key="user_input",
        placeholder="Place your ingredient here...",
        on_change=process_user_input
    )
    # Reset chat button
    if st.button("New Chat"):
        st.session_state.chat_history = []

    # Export chat history as PDF
    if st.button("Export Recipe"):
        if st.session_state.chat_history:
            export_chat_to_pdf(st.session_state.chat_history, filename="Recipe_for_you.pdf")
            st.success("Recipe saved successfully!")
        else:
            st.warning("No chat history to export.")

# Right section: Summary, file upload, and model selection
with col2:
    st.subheader("Upload Cookbook")
    uploaded_files = st.file_uploader(
        "Upload your cookbook (PDF)", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        combined_text = ""  # Initialize variable to store combined text

        for uploaded_file in uploaded_files:
            # Extract text from each uploaded PDF
            pdf_text = extract_text_from_pdf(uploaded_file)
            
            # Preprocess the extracted text
            preprocessed_text = preprocess_text(pdf_text)
            
            # Append content with separator showing file name
            combined_text += f"\n--- Content from {uploaded_file.name} ---\n{preprocessed_text}\n"

        # Store combined preprocessed text in session state
        st.session_state.pdf_text = combined_text

        # Show combined content
        st.subheader("Your Cookbook Content")
        with st.expander("Let see..."):
            st.text_area("Document Content", st.session_state.pdf_text, height=300)

    # Model selection dropdown
    st.subheader("Select Recipe Consultant")
    model_choice = st.selectbox(
        "Choose your recipe consultant:",
        ["Mixtral-8x7b-32768", "Gemini-1.5-flash-8b", "Llama-3.1-8b-instant"]
    )
    st.session_state.model_choice = model_choice

   # Display the social media share buttons when the 'Share' button is clicked
if st.button("Share"):
    # Using markdown to show icons and links
    st.markdown("""
                
        
    <div style="display: flex; justify-content: space-around; padding: 50px;">
 <!-- Instagram -->
 <a href="https://www.instagram.com" target="_blank">
     <img src="https://img.icons8.com/ios-filled/50/000000/instagram.png" alt="Instagram" style="width: 50px; height: 50px;">
 </a>    

    
    <!-- Facebook -->
    <a href="https://www.facebook.com" target="_blank">
        <img src="https://img.icons8.com/ios-filled/50/000000/facebook.png" alt="Facebook" style="width: 50px; height: 50px;">
    </a>
    <!-- WhatsApp -->
        <a href="https://wa.me" target="_blank">
            <img src="https://img.icons8.com/ios-filled/50/000000/whatsapp.png" alt="WhatsApp" style="width: 50px; height: 50px;">
        </a>
       
    </div>
    """, unsafe_allow_html=True)
