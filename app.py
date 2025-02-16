from dotenv import load_dotenv
import os
import streamlit as st
from PIL import Image
import google.generativeai as genai
from summary import summarize_text
from sentiment import get_sentiment
from translate import translate_text 
from Pdf_chat import get_pdf_text, get_text_chunks, get_vector_store, user_input 

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def get_gemini_response(input_text=None, image=None):
    model = genai.GenerativeModel("gemini-1.5-pro")
    if input_text and image:
        response = model.generate_content([input_text, image])
    elif input_text:
        response = model.generate_content(input_text)
    elif image:
        response = model.generate_content(["Analyze this image", image])
    else:
        return "Please provide either text input, an image, or both."

    st.session_state.conversation.append({
        "user": input_text if input_text else "[Image Uploaded]",
        "assistant": response.text,
        "image": image if image else None
    })

    for entry in st.session_state.conversation:
        messages.chat_message("user").write(entry['user'])
        if entry["image"]:
            st.image(entry["image"], caption="Uploaded Image", use_column_width=True)
        messages.chat_message("assistant").write(entry['assistant'])

    return response.text

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if 'clicked' not in st.session_state:
    st.session_state.clicked = False
    
if "selected_option" not in st.session_state:
    st.session_state.selected_option = None 

def select_option(option):
    if st.session_state.selected_option == option:
        st.session_state.selected_option = None 
    else:
        st.session_state.selected_option = option 
    st.session_state.clicked = False 
    

def toggle_clicked():
    st.session_state.clicked = not st.session_state.clicked
    st.session_state.selected_option = None

st.set_page_config(page_title="Multimodal Chatbot")


with st.sidebar:
    st.markdown("**Note:** You can select only one option at a time. Deselect to choose another.")
    st.header("Options ➤")
    summarize_option = st.checkbox("Summarize Text 📝", value=(st.session_state.selected_option == "summarize"), on_change=select_option, args=("summarize",))
    sentiment_option = st.checkbox("Analyze Sentiment 🕊", value=(st.session_state.selected_option == "sentiment"), on_change=select_option, args=("sentiment",))
    translate_option = st.checkbox("Translate Text 🔄", value=(st.session_state.selected_option == "translate"), on_change=select_option, args=("translate",))
    pdf_chat_option = st.checkbox("PDF Chat 📕", value=(st.session_state.selected_option == "pdf_chat"), on_change=select_option, args=("pdf_chat",))


    if translate_option:
        source_language = st.selectbox(
            "Select Source Language",
            ["English", "French", "German", "Spanish", "Hindi", "Bengali"],
            help="Language of the input text."
        )
        target_language = st.selectbox(
            "Select Target Language",
            ["English", "French", "German", "Spanish", "Hindi", "Bengali"],
            help="Language to translate the text into."
        )
    else:
        source_language = None
        target_language = None

    if pdf_chat_option:
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
                st.balloons()

st.markdown(
    """
    <style>
    /* Title Styling */
    .title-text {
        color: #ffcc00; /* Gold color */
        font-size: 36px;
        font-weight: bold;
        text-align: left;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Keyframes for Gradient Animation */
    @keyframes animatedBackground {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Gradient Background with Animation */
    .stApp {
        background-image: radial-gradient(circle farthest-corner at 10% 20%, rgba(37,145,251,0.98) 0.1%, rgba(0,7,128,1) 99.8%);
        background-size: 200% 200%;
        animation: animatedBackground 8s ease infinite;
        padding: 20px;
        border-radius: 15px;
        overflow: hidden;
    }

    .stButton button {
        background-image: linear-gradient(to right, #EB5757 0%, #000000 51%, #EB5757 100%) !important;
        text-transform: uppercase;
        transition: 0.5s;
        background-size: 200% auto;
        color: white !important;
        font-weight: bold !important;
        box-shadow: 0 0 5px #eee;
        border-radius: 10px !important;
        border: none !important;
        cursor: pointer;
    }

    .stButton button:hover {
        background-position: right center !important;
        color: #fff !important;
        background-color: none !important;
        text-decoration: none !important;
    }
    </style>
    """,
    
    unsafe_allow_html=True
)

col1, col2 = st.columns([5, 1], gap="large", vertical_alignment="bottom")

with col1:
    st.markdown('<h1 class="title-text">🤖 Multimodal Chatbot</h1>', unsafe_allow_html=True)
with col2:
    if st.button("Upload Image" if not st.session_state.clicked else "Close Image", on_click=toggle_clicked):
        st.session_state.clicked = not st.session_state.clicked

uploaded_image = None
if st.session_state.clicked:
    uploaded_file = st.file_uploader("Upload an image (optional):", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        uploaded_image = Image.open(uploaded_file)

if st.session_state.clicked and st.session_state.selected_option:
    st.warning("You cannot use image upload while an option is selected. Please deselect the current option.")

messages = st.container(border=True, height=400)

st.subheader("Input Box")

input_text = st.text_area("Enter a text prompt, question, or paragraph:")
submit_response = st.button("Generate Response 🪄")

if submit_response:
    if input_text or uploaded_image:
        action_performed = False 
        
        if summarize_option and input_text:
            summary = summarize_text(input_text)
            st.session_state.conversation.append({
                "user": input_text,
                "assistant": summary
            })
            action_performed = True
            for entry in st.session_state.conversation:
                messages.chat_message("user").write(entry['user'])
                messages.chat_message("assistant").write(entry['assistant'])

        if sentiment_option and input_text:
            sentiment = get_sentiment(input_text)
            st.session_state.conversation.append({
                "user": input_text,
                "assistant": sentiment
            })
            action_performed = True
            for entry in st.session_state.conversation:
                messages.chat_message("user").write(entry['user'])
                messages.chat_message("assistant").write(entry['assistant'])

        if translate_option and input_text:
            if source_language and target_language:
                translated_text = translate_text(input_text, source_language, target_language)
                st.session_state.conversation.append({
                    "user": input_text,
                    "assistant": translated_text
                })
                action_performed = True
                for entry in st.session_state.conversation:
                    messages.chat_message("user").write(entry['user'])
                    messages.chat_message("assistant").write(entry['assistant'])
            else:
                st.error("Please select both source and target languages.")

        if pdf_chat_option and input_text:    
            pdf_question = input_text
            if pdf_question:
                response = user_input(pdf_question)
                st.session_state.conversation.append({
                    "user": pdf_question,
                    "assistant": response
                })
                action_performed = True
                for entry in st.session_state.conversation:
                    messages.chat_message("user").write(entry['user'])
                    messages.chat_message("assistant").write(entry['assistant'])
                    
        if not action_performed:
            response = get_gemini_response(input_text, uploaded_image)
    else:
        st.write("Please provide either a text input or an image.")
    
    
if summarize_option or sentiment_option or translate_option or pdf_chat_option:
    st.info("You have selected an option in the sidebar. Proceed with your input for the corresponding action.")