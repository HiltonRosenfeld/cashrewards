import streamlit as st
from config import Config
from functions import stream_parser
from functions import analyse_image_file
from functions import search_similar_images
from langchain.schema import HumanMessage, AIMessage



# Start with empty messages, stored in session state
if "messages" not in st.session_state:
    st.session_state.messages = []


# configures page settings
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    initial_sidebar_state=Config.INITIAL_SIDEBAR_STATE
    )

# page title
st.title(Config.PAGE_TITLE)

st.markdown("#### Select an image file to analyse.")

# Sidebar nav widgets
#with st.sidebar:   
#    # creates selectbox to pick the model we would like to use
#    image_model = st.selectbox('Which image model would you like to use?', Config.VISION_MODEL)


# displays file upload widget
with st.form('load_files'):
    uploaded_file = st.file_uploader("Choose image file", type=['png', 'jpg', 'jpeg'] )
    submitted = st.form_submit_button('Analyse image')
    if submitted:
        if uploaded_file is None:
            st.error('You must select an image file to analyse!')
            st.stop()
        # Analyse image and display output
        with st.status(":red[Processing image file...]", expanded=True) as status:
            st.write(":orange[Analysing Image File...]")

            description_stream = analyse_image_file(uploaded_file)
            st.session_state.image_description = st.write_stream(stream_parser(description_stream))

            st.write(":green[Done analysing image file]")


# Draw all messages, both user and agent so far (every time the app reruns)
if "messages" in st.session_state:
    for message in st.session_state.messages:
        st.chat_message(message.type).markdown(message.content)


if "image_description" in st.session_state:
    if chat_input := st.chat_input("What would you like to ask?"):

        # Add the prompt to messages in session state
        st.session_state.messages.append(HumanMessage(content=chat_input))
        
        # Display the user input on the page
        with st.chat_message("user"):
            st.markdown(chat_input)

        # Search for similar images based on image analysis output
        with st.chat_message("assistant"):
            st.write(":orange[Searching for similar images based on analysis output...]")

            search_stream = search_similar_images(st.session_state.image_description,chat_input)
            search_output = st.write_stream(stream_parser(search_stream))

            # Add the answer to the messages in session state
            st.session_state.messages.append(AIMessage(content=search_output))

            st.write(":green[Done searching for similar images]")
