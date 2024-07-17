import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain.prompts.chat import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import AstraDBChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
from langchain.schema.runnable import RunnableMap
#from langchain.callbacks.base import BaseCallbackHandler

#import tempfile

import streamlit as st

from dotenv import load_dotenv
load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_API_ENDPOINT = os.environ["ASTRA_DB_API_ENDPOINT"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

ASTRA_DB_KEYSPACE = "cashrewards"
ASTRA_DB_COLLECTION = "goodguys_ai_description"

os.environ["LANGCHAIN_PROJECT"] = "cashrewards"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


print("Started")



#################
### Constants ###
#################

# Define the number of docs to retrieve from the vectorstore and memory
top_k_vectorstore = 8
top_k_memory = 1

###############
### Globals ###
###############

global embedding
global vectorstore
global retriever
global model
global chat_history
global memory
global policy


#############
### Login ###
#############
# Close off the app using a password
def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("credentials"):
            st.caption('Using a unique name will keep your content seperate from other users.')
            st.text_input('Username', key='username')
            #st.text_input('Password', type='password', key='password')
            st.form_submit_button('Login', on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        #if st.session_state['username'] in st.secrets['passwords'] and hmac.compare_digest(st.session_state['password'], st.secrets.passwords[st.session_state['username']]):
        if len(st.session_state['username']) > 5:
            st.session_state['password_correct'] = True
            st.session_state.user = st.session_state['username']
            #del st.session_state['password']  # Don't store the password.
        else:
            st.session_state['password_correct'] = False

    # Return True if the username + password is validated.
    if st.session_state.get('password_correct', False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error('😕 Username must be 6 or more characters')
    return False

def logout():
    del st.session_state.password_correct
    del st.session_state.user
    del st.session_state.messages
    #load_chat_history.clear()
    #load_memory.clear()
    load_retriever.clear()


# Check for username/password and set the username accordingly
if not check_password():
    st.stop()  # Do not continue if check_password is not True.

username = st.session_state.user


#################
### Functions ###
#################

# handles stream response back from LLM
def stream_parser(stream):
    for chunk in stream:
        #yield chunk["response"]                # for Ollama
        #yield chunk.choices[0].delta.content   # for openAI direct
        yield chunk.content                     # for openAI via Langchain



#######################
### Resources Cache ###
#######################

# Cache OpenAI Embedding for future runs
@st.cache_resource(show_spinner='Getting the Embedding Model...')
def load_embedding():
    print("load_embedding")
    # Get the OpenAI Embedding
    return OpenAIEmbeddings(model="text-embedding-3-small")
    

# Cache Vector Store for future runs
@st.cache_resource(show_spinner='Getting the Vector Store from Astra DB...')
def load_vectorstore():
    print(f"load_vectorstore: {ASTRA_DB_API_ENDPOINT} / {ASTRA_DB_KEYSPACE} / {ASTRA_DB_COLLECTION}")
    # Get the load_vectorstore store from Astra DB
    return AstraDBVectorStore(
        embedding=embedding,
        namespace=ASTRA_DB_KEYSPACE,
        collection_name=ASTRA_DB_COLLECTION,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
    )

# Cache Retriever for future runs
@st.cache_resource(show_spinner='Getting the retriever...')
def load_retriever():
    print("load_retriever")
    # Get the Retriever from the Vectorstore
    return vectorstore.as_retriever(
        search_kwargs={
            "k": top_k_vectorstore,
            #"filter": {"brand": "Apple"},
        },
    )

# Cache Chat Model for future runs
@st.cache_resource(show_spinner='Getting the Chat Model...')
def load_model(model_id="openai.gpt-3.5"):
    print(f"load_model: {model_id}")
    # if model_id contains 'openai' then use OpenAI model
    if '3.5' in model_id:
        gpt_version = 'gpt-3.5-turbo'
    else:
        gpt_version = 'gpt-4-turbo'
    return ChatOpenAI(
        temperature=0.2,
        model=gpt_version,
        streaming=True,
        verbose=True
        )

# Cache Chat History for future runs
@st.cache_resource(show_spinner='Getting the Message History from Astra DB...')
def load_chat_history(username):
    print("load_chat_history")
    return AstraDBChatMessageHistory(
        session_id=username,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        namespace=ASTRA_DB_KEYSPACE,
    )

@st.cache_resource(show_spinner='Getting the Message History from Astra DB...')
def load_memory():
    print("load_memory")
    return ConversationBufferWindowMemory(
        #chat_memory=chat_history,
        return_messages=True,
        k=top_k_memory,
        memory_key="chat_history",
        input_key="question",
        output_key='answer',
    )

@st.cache_resource()
def load_policy(policy_id="the good guys"):
    print(f"load_policy: {policy_id}")
    # remove spaces in selected_merchant
    policy_id = policy_id.replace(" ", "")
    # read the policy file policy.txt
    policy_file = f"policy_{policy_id}.txt"
    return Path(policy_file).read_text()




# Cache prompt
# Use the previous chat history to answer the question:
# {chat_history}
# When I write BEGIN DIALOGUE you will enter this role, and all further input from the "Human:" will be from a user seeking a sales or customer support question.
# If you don't know the answer, just say 'I do not know the answer'.

@st.cache_data()
def load_prompt():
    print("load_prompt")
    template = """You are a shopping assistant for a business that provides cash rewards when shopping. 
Your goal is to assist the user based on their query, either by recommending the best products based on the product details provided or by explaining the rewards process and policies based on the company's cash reward policy.
Here are the details:

### Cash Reward Policy:
{policy}

### Product Details:
{context}

### User Query:
{question}

### Instructions:
Based on the user query, provide an appropriate response using only the information provided in the cash reward policy and the product details:
- If the user asks a question about products, recommend the best products from the list provided, considering factors such as category, brand, and price.
    Include an image of the product taken from the img attribute in the metadata.
    Include the price of the product taken from the price attribute in the metadata.
    Include the amount of cash rewards the user will receive when purchasing based on the policy percentage.
    Include a link to buy each item you recommend taken from the source attribute in the metadata. Here is a sample buy link:
    Buy Now: [Product Name](https://www.thegoodguys.com.au/product-name)
- If the user asks a question about how the rewards process works or about the rewards policies, answer using the information from the cash rewards policy. Do not recommend any products.

Respond accordingly to the user's query using only the information provided.
"""

    return ChatPromptTemplate.from_messages([("system", template)])



#####################
### Session state ###
#####################

# Start with empty messages, stored in session state
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hi, How can I help you today?")]


############
### Main ###
############
st.set_page_config(
    page_title="CashRewards Shopping Assistant",
    initial_sidebar_state="expanded",
    #initial_sidebar_state="collapsed"
    )

# Write the welcome text
#st.markdown(Path('welcome.md').read_text())

# DataStax logo
with st.sidebar:
    st.image('./public/logo.svg')
    st.text('')

# Logout button
with st.sidebar:
    st.button(f"Logout '{username}'", on_click=logout)

# Initialize
with st.sidebar:
    embedding = load_embedding()
    #model = load_model()  # the model is loaded in the sidebar "model selection"
    vectorstore = load_vectorstore()
    retriever = load_retriever()
    #chat_history = load_chat_history(username)
    #memory = load_memory()
    prompt = load_prompt()
    #policy = load_policy("the good guys")

# Drop the Conversational Memory
#with st.sidebar:
#    with st.form('delete_memory'):
#        st.caption('Delete the history in the conversational memory.')
#        submitted = st.form_submit_button('Delete chat history')
#        if submitted:
#            with st.spinner('Delete chat history'):
#                memory.clear()


# Add a drop down to select Merchant
with st.sidebar:
    st.caption('Choose the Merchant')
    merchant = st.selectbox('Merchant', [
        'The Good Guys',
        'Other',
    ])
    merchant = merchant.lower()
    policy = load_policy(merchant)


# Add a drop down to choose the LLM model
with st.sidebar:
    st.caption('Choose the LLM model')
    model_id = st.selectbox('Model', [
        'openai.gpt-4o',
        'openai.gpt-4',
        'openai.gpt-3.5',
        ])
    model = load_model(model_id)

st.markdown("<style> img {width: 200px;} </style>", unsafe_allow_html=True,
)

# Draw all messages, both user and agent so far (every time the app reruns)
for message in st.session_state.messages:
    st.chat_message(message.type).markdown(message.content)

# Now get a prompt from a user
if question := st.chat_input("How can I help you?"):
    print(f"Got question: \"{question}\"")

    # Add the prompt to messages, stored in session state
    st.session_state.messages.append(HumanMessage(content=question))

    # Draw the prompt on the page
    print("Display user prompt")
    with st.chat_message("user"):
        st.markdown(question)

    # Get the results from Langchain
    print("Get AI response")
    with st.chat_message("assistant"):

        #history = memory.load_memory_variables({})
        #print(f"Using memory: {history}")

        inputs = RunnableMap({
            'context': lambda x: retriever.get_relevant_documents(x['question']),
            #'chat_history': lambda x: x['chat_history'],
            'policy': lambda x: x['policy'],
            'question': lambda x: x['question']
        })
        #print(f"Using inputs: {inputs}")

        chain = inputs | prompt | model
        #print(f"Using chain: {chain}")

        # Call the chain and stream the results into the UI
        #response = chain.stream({'question': question, 'chat_history': history}, config={'callbacks': [StreamHandler(response_placeholder)], "tags": [username]})
        response = chain.stream({'question': question, 'policy': policy}, config={"tags": [username]})
        content = st.write_stream(stream_parser(response))

        # Add the result to memory
        #memory.save_context({'question': question}, {'answer': content})

        # Add the answer to the messages session state
        st.session_state.messages.append(AIMessage(content=content))
