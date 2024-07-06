import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

#import streamlit_nested_layout
from classes import get_primer, format_question, run_request
import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_icon="chat2vis.png", layout="wide", page_title="CSV_VISUALISATION")

available_models = {"CodeLlama-34b-Instruct-hf": "CodeLlama-34b-Instruct-hf"}

# List to hold datasets
if "datasets" not in st.session_state:
    datasets = {}
    # Preload datasets
    datasets["Movies"] = pd.read_csv("movies.csv")
else:
    # use the list already loaded
    datasets = st.session_state["datasets"]


hf_key = os.getenv("HUGGINGFACE_API_KEY")
tab1, tab2 = st.tabs(["Visualize Data", "Chat"])

with tab1:

  with st.sidebar:
    # First we want to choose the dataset, but we will fill it with choices once we've loaded one
    dataset_container = st.empty()

    # Add facility to upload a dataset
    try:
        uploaded_file = st.file_uploader(":computer: Load a CSV file:", type="csv")
        index_no = 0
        if uploaded_file:
            # Read in the data, add it to the list of available datasets. Give it a nice name.
            file_name = uploaded_file.name[:-4].capitalize()
            datasets[file_name] = pd.read_csv(uploaded_file)
            # We want to default the radio button to the newly added dataset
            index_no = len(datasets) - 1
    except Exception as e:
        st.error("File failed to load. Please select a valid CSV file.")
        print("File failed to load.\n" + str(e))
    # Radio buttons for dataset choice
    chosen_dataset = dataset_container.radio(":bar_chart: Choose your data:", datasets.keys(), index=index_no)  # ,horizontal=True,)

    # Check boxes for model choice
    st.write(":brain: Choose your model(s):")
    # Keep a dictionary of whether models are selected or not
    use_model = {}
    for model_desc, model_name in available_models.items():
        label = f"{model_desc} ({model_name})"
        key = f"key_{model_desc}"
        use_model[model_desc] = st.checkbox(label, value=True, key=key)

# Text area for query
question = st.text_area(":eyes: What would you like to visualise?", height=10)
go_btn = st.button("Go...")

# Make a list of the models which have been selected
selected_models = [model_name for model_name, choose_model in use_model.items() if choose_model]
model_count = len(selected_models)

# Execute chatbot query
if go_btn and model_count > 0:
    api_keys_entered = True
    # Check API keys are entered.
    if "Code Llama" in selected_models:
        if not hf_key.startswith('hf_'):
            st.error("Please enter a valid HuggingFace API key.")
            api_keys_entered = False
    if api_keys_entered:
        # Place for plots depending on how many models
        plots = st.columns(model_count)
        # Get the primer for this dataset
        primer1, primer2 = get_primer(datasets[chosen_dataset], 'datasets["' + chosen_dataset + '"]')
        # Create model, run the request and print the results
        for plot_num, model_type in enumerate(selected_models):
            with plots[plot_num]:
                st.subheader(model_type)
                try:
                    # Format the question
                    question_to_ask = format_question(primer1, primer2, question, model_type)
                    # Run the question
                    answer = ""
                    answer = run_request(question_to_ask, available_models[model_type], key=None, alt_key=hf_key)
                    # the answer is the completed Python script so add to the beginning of the script to it.
                    answer = primer2 + answer
                    print("Model: " + model_type)
                    print(answer)
                    plot_area = st.empty()
                    plot_area.pyplot(exec(answer))
                except Exception as e:
                    if "Code Llama" in selected_models:
                        if not hf_key.startswith('hf_'):
                            st.error("Please enter a valid HuggingFace API key.")
                            api_keys_entered = False
                    else:
                        st.error("Unfortunately the code generated from the model contained errors and was unable to execute.")

# Display the datasets in a list of tabs
# Create the tabs
tab_list = st.tabs(datasets.keys())

# Load up each tab with a dataset
for dataset_num, tab in enumerate(tab_list):
    with tab:
        # Can't get the name of the tab! Can't index key list. So convert to list and index
        dataset_name = list(datasets.keys())[dataset_num]
        st.subheader(dataset_name)
        st.dataframe(datasets[dataset_name], hide_index=True)

# Insert footer to reference dataset origin
footer = """<style>.footer {position: fixed;left: 0;bottom: 0;width: 100%;text-align: center;}</style><div class="footer">
<p> <a style='display: block; text-align: center;'> Datasets courtesy of NL4DV, nvBench and ADVISor </a></p></div>"""
st.caption("Datasets courtesy of NL4DV, nvBench and ADVISor")

# Hide menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with tab2:
    import streamlit as st
from streamlit_chat import message
import tempfile   # temporary file
from langchain.document_loaders.csv_loader import CSVLoader  # using CSV loaders
from langchain.embeddings import HuggingFaceEmbeddings # import hf embedding
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss' # # Set the path of our generated embeddings


# Loading the model of your choice
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=2000,
        temperature=0.5
    )
    # the model defined, can be replaced with any ... vicuna,alpaca etc
    # name of model
    # tokens
    # the creativity parameter
    return llm


st.title("Generative bi Chat")
#st.markdown("<h3 style='text-align: center; color: white;'>Built by <a href=''></a></h3>",unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload File", type="csv") # uploaded file is stored here
# file uploader
if uploaded_file:
    # tempfile needed as CSVLoader accepts file_path exclusively
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name # save file locally

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','}) # any loader can be put here based on the data being used
    # csv_args={'delimiter': ','} for faulty formatted csv
    data = loader.load() # load the data
    #st.json(data)   # uncomment to check uploaded data
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device': 'cpu'}) # use sentence transformer to create embeddings

    # FAISS Can be replaced by Chroma... so it will be like CHROMA.fromdocuments...
    db = FAISS.from_documents(data, embeddings) # pass data embeddings vector data here
    db.save_local(DB_FAISS_PATH) # save vector embedding here on mentioned path
    llm = load_llm() # Load the Language model here

    # the conversational chain which preserves context learning in chat
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
    # ConversationalRetrievalChain can be replaced by LLMChain,retrivalQA

    # func for streamlit chat takes query from User
    def conversational_chat(query):
        # key value pairs of conversational history
        result = chain({"question": query, "chat_history": st.session_state['history']}) # enduser query and result variable

        # add all responses here with query to preserve context
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"] # get the generated result


    # appending history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Start message, in context of no question having being not asked yet
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    # container for the chat history
    response_container = st.container() # form

    # container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to csv data 👉 (:", key='input') # user input values are here
            submit_button = st.form_submit_button(label='Send') # button to retrieve answer

        if submit_button and user_input:
            output = conversational_chat(user_input)

            st.session_state['past'].append(user_input) # old user input is appended
            st.session_state['generated'].append(output) # append the generated

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")








