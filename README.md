# Akira - Your personal Data Analyst

## CSV Visualization and Chatbot
This project provides a Streamlit-based application that allows users to visualize data from CSV files and interact with a chatbot to ask questions about the data.In short your personal Data Analyst.

## Features
**-Data Visualization:** -Users can upload their own CSV files or select from pre-loaded datasets.
The application generates appropriate data visualizations based on the chosen dataset.
Users can choose from multiple models (e.g., Code Llama) to generate the visualizations.

**-Chatbot:** -Users can ask questions about the data and receive answers from the chatbot.
The chatbot uses a ConversationalRetrievalChain from LangChain to provide context-aware responses.
The chatbot is powered by a locally loaded LLM (Language Model), such as LLaMA or Vicuna.
Installation.

## Getting Started:

**Clone the repository:**
```sh
git clone https://github.com/your-username/csv-visualization-chatbot.git
```
**Change to the project directory:**
```sh
cd csv-visualization-chatbot
```
**Install the required dependencies:**
```sh
pip install -r requirements.txt
```


Ensure you have the necessary local model files (e.g., "codellama-34b-instruct.Q2_K.gguf") in the appropriate directory.

**Start the Streamlit application:**
```sh
streamlit run app.py
```
