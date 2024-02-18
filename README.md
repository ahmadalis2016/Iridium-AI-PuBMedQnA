
![image](https://github.com/ahmadalis2016/Iridium-AI-Image-Analysis/assets/130319416/3590b637-b72a-4a41-86ba-a556e9c22016) 

## Iridium AI: Q&A 💬 Chatbot Powered by PubMed & Mistral AI

# Overview
This project implements a Question-Answering (Q&A) chatbot powered by PubMed and Mistral AI, allowing users to ask questions about biomedical topics and receive answers from a friendly AI assistant. The chatbot utilizes the Haystack library for building pipelines and components, and Gradio for creating a user-friendly interface.

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://llama2.streamlitapp.com/)

## Dependencies
# Streamlit: 
A library for creating user interfaces for machine learning models.
# haystack: 
A framework for building end-to-end search pipelines for question-answering systems.
# pymed: 
A library for accessing the PubMed database and retrieving articles.
# dotenv:
A library for loading environment variables from a .env file.

## Prerequisite libraries

```
haystack-ai
pymed
streamlit
python-dotenv
transformers
```

## Components
# PubMedFetcher: 
A component for fetching articles from PubMed based on user queries. It queries PubMed for relevant articles based on user input, retrieves abstracts, titles, and keywords, and converts them into Haystack Documents.

# HuggingFaceTGIGenerator: 
A component for generating responses using the Hugging Face model. It employs Mistral AI's Mistral-7B-Instruct-v0.2 model to generate responses to user queries. This component generates responses based on the keywords extracted from PubMed articles and the user's question.

# PromptBuilder:
A component for building prompts for the HuggingFaceTGIGenerator. It constructs prompts for the model based on the user's question and the extracted keywords from PubMed articles.

# Pipeline:
 A pipeline that connects the components together. It orchestrates the flow of data between the PubMedFetcher, PromptBuilder, HuggingFaceTGIGenerator, and the user interface.


## Usage
# Ask Questions:
Users can input questions about biomedical topics into the provided text box.
# Get Answers:
Upon submitting a question, the chatbot retrieves relevant articles from PubMed, generates responses using Mistral AI's model, and displays the answers in markdown format.
# Examples: 
Users can refer to the provided examples for asking questions related to Alzheimer's disease research, multiple sclerosis, chemotherapy side effects, and more.

## Environment Variables
HUGGINGFACE_API_KEY: API key required for accessing the Hugging Face model.

## Additional Notes
Ensure that the HUGGINGFACE_API_KEY environment variable is properly set before running the script.
Make sure to have an active internet connection to fetch articles from PubMed and generate responses from the Hugging Face model.
Feel free to explore and interact with the chatbot to obtain answers to your biomedical queries!




### Contributing
Ahmad A Shaik, Ph. D.
 *Inspired by Data professor and a 16 z-infra
### License
This project is licensed under the MIT License. See the LICENSE file for details.

### PubMed: 
PubMed is a comprehensive online database containing millions of biomedical literature citations and abstracts. Developed and maintained by the National Center for Biotechnology Information (NCBI), PubMed serves as a vital resource for researchers, healthcare professionals, and students in the biomedical field.
