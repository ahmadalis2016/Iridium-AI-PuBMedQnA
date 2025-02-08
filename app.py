from typing import List
import streamlit as st
from haystack import component, Document
from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.components.builders.prompt_builder import PromptBuilder
from dotenv import load_dotenv
from pymed import PubMed
from PIL import Image
import os

load_dotenv()

# Streamlit app initialization.
st.set_page_config(page_title="Iridium AI")

# Load and display Iridium logo.
logo_path = "Images/IridiumAILogo.png"
iridium_logo = Image.open(logo_path)
st.image(iridium_logo, use_column_width=False)

st.header("Q&A Chatbot Powered by PubMed & Mistral AI")

os.environ['HUGGINGFACE_API_KEY'] = os.getenv('HUGGINGFACE_API_KEY')

pubmed = PubMed(tool="Haystack2.0Prototype", email="emailabc@gmail.com")

def documentize(article):
    return Document(content=article.abstract, meta={'title': article.title, 'keywords': article.keywords})

@component
class PubMedFetcher():
    @component.output_types(articles=List[Document])
    def run(self, queries: list[str]):
        cleaned_queries = [query.strip() for query in queries[0].split('\n')]
        articles = []
        try:
            for query in cleaned_queries:
                response = pubmed.query(query, max_results=1)
                documents = [documentize(article) for article in response]
                articles.extend(documents)
        except Exception as e:
            print(e)
            print(f"Couldn't fetch articles for queries: {queries}")
        return {'articles': articles}

keyword_llm = HuggingFaceTGIGenerator("mistralai/Mistral-7B-Instruct-v0.3")
keyword_llm.warm_up()

llm = HuggingFaceTGIGenerator("mistralai/Mistral-7B-Instruct-v0.3")
llm.warm_up()

keyword_prompt_template = """
Your task is to convert the following question into 3 keywords that can be used to find relevant medical research papers on PubMed.
Here is an examples:
question: "What are the latest treatments for major depressive disorder?"
keywords:
Antidepressive Agents
Depressive Disorder, Major
Treatment-Resistant depression
---
question: {{ question }}
keywords:
"""

prompt_template = """
Answer the question truthfully based on the given documents.
If the documents don't contain an answer, use your existing knowledge base.

q: {{ question }}
Articles:
{% for article in articles %}
  {{article.content}}
  keywords: {{article.meta['keywords']}}
  title: {{article.meta['title']}}
{% endfor %}
"""

keyword_prompt_builder = PromptBuilder(template=keyword_prompt_template)
prompt_builder = PromptBuilder(template=prompt_template)
fetcher = PubMedFetcher()

pipe = Pipeline()
pipe.add_component("keyword_prompt_builder", keyword_prompt_builder)
pipe.add_component("keyword_llm", keyword_llm)
pipe.add_component("pubmed_fetcher", fetcher)
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", llm)

pipe.connect("keyword_prompt_builder.prompt", "keyword_llm.prompt")
pipe.connect("keyword_llm.replies", "pubmed_fetcher.queries")
pipe.connect("pubmed_fetcher.articles", "prompt_builder.articles")
pipe.connect("prompt_builder.prompt", "llm.prompt")

def ask(question):
    output = pipe.run(data={"keyword_prompt_builder": {"question": question},
                            "prompt_builder": {"question": question},
                            "llm": {"generation_kwargs": {"max_new_tokens": 500}}})
    return output['llm']['replies'][0]

question = st.text_input("Enter your question")
if st.button("Ask"):
    answer = ask(question)
    st.markdown(answer)

st.markdown("## Examples")
st.markdown("- What are the latest advancements in Alzheimer's disease research?")
st.markdown("- What is the current understanding of multiple sclerosis?")
st.markdown("- Tell me about the side effects of chemotherapy.")
