from typing import List
import streamlit as st
from haystack import component, Document, Pipeline
from haystack.components.generators import HuggingFaceTGIGenerator
from haystack.components.prompt import PromptBuilder
from dotenv import load_dotenv
from pymed import PubMed
from PIL import Image
import os

load_dotenv()

# Streamlit setup
st.set_page_config(page_title="Iridium AI")
logo = Image.open("Images/IridiumAILogo.png")
st.image(logo, use_column_width=False)
st.header("Q&A Chatbot Powered by PubMed & Mistral AI")

# Initialize PubMed client
pubmed = PubMed(tool="Haystack2.0Prototype", email="emailabc@gmail.com")

def documentize(article):
    return Document(content=article.abstract, meta={'title': article.title, 'keywords': article.keywords})

@component
class PubMedFetcher:
    @component.output_types(articles=List[Document])
    def run(self, queries: List[str]):
        articles = []
        for query in queries:
            try:
                response = pubmed.query(query, max_results=1)
                articles.extend([documentize(art) for art in response])
            except Exception as e:
                st.error(f"Error fetching {query}: {e}")
        return {'articles': articles}

# Initialize components
keyword_llm = HuggingFaceTGIChatGenerator(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token=os.getenv('HUGGINGFACE_API_KEY')
)
llm = HuggingFaceTGIChatGenerator(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token=os.getenv('HUGGINGFACE_API_KEY')
)

keyword_prompt_template = """
Your task is to convert the following question into 3 keywords for PubMed searches.
Example:
Question: "Latest treatments for major depressive disorder?"
Keywords:
Antidepressive Agents
Depressive Disorder, Major
Treatment-Resistant depression

Question: {{ question }}
Keywords:
"""

prompt_template = """
Answer based on the documents. If unsure, say so.

Question: {{ question }}
Documents:
{% for doc in articles %}
  Title: {{ doc.meta['title'] }}
  Keywords: {{ doc.meta['keywords'] }}
  Content: {{ doc.content }}
{% endfor %}
Answer:
"""

# Build pipeline
pipe = Pipeline()
pipe.add_component("keyword_builder", PromptBuilder(keyword_prompt_template))
pipe.add_component("keyword_llm", keyword_llm)
pipe.add_component("fetcher", PubMedFetcher())
pipe.add_component("prompt_builder", PromptBuilder(prompt_template))
pipe.add_component("llm", llm)

pipe.connect("keyword_builder.prompt", "keyword_llm.prompt")
pipe.connect("keyword_llm.replies", "fetcher.queries")
pipe.connect("fetcher.articles", "prompt_builder.articles")
pipe.connect("prompt_builder.prompt", "llm.prompt")

def ask(question):
    output = pipe.run(
        data={
            "keyword_builder": {"question": question},
            "prompt_builder": {"question": question},
            "llm": {"generation_kwargs": {"max_new_tokens": 500}}
        }
    )
    return output["llm"]["replies"][0]

# Streamlit UI
question = st.text_input("Enter your question")
if st.button("Ask"):
    answer = ask(question)
    st.markdown(f"**Answer:** {answer}")

st.markdown("## Examples")
st.markdown("- Latest advancements in Alzheimer's disease research?")
st.markdown("- Current understanding of multiple sclerosis?")
