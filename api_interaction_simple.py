import os
from constants import hugging_face_access_key
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

import streamlit as st

# initializing the reqiured constants used in the file
print(f'initializing the constants')
HUGGINGFACEHUB_API_TOKEN = hugging_face_access_key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


print(f'initialed the constants')



# Selecting a model
print(f'selecting a model')
# repo_id = "google/flan-t5-xl"
repo_id = "stabilityai/stablelm-tuned-alpha-3b"
print(f'selecting model: {repo_id}')

print(f'initializing HuggingFace with basic parameters')
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0, "max_length": 64})

print(f'temaplate initialized')
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)


question = "Who won the FIFA World Cup in the year 1994? "

print(llm("tell me a joke !"))
# print(llm_chain.run(question))




    