import os
from constants import hugging_face_access_key
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain


# initializing the reqiured constants used in the file
print(f"initializing the constants")
HUGGINGFACEHUB_API_TOKEN = hugging_face_access_key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
print(f"initialed the constants")


# Selecting a model
print(f"selecting a model")
# repo_id = "google/flan-t5-xl"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
repo_id = "facebook/bart-base"

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0, "max_length": 64})
print(f"selected model with id: {repo_id}")


if __name__ == "__main__":
    template = """Question: {question}

    Answer: Let's think step by step."""

    print(f"template initialized")
    prompt = PromptTemplate(template=template, input_variables=["question"])
    print(f"prompt initialized with template : {template}")

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    print(f"template initialized")

    question = "Who won the FIFA World Cup in the year 1994? "

    print(f"running the llm_chain")
    print(llm_chain.run(question))
