## Integrate code with OpenAI

import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

import streamlit as st

os.environ['OPENAI_API_KEY'] = openai_key

## OPENAI LLMS
llm = OpenAI(temperature=0.8 )

# Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template='Tell me about the celebrity {name}'
)

chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True)

# streamlit framework
st.title('Celebrity Search Results')
input_text = st.text_input('Search the topic you want')

if input_text:
    st.write(chain.run(input_text))