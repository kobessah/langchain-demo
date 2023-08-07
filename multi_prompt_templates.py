## Integrate code with OpenAI

import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

import streamlit as st

os.environ['OPENAI_API_KEY'] = openai_key

## OPENAI LLMS
llm = OpenAI(temperature=0.8 )

# First Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template='Tell me about the celebrity {name}'
)
chain1 = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='person')

# Second Prompt Templates
second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template='when was {person} born'
)
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob')

# Third Prompt Templates
third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template='Mention 5 major events that happend around {dob} in the world'
)
chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key='description')

chain = SequentialChain(
    chains=[chain1, chain2, chain3], 
    input_variables=['name'], 
    output_variables=['person', 'dob', 'description'],
    verbose=True
)

# streamlit framework
st.title('Celebrity Search Results')
input_text = st.text_input('Search the topic you want')

if input_text:
    st.write(chain({'name': input_text}))