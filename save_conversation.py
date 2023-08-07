## Integrate code with OpenAI

import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

from langchain.memory import ConversationBufferMemory

import streamlit as st

os.environ['OPENAI_API_KEY'] = openai_key

## OPENAI LLMS
llm = OpenAI(temperature=0.8 )

## Memory
person_memory = ConversationBufferMemory(input_key='name', memory_key='name_hostory')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='dob_hstory')
description_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

# First Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template='Tell me about the celebrity {name}'
)
chain1 = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='person', memory=person_memory)

# Second Prompt Templates
second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template='when was {person} born'
)
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob', memory=dob_memory)

# Third Prompt Templates
third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template='Mention 5 major events that happend around {dob} in the world'
)
chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key='description', memory=description_memory)

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

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Major Events'):
        st.info(description_memory.buffer)