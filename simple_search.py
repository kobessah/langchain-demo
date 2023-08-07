## Integrate code with OpenAI

import os
from constants import openai_key
from langchain.llms import OpenAI

import streamlit as st

os.environ['OPENAI_API_KEY'] = openai_key

# streamlit frameowrk
st.title('Langchain Demo with OPENAI API')
input_text = st.text_input('Search the topic you want')

## OPENAI LLMS
llm = OpenAI(temperature=0.8 )

if input_text:
    st.write(llm(input_text))