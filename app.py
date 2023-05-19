import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# SETUP
os.environ["OPENAI_API_KEY"] = apikey
st.title("ProjectGPT")

# OPTIONS
cols = st.columns(3)
project_type = cols[0].selectbox(
    "Project Type:",
    [
        "Data Engineering",
        "Software Engineering",
        "Machine Learning",
        "Game Development",
        "Cybersecurity",
        "Robotics",
    ],
)
project_language = cols[1].selectbox(
    "Programming Language:",
    [
        "Python",
        "JavaScript",
        "C/C++",
        "Dart",
        "Go",
    ],
)
project_topic = cols[2].text_input("Enter a topic:")

# TEMPLATES
idea_template = PromptTemplate(
    input_variables=["project_type", "project_language", "project_topic"],
    template="Come up with an original, interesting, and useful {project_type} project idea about {project_topic} using {project_language}.",
)
tools_template = PromptTemplate(
    input_variables=["idea"],
    template="Based the following idea: {idea}\n\nWhat technologies, libraries, or APIs would be the best to use for this project?",
)
steps_template = PromptTemplate(
    input_variables=["idea", "tools"],
    template="Based the following idea: {idea}\n\nAnd the following tools: {tools}. \n\nWhat are the steps to implement this project?",
)

# MEMORY
topic_memory = ConversationBufferMemory(
    input_key="project_topic", memory_key="chat_history"
)
idea_memory = ConversationBufferMemory(input_key="idea", memory_key="chat_history")
tools_memory = ConversationBufferMemory(input_key="tools", memory_key="chat_history")

# CHAINS
llm = OpenAI(temperature=0.9)
idea_chain = LLMChain(
    llm=llm, prompt=idea_template, output_key="idea", memory=topic_memory, verbose=True
)
tools_chain = LLMChain(
    llm=llm, prompt=tools_template, output_key="tools", memory=idea_memory, verbose=True
)
steps_chain = LLMChain(
    llm=llm,
    prompt=steps_template,
    output_key="steps",
    memory=tools_memory,
    verbose=True,
)

# GENERATE
if project_topic and st.button("Generate"):
    idea = idea_chain.run(
        {
            "project_type": project_type,
            "project_language": project_language,
            "project_topic": project_topic,
        }
    )
    tools = tools_chain.run(idea=idea)
    steps = steps_chain.run(idea=idea, tools=tools)

    st.write(f"\nIDEA \n{idea}")
    st.write(f"TOOLS \n{tools}")
    st.write(f"STEPS \n{steps}")
