# dependencies/libaries: langchain_openai, os, dotenv
import os # call functions from the operating system
from dotenv import load_dotenv # load the env variables (variables stored on your shell - RAM memory) ---> loads the .env file
from langchain_openai import ChatOpenAI # import the ChatOpenAI class from the langchain_openai module - function to iteract with openai models

load_dotenv() # loading the env variables ---> executing the function laod_dotenv

model = ChatOpenAI(model="gpt-4o-mini") # creating an instance of the ChatOpenAI class with the model parameter set to "gpt-4o-mini"

user_input = input("Type your input: ") # expects for the user to type the input

output = model.invoke(user_input) # invoking the model with the user_input from the user

# {"content": "the content output", "usage_metadata": "the usage metadata", ...}
#
#

# output.content output.usage_metadata
print(f"output = {output.content}") # printing the output from the model