from crewai import Agent, Task, Crew, LLM, Process
from dotenv import load_dotenv
from datetime import datetime
from crewai_tools import (
    SerperDevTool,
    WebsiteSearchTool
)


load_dotenv()

messages = []

data_de_hoje = datetime.now().strftime("%d/%m/%Y")

llm = LLM(
    model="gpt-4.1",
)

ferramentas = [SerperDevTool(), WebsiteSearchTool()]

user_input = input("\nSobre o que você deseja saber: ")

pesquisador = Agent(
    role="Pesquisador", # Função do Agente
    goal="Pesquisar notícias e informações relevantes na internet, e trazer essas informações sempre atualizadas.", # Objetivo do Agente
    backstory="Você é um pesquisador com 10 anos de experiência, você é especializado em trazer as notícias mais atuais sobre determinados temas na internet.", # Experiência de Trabalho do Agente
    llm=llm,
    allow_delegation=True,
    tools=ferramentas,
)

conversation_task = Task(
    description="Pesquise sobre o tema: " + user_input + f", e traga as informações mais atualizadas possíveis, considerando que a data de hoje é: {data_de_hoje}. Para encontrar essas informações, você deve consultar a web para pesquisar notícias relevantes!",
    expected_output="",
    agent=pesquisador,
)

crew_conversation = Crew(
    tasks=[conversation_task],
    agents=[pesquisador],
    verbose=True,
    process=Process.sequential,
    memory=True,
)

output = str(crew_conversation.kickoff())

print("Saída Final:\n", output)