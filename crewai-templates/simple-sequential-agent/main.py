from crewai import Agent, Task, Crew, LLM, Process
from dotenv import load_dotenv

load_dotenv()

messages = []

llm = LLM(
    model="gpt-4o-mini",
)

llm_supervisor = LLM(
    model="gpt-4o",
)

user_input = input("Por favor, faça sua pergunta: ")

follow_up_agent = Agent(
    role="Agente de Acompanhamento",
    goal="Responder perguntas e fornecer suporte a clientes que já compraram um produto.",
    backstory="Um cliente já comprou um produto e possui uma dúvida sobre ele. Você está aqui para fornecer suporte, possui muita experiência em atendimento ao cliente e é muito paciente.",
    llm=llm,
    allow_delegation=True,
)

conversation_task = Task(
    description="Tenha uma conversa com um cliente que já comprou um produto. Aqui está a pergunta do usuário: "
    + user_input,
    expected_output="Responda a quaisquer perguntas que o cliente possa ter e forneça suporte conforme necessário.",
    agent=follow_up_agent,
)

crew_conversation = Crew(
    tasks=[conversation_task],
    agents=[follow_up_agent],
    verbose=True,
    process=Process.sequential,
    memory=True,
)

output = str(crew_conversation.kickoff())

print("Saída Final:\n", output)
