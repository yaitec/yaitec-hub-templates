from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Modelo OpenAI
llm = ChatOpenAI(
    model="gpt-5-mini",   # pode ser gpt-4o, gpt-4.1, gpt-3.5-turbo, etc.
    temperature=0.1,
)

# Pydantic
class pessoa(BaseModel):
    """"Retorna Informações de uma Pessoa de forma Estruturada"""

    nome: str = Field(description="O nome da pessoa")
    idade: str = Field(description="A idade da pessoa")
    sexo: Optional[str] = Field(
        default=None, description="O sexo da pessoa, caso seja possivel determinar"
    )


# notação legado: str | llm | parser 
structured_llm = llm.with_structured_output(pessoa)

saida = structured_llm.invoke("O Mateus é um homem de 28 anos, que estudou o curso de Direito durante 6 anos, foi muito dificil pra ele...")

print(f"O nome do usuário: {saida.nome}") 
