import os
from openai import OpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

def get_client():
    # Carregar a chave de API da OpenAI a partir da variável de ambiente
    opanai_api_key = os.getenv('OPENAI_API_KEY')
    # Verificar se a chave de API foi carregada corretamente
    if opanai_api_key is None:
        raise ValueError("A chave de API da OpenAI não foi encontrada. Certifique-se de que a variável de ambiente 'OPENAI_API_KEY' está definida.")
    # Inicialize o cliente da OpenAI
    client = OpenAI(
        api_key=opanai_api_key
    )
    return client

# Função para gerar um score de engajamento a partir de um headline e um summary
def get_engagement_score(client, headline, summary):

    model = ChatOpenAI(temperature=0)

    prompt_template = """You are a Marketing Expert and works for an advertising agency. 
     You will receive a Headline (the ad title for the item being shown to the users) and a Summary (the ad description for the item being shown to the users).
     Given these inputs, you should calculate an engagement score that ranges from 0 to 100.
     You should evaluate if the following criteria is met to calculate your engagement score:

     ### 1. **Clear and Compelling Message**
        - **Clarity**: The message should be clear and easy to understand.
        - **Relevance**: It should be relevant to the target audience's needs and interests.
        - **Value Proposition**: Clearly communicate the benefits and value of the product or service.
     ### 2. **Strong Call to Action (CTA)**
        - **Clear CTA**: Include a clear and concise call to action that tells the audience what to do next (e.g., "Buy Now," "Learn More," "Sign Up").
        - **Urgency**: Create a sense of urgency to encourage immediate action (e.g., "Limited Time Offer," "Act Now").
    
     {format_instructions}

     Headline: {headline}
     Summary: {summary}
    """

    # Classe utilizada dentro da função `get_engagement_score` em `JsonOutputParser`
    class Engagement(BaseModel):
        headline: str = Field(description="ad title for the item being shown to the users")
        summary: str = Field(description="ad description for the item being shown to the users")
        engagement_score: str = Field(description="ad engagement score that ranges from 0 to 100")

    parser = JsonOutputParser(pydantic_object=Engagement)

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["headline", "summary"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | model | parser

    try:
        engagement_json = chain.invoke({"headline": headline, "summary": summary})
        return engagement_json
    except:
        print(f"An error occurred during engagement_score generation. Verify your final results for `None` values.")
        return None

    

# Função que gera estados a partir de um texto em inglês
def get_state(client, input_text):

    model = ChatOpenAI(temperature=0)

    prompt_template = """You are a helpful assistant. I will give you a text, that can be a city name, region name, or anything that resembles a state name,
    and I want you to return me the State name.

    {format_instructions}

    Text: {input_text}."""

    class State(BaseModel):
        input_text: str = Field(description="a random text, that can be a city name, region name, or anything that resembles a state name")
        state: str = Field(description="state name extracted from input_text")

    parser = JsonOutputParser(pydantic_object=State)

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["input_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | model | parser

    try:
        state_json = chain.invoke({"input_text": input_text})
        return state_json
    except:
        print(f"An error occurred during State extraction. Verify your final results for `None` values.")
        state_json = None
        return state_json

    

# Função para gerar uma resposta usando o modelo GPT-4o
def gerar_resposta(client, user_message, max_response_tokens=None, memory=[]):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.extend(memory)
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            max_tokens=max_response_tokens,
            messages=messages
        )
        reply = response.choices[0].message.content
        memory.append({"role": "user", "content": user_message})
        memory.append({"role": "assistant", "content": reply})
        return reply, memory

    except Exception as e:
        return f"An error occurred: {e}"
    
# Função principal para manter a conversa ativa no terminal
def chat_with_openai():
    client = get_client()
    memory = []
    print("Iniciando a conversa com o modelo OpenAI. Digite 'sair' para encerrar.")
    
    while True:
        user_input = input("\nVocê:\n")
        if user_input.lower() == 'sair':
            print("\nConversa encerrada.")
            break
        
        response, memory = gerar_resposta(client, user_input, memory=memory)
        print(f"\nAssistente:\n{response}")

if __name__ == "__main__":
    chat_with_openai()