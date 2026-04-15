
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv.ipython import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import create_retriever_tool

load_dotenv(override=True)



chunks = [
"Je m'appelle Niema Nassime, je suis étudiant en ingénierie informatique.",
"J'étudie à l'EMSI, dans la spécialité Intelligence Artificielle et Data.",
"Je suis intéressé par le Machine Learning et l'analyse de données.",
"J'aime aussi apprendre de nouvelles technologies en informatique.",
"Après le baccalauréat, j'ai intégré une école d'ingénieurs.",
"Pendant mes études, j'ai réalisé plusieurs projets en développement web.",
"Je travaille aussi sur un projet de détection de fraude.",
"En parallèle de mes études, je développe mes compétences en IA et Data."
]

embeddings_model = OpenAIEmbeddings()


vector_store = Chroma.from_texts(
          texts=chunks,
          collection_name="Agentic_AI",
            embedding=embeddings_model
)

retriever = vector_store.as_retriever()
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="cv_tool",
    description="Get information about me.",

)

@tool
def get_employee_info(name:str):

    """
    Get information about an employee based on their name,salary,seniority.
    """
    print("get_employee_info tool invoked")
    return{"name": name, "salary": 12000, "seniority": "5"}

@tool
def send_email(email: str, subject: str, content: str):
    """
    Send an email to the specified address with the given subject and content.
    """
    print(f"Sending email to {email}, subject : {subject}, content: {content}")
    return f"Sending email to {email}, subject : {subject}, content: {content}"

llm = ChatOpenAI (model="gbt-4o", temperature=0)
agent = create_agent(
    model=llm,
    tools=[get_employee_info, send_email],
    system_prompt="Answer to user query using provided tools",

)

#resp = agent.invoke({
 #   "messages": [HumanMessage(content="Quel est le salaire de Ahmed?")]
#})
#print(resp['messages'][-1].content)