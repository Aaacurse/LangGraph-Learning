from langgraph.graph import StateGraph,END,START
from langgraph.types import interrupt,Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from typing  import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
import requests
from langchain_core.messages import BaseMessage,HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

@tool
def get_stock_price(symbol:str)->dict:
    """Fetch the latest stock price for a given symbol (eg 'AAPL','TSLA')
    using Alpha Vantage with API key in the URL"""
    url=f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=J6C840U93PGOEYF1"
    r=requests.get(url)
    
    return r.json()  

@tool
def purchase_stock(symbol:str,quantity:int)->str:
    """Purchase a given quantity of stock for a symbol. This is a mock function that simulates a stock purchase.
    
    HUMAN IN THE LOOP: Before confirming the purchase,this tool will interrupt and wait for a human decision ("yes"/anything else)     
    """
    decision=interrupt(f"Approve purchase of {quantity} shares of {symbol}? yes/no")
    
    if isinstance(decision,str) and decision.lower()=='yes':
        return {
            'status':'success',
            'message':f'Purchased {quantity} shares of {symbol}',
            'symbol':symbol,
            'quantity':quantity
        }
    else:
        return {
            'status':'cancelled',
            'message':f'Purchase of {quantity} shares of {symbol} cancelled by user',
            'symbol':symbol,
            'quantity':quantity
        }
        
tools=[get_stock_price,purchase_stock]
llm_with_tools=llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]
    
def chat_node(state:ChatState)->ChatState:
    message=state['messages']
    response=llm_with_tools.invoke(message)
    return {'messages': [response]}

tool_node=ToolNode(tools)

memory=InMemorySaver()

graph=StateGraph(ChatState)
graph.add_node('chat',chat_node)
graph.add_node('tools',tool_node)
graph.add_edge(START,'chat')
graph.add_conditional_edges('chat',tools_condition)
graph.add_edge('tools','chat')

chatbot=graph.compile(memory)

if __name__=="__main__":
    thread_id='5678'
    
    while True:
        user_input=input("You: ")
        if user_input.lower() in ['exit','quit']:
            print("Exiting chatbot. Goodbye!")
            break
        
        initial_message={'messages': [HumanMessage(content=user_input)]}
        
        result=chatbot.invoke(
            initial_message,
            config={'configurable':{'thread_id':thread_id}}
        )
        
        interrupts=result.get('__interrupt__',[])
        if interrupts:
            prompt_to_human=interrupts[0].value
            print(f"HITL: {prompt_to_human}")
            decision=input("Your Decision (yes/no): ").strip().lower()
            
            result=chatbot.invoke(
                Command(resume=decision),
                config={'configurable':{'thread_id':thread_id}}
            )
            
        messages=result['messages']
        last_msg=messages[-1]
        print(f"Bot: {last_msg.content}\n")