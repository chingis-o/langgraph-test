from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, List, TypedDict


class State(TypedDict):
    messages: Annotated[List, add_messages]


graph_builder = StateGraph(State)


llm = ChatOllama(model="deepseek-r1:7b", temperature=0)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    print("\nAssistant:", end=" ", flush=True)

    for chunk in llm.stream(user_input):
        print(chunk.content, end="", flush=True)

    print()


def run_chat():
    print("Start chatting with the bot (type 'quit' to exit):")
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except EOFError:
            print("\nInput not available. Exiting.")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Trying sample input...")
            stream_graph_updates("What do you know about LangGraph?")
            break


run_chat()
