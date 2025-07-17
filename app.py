from typing import Dict, List, Tuple

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage

from agents import TripState, agent


def create_initial_state():
    return TripState(messages=[], location=None, interests=None, plan=None)

current_state = create_initial_state()


def predict(message: str, history: List[Dict[str, str]]) -> Tuple[str, TripState]:
    global current_state

    current_state["messages"].append(HumanMessage(content=message))
    result_state = agent.invoke(current_state)

    bot_message = ""
    # Find the last AI message to display
    for msg in reversed(result_state["messages"]):
        if isinstance(msg, AIMessage):
            bot_message = msg.content
            break

    # If a plan was generated, the conversation is over. Reset state.
    if result_state.get("plan"):
        current_state = create_initial_state()
    else:
        # Update the state for the next turn
        current_state = result_state

    return bot_message


with gr.Blocks() as demo:
    chat_interface = gr.ChatInterface(
        fn=predict,
        title="Trip Planner Agent",
        description="Ask me to plan a trip. I'll ask you about your destination and interests.",
        examples=[
            ["Hi, I want to plan a trip."],
            ["I want to go to Paris."],
            ["I'm interested in museums and food."],
            ["Plan a 3-day trip to Tokyo for a foodie."]
        ],
        type="messages",
    )

if __name__ == "__main__":
    demo.launch()
