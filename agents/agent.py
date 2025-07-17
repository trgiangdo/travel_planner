import os

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from .models import ExtractedInfo, TripState

llm_model = os.getenv("OPENAI_MODEL")
llm_api_key = os.getenv("OPENAI_API_KEY")
llm_temperature = float(os.getenv("TEMPERATURE", 0))
llm_max_tokens = int(os.getenv("MAX_TOKENS", 512))


def extract_info(state: TripState, config: RunnableConfig):
    if state.get("location") and state.get("interests"):
        return {}

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at extracting information from a user's message.
Your goal is to extract the destination (location) and travel interests from the last user message.
The user might provide this information over multiple messages.
If the user provides a location, update the 'location' field.
If the user provides interests, update the 'interests' field.
Do not guess or infer information not explicitly provided.
Respond with a JSON object containing the extracted 'location' and 'interests', or null if not found in the last message.
Current state is:
Location: {location}
Interests: {interests}
""",
            ),
            ("user", "{last_message}"),
        ]
    )
    llm = ChatOpenAI(
        model=llm_model,
        temperature=llm_temperature,
        api_key=llm_api_key,
        max_tokens=llm_max_tokens,
    ).with_structured_output(ExtractedInfo)
    chain = prompt | llm

    last_message = state["messages"][-1].content
    extracted = chain.invoke(
        {
            "location": state.get("location"),
            "interests": state.get("interests"),
            "last_message": last_message,
        },
        config=config,
    )

    updated_state = {}
    if not state.get("location") and extracted.location:
        updated_state["location"] = extracted.location
    if not state.get("interests") and extracted.interests:
        updated_state["interests"] = extracted.interests

    return updated_state


def ask_for_location(state: TripState):
    return {"messages": [AIMessage(content="What is the destination of your travel?")]}


def ask_for_interests(state: TripState):
    return {"messages": [AIMessage(content="What are your interests for this trip?")]}


def generate_plan(state: TripState, config: RunnableConfig):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful travel agent. Your goal is to create a travel plan for the user based on their destination and interests.",
            ),
            ("user", "My destination is {location} and my interests are {interests}."),
        ]
    )
    llm = ChatOpenAI(
        model=llm_model,
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
        api_key=llm_api_key,
    )
    chain = prompt | llm
    response = chain.invoke(
        {"location": state["location"], "interests": state["interests"]}, config=config
    )
    return {
        "plan": response.content,
        "messages": [AIMessage(content=response.content)],
    }


def router(state: TripState):
    if state.get("location") is None:
        return "ask_for_location"
    if state.get("interests") is None:
        return "ask_for_interests"
    return "generate_plan"


graph = StateGraph(TripState)

graph.add_node("extract_info", extract_info)
graph.add_node("ask_for_location", ask_for_location)
graph.add_node("ask_for_interests", ask_for_interests)
graph.add_node("generate_plan", generate_plan)

graph.set_entry_point("extract_info")

graph.add_conditional_edges(
    "extract_info",
    router,
    {
        "ask_for_location": "ask_for_location",
        "ask_for_interests": "ask_for_interests",
        "generate_plan": "generate_plan",
    },
)
# graph.add_edge("ask_for_location", "extract_info")
# graph.add_edge("ask_for_interests", "extract_info")
graph.add_edge("ask_for_location", END)
graph.add_edge("ask_for_interests", END)
graph.add_edge("generate_plan", END)

agent = graph.compile()

# Save the agent graph as a PNG file for documentation or visualization purposes
with open("agents/graphs/trip_planner_agent.png", "wb") as f:
    f.write(
        agent.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    )

