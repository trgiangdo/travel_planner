import operator
from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class TripState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    location: str | None
    interests: str | None
    plan: str | None


class ExtractedInfo(BaseModel):
    """Information extracted from the user's message."""

    location: Optional[str] = Field(
        None, description="The destination city or country, or null if not present"
    )
    interests: Optional[str] = Field(
        None, description="The user's interests for the trip, or null if not present"
    )
