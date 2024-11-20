from typing import Dict

from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector


class RecommendationInput(BaseModel):
    required_calories: float = Field(
        ..., description="Required calories for the meal in one day"
    )
    preferred_menu: str = Field(..., description="Preferred menu for the meal")


@tool("menu-recommendation", args_schema=RecommendationInput, return_direct=True)
def menu_recommendation(required_calories: float, preferred_menu: str) -> list[Dict]:
    """Give Recommendation Process for a meal based on required calories"""

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    connection = "postgresql+psycopg://postgres:Ep99DiEJ05RxqEjtgHPVG9O1JSIGDSvXgEMgv4x3OWgIl9XuMN9vHJo7FeIJD623@145.223.117.210:6543/postgres"  # Uses psycopg3!
    collection_name = "menu"

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )

    results = vector_store.similarity_search(
        query=preferred_menu,
        k=3,
        filter={
            "calories": {"$lt": required_calories / 3},
        },
    )

    metadata_results = [result.metadata for result in results]
    return metadata_results
