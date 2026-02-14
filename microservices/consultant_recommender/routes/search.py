"""Routes for searching consultants."""

import json

from db.database import get_search_database
from fastapi import APIRouter, Depends, HTTPException
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import numpy as np
from utils.embeddings import get_embedding

from microservices.shared.authentication.main import verify_token

router = APIRouter()

# Define the consultant JSON format template as a constant
CONSULTANT_JSON_TEMPLATE = """{{
    'name': match.metadata.get('name'),
    'expertise': match.metadata.get('expertise'),
    'experience': match.metadata.get('experience'),
    'website': match.metadata.get('website'),
    'phone': match.metadata.get('phone'),
    'gmail': match.metadata.get('gmail'),
    'apps_included': match.metadata.get('apps_included'),
    'language': match.metadata.get('language'),
    'country': match.metadata.get('country'),
    'company_name': match.metadata.get('company_name'),
    'type_of_services': match.metadata.get('type_of_services'),
    'countries_with_office_locations': match.metadata.get(
        'countries_with_office_locations'
    ),
    'about': match.metadata.get('about'),
    'date': match.metadata.get('date'),
    'time': match.metadata.get('time')
}}"""


@router.get("/search_consultant")
def search_consultants(
    query: str, user_work_description: str, current_user: dict = Depends(verify_token)
):
    """Search for consultants based on a query and user work description."""
    query_embedding = get_embedding(query)
    query_embedding = np.array(query_embedding, dtype=np.float32).tolist()

    db = get_search_database()

    # Query Pinecone using keyword arguments
    response = db.query(
        vector=query_embedding,
        top_k=10,
        namespace="consultants",
        include_values=True,
        include_metadata=True,
    )

    if not response.matches:
        return {"message": "No consultants found."}

    for match in response.matches:
        print(match.metadata)

    # Break down the template into multiple lines for readability
    prompt_text = (
        "You are the best consultant recommender. "
        "If Need translate in english Experience in english. "
        "The best choice depends on the specific needs of "
        "the business seeking consulting services. "
        "Given the following consultants with their "
        "work descriptions, select only the best "
        "ones based on expertise and experience. "
        "Return only the selected consultants as "
        "a JSON array, with each consultant as a JSON object "
        f"in the format:\n\n{CONSULTANT_JSON_TEMPLATE}\n\n"
        "Do not include any additional text or description. "
        "Consultants: {consultants}. "
        "User's work description: {user_work_description}. "
        "User's tools: {query}"
    )

    prompt_template = ChatPromptTemplate.from_template(prompt_text)
    consultants = [match.metadata for match in response.matches]

    print(prompt_template)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=4000)
    chain = prompt_template | llm

    best_consultants = chain.invoke(
        {
            "consultants": consultants,
            "user_work_description": user_work_description,
            "query": query,
        }
    )

    # Parse the AI's response to extract the JSON list
    try:
        # Use replace() instead of strip() for removing code block markers
        response_content = (
            best_consultants.content.replace("```json", "").replace("```", "").strip()
        )
        best_consultants_list = json.loads(response_content)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print(f"Response content: {best_consultants.content}")
        return HTTPException(
            status_code=404, detail="Failed to parse consultant recommendations."
        )

    return {
        "user": current_user,
        "best_consultants": best_consultants_list,
    }
