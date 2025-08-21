import datetime
import time
import uuid

from fastapi import APIRouter, Request, Depends, BackgroundTasks, HTTPException

from app.models.models import QueryInput, QueryResponse
from app.llm.main_agent.agent import MainAgent
router = APIRouter()


@router.post("/chat", response_model=QueryResponse)
async def chat(
        query_input: QueryInput,
        ):
    session_id = query_input.session_id or str(uuid.uuid4())
    print(f"Session ID: {session_id}, User Query: {query_input.question}")

    # agent code
    agent = MainAgent()
    response = await agent.get_response(message=query_input.question, session_id=session_id, user_id=session_id)
    answer = response["messages"][-1].content
    print(answer)

    # insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
    print(f"Session ID: {session_id}, AI Response: {answer}")
    return QueryResponse(answer=answer, session_id=session_id)

    # try:
    #     survey_data = payload.survey.data
    #
    #     agent = MealSuggestionAgent()
    #
    #     user_message = f"""
    #     Survey data: {survey_data}
    #     Location: {payload.location}
    #     """
    #     response = await agent.get_response(
    #         user_id=payload.user_id,
    #         session_id=payload.user_id,
    #         message=user_message,
    #     )
    #
    #     # print(response["messages"][-1].content)
    #     structured_response = agent.extract_structured_response(response)
    #     # print(f"Structured response {structured_response.meal_suggestions}")
    #     return MealSuggestionResponse(user_id=payload.user_id, generated_at=datetime.datetime.now(), meal_suggestions=structured_response.meal_suggestions)
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))