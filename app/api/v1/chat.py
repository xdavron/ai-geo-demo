import datetime
import json
import time
import uuid

from fastapi import APIRouter, Request, Depends, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from app.models.models import QueryInput, QueryResponse, StreamQueryResponse
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


@router.post("/chat/stream")
async def chat_stream(
    query_input: QueryInput,
):
    """Process a chat request using LangGraph with streaming response.

    Args:
        query_input: The FastAPI request object for rate limiting.

    Returns:
        StreamingResponse: A streaming response of the chat completion.

    Raises:
        HTTPException: If there's an error processing the request.
    """
    agent = MainAgent()
    session_id = query_input.session_id or str(uuid.uuid4())
    print(f"Session ID: {session_id}, User Query: {query_input.question}")
    try:

        async def event_generator():
            """Generate streaming events.

            Yields:
                str: Server-sent events in JSON format.

            Raises:
                Exception: If there's an error during streaming.
            """
            try:
                full_response = ""
                async for chunk in agent.get_stream_response(
                    query_input.question, session_id, user_id=session_id
                 ):
                    print(chunk)
                    full_response += chunk
                    response = StreamQueryResponse(content=chunk, done=False)
                    yield f"data: {json.dumps(response.model_dump())}\n\n"

                # Send final message indicating completion
                final_response = StreamQueryResponse(content="", done=True)
                yield f"data: {json.dumps(final_response.model_dump())}\n\n"

            except Exception as e:
                error_response = StreamQueryResponse(content=str(e), done=True)
                yield f"data: {json.dumps(error_response.model_dump())}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))