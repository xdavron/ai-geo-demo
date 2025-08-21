import asyncio
import os
import traceback
import uuid
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Literal,
    Optional,
)

from langchain_core.messages import HumanMessage, SystemMessage
from langfuse._client.client import Langfuse
from psycopg_pool import AsyncConnectionPool

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import StateSnapshot

from langfuse.langchain import CallbackHandler
from langfuse import observe, get_client

from app.llm.main_agent.state import State
from app.llm.main_agent.prompt import prompt
from app.llm.main_agent.schemas import MainAgentResponse

from app.llm.main_agent.pdf_rag import get_information_from_documents

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

openai = ChatOpenAI(base_url=os.getenv("OPENAI_API_URL"), temperature=float(os.getenv("DEFAULT_LLM_TEMPERATURE")))
class MainAgent:
    def __init__(self):
        self.model = openai #init_chat_model(model_provider="openai", temperature=os.getenv("DEFAULT_LLM_TEMPERATURE"), base_url=os.getenv("OPENAI_API_URL"))
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._agent: Optional[CompiledStateGraph] = None
        print(f"MainAgent initialized , model={self.model.model_dump()}")

    async def _create_agent(self):
        if self._agent is None:
            try:
                # Get connection pool (may be None if DB unavailable)
                connection_pool = await self._get_connection_pool()
                if connection_pool:
                    checkpointer = AsyncPostgresSaver(connection_pool)
                    await checkpointer.setup()
                else:
                    checkpointer = InMemorySaver()

                # Agent creation
                self._agent = create_react_agent(
                    model=self.model,
                    tools=[get_information_from_documents],
                    prompt=prompt,
                    name="MainAgent",
                    checkpointer=checkpointer,
                )
            except Exception as e:
                print(f"Agent creation failed, error: {str(e)}, traceback: {traceback.format_exc()}")
        return self._agent

    async def _get_connection_pool(self) -> AsyncConnectionPool:
        """Get a PostgreSQL connection pool using environment-specific settings.

        Returns:
            AsyncConnectionPool: A connection pool for PostgreSQL database.
        """
        # if self._connection_pool is None:
        #     try:
        #         # Configure pool size based on environment
        #         max_size = settings.POSTGRES_POOL_SIZE
        #
        #         self._connection_pool = AsyncConnectionPool(
        #             settings.POSTGRES_URL,
        #             open=False,
        #             max_size=max_size,
        #             kwargs={
        #                 "autocommit": True,
        #                 "connect_timeout": 5,
        #                 "prepare_threshold": None,
        #             },
        #         )
        #         await self._connection_pool.open()
        #         logger.info("connection_pool_created", max_size=max_size, environment=settings.ENVIRONMENT.value)
        #     except Exception as e:
        #         logger.error("connection_pool_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
        #         # In production, we might want to degrade gracefully
        #         if settings.ENVIRONMENT == Environment.PRODUCTION:
        #             logger.warning("continuing_without_connection_pool", environment=settings.ENVIRONMENT.value)
        #             return None
        #         raise e
        # return self._connection_pool
        return None

    @observe()
    async def get_response(
            self,
            message: str,
            session_id: str,
            user_id: Optional[str] = None,
            system_message: Optional[str] = '',
    ) -> dict:
        """Get a response from the LLM.

        Args:
            message (str): The message to send to the LLM.
            session_id (str): The session ID for Langfuse tracking.
            user_id (Optional[str]): The user ID for Langfuse tracking.
            system_message (Optional[str]): The system message for agent.

        Returns:
            str: The content of response from the LLM.
        """
        if self._agent is None:
            self._agent = await self._create_agent()

        langfuse = get_client()
        langfuse.update_current_trace(
            name="LessonConversationAgent",
            session_id=session_id,
            user_id=user_id,
            metadata={"user_id": user_id, "session_id": session_id, "debug": False}
        )
        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [CallbackHandler()],
        }
        try:
            response = await self._agent.ainvoke(
                {
                    "messages": [SystemMessage(content=system_message),HumanMessage(content=message)]
                },
                config
            )
            return response
            # return response["messages"][-1].content

        except Exception as e:
            print(f"Error getting response: {str(e)}, {traceback.format_exc()}")

    @observe()
    async def get_stream_response(
            self, message: str, session_id: str, user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Get a stream response from the LLM.

        Args:
            message (str): The message to send to the LLM.
            session_id (str): The session ID for the conversation.
            user_id (Optional[str]): The user ID for the conversation.

        Yields:
            str: Tokens of the LLM response.
        """
        langfuse = get_client()
        langfuse.update_current_trace(
            name="LessonConversationAgent",
            session_id=session_id,
            user_id=user_id,
            metadata={"user_id": user_id, "session_id": session_id, "debug": False}
        )
        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [CallbackHandler()],
        }
        if self._agent is None:
            self._agent = await self._create_agent()

        try:
            async for token, _ in self._agent.astream(
                    {
                        "messages": [HumanMessage(content=message)]
                    },
                    config,
                    stream_mode="messages"
            ):
                try:
                    yield token.content
                except Exception as token_error:
                    print(f"Error processing token, error={str(token_error)}, session_id={session_id}")
                    # Continue with next token even if current one fails
                    continue
        except Exception as stream_error:
            print(f"Error in stream processing, error={str(stream_error)}, session_id={session_id}")
            raise stream_error

    @staticmethod
    def extract_structured_response(response: Dict[str, Any]) -> MainAgentResponse | None:
        """Extract the structured response from an agent ainvoke response.

        Args:
            response (Dict[str, Any]): The raw response dictionary from the agent's ainvoke call.

        Returns:
            MainAgentResponse | None: The structured response containing message, is_conversation_finished, and hint,
            or None if the structured response is missing or invalid.
        """
        try:
            structured_response = response.get("structured_response")
            if structured_response:
                return structured_response
                # {
                #     "message": structured_response.message,
                #     "is_conversation_finished": structured_response.is_conversation_finished,
                #     "hint": structured_response.hint
                # }
            else:
                raise ValueError("No structured_response found in the provided response")
        except Exception as e:
            print(f"Error extracting structured response: {str(e)}, {traceback.format_exc()}")
            return None

    async def get_chat_history(self, session_id: str) -> list[str]:
        """Get the chat history for a given thread ID.

        Args:
            session_id (str): The session ID for the conversation.

        Returns:
            list[str]: The chat history.
        """
        if self._agent is None:
            self._agent = await self._create_agent()

        state: StateSnapshot = await self._agent.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )

        return state.values["messages"] if state.values else []

    async def clear_chat_history(self, session_id: str) -> None:
        """Clear all chat history for a given thread ID.

        Args:
            session_id: The ID of the session to clear history for.

        Raises:
            Exception: If there's an error clearing the chat history.
        """
        if self._agent is not None:
            try:
                await self._agent.checkpointer.adelete_thread(thread_id=session_id)
                print(f"Thread {session_id} deleted")
            except Exception as e:
                print(f"Error clearing {session_id} chat history, error={str(e)}, {traceback.format_exc()}")


async def main():
    agent = MainAgent()
    session_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    print(f"\n🚀 Система запущена!")
    print(f"📱 Session ID: {session_id}")
    print(f"🧵 User ID: {user_id}")
    print("💬 Введите 'q' для выхода\n")
    while True:
        try:
            user_input = input("вы: ")

            if user_input == "q":
                break

            if not user_input.strip():
                print("⚠️ Пожалуйста, введите сообщение.")
                continue

            # Вызываем граф с сообщением пользователя
            response = await agent.get_response(message=user_input, session_id=session_id, user_id=user_id)
            print(response)
            # Вывод сообщений бота через pretty_print
            if response and "messages" in response:
                for m in response["messages"]:
                    m.pretty_print()
            else:
                print("❌ Извините, произошла ошибка при обработке запроса.")

            # structured_response = agent.extract_structured_response(response)
            # print(f"Structured response {structured_response}")
            # Code for streaming response
            # async for token in agent.get_stream_response(message=user_input, session_id=session_id, user_id=user_id):
            #     print(token, end="", flush=True)  # Use end="" for continuous output like a stream

        except KeyboardInterrupt:
            print("\n👋 Чат прерван пользователем. До свидания!")
            break
        except Exception as e:
            print(f"❌ Неожиданная ошибка: {e} {traceback.format_exc()}")
            continue

if __name__ == '__main__':
    langfuse = Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_HOST")
    )
    asyncio.run(main())
