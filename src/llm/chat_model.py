from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from config.config import Config


class ChatModel:
    def __init__(self):
        self.config = Config()
        self.llm = ChatGoogleGenerativeAI(model=self.config.LLM_MODEL)

    def create_message(self, query, context):
        messages = [
            SystemMessage(
                content=self.config.SYSTEM_PROMPT.format(context=context)
            ),
            HumanMessage(
                content=query
            )
        ]

        return messages
    
    def generate_answer(self, query, context):
        messages = self.create_message(query, context)
        answer = self.llm.invoke(messages)
        messages.append(answer)

        return answer, messages