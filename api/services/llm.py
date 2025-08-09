import logging
import re
from typing import Dict, List
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

class LLMService:
    def __init__(self, llm_provider: str = "openai", model: str | None = None):
        """Initialize LLM service with specified provider
        
        Args:
            llm_provider: The LLM provider to use ('openai', 'gemini')
            model: Optional model name override to use for the given provider
        """
        self.provider = llm_provider
        self.model = model
        self.llm = self._get_llm(llm_provider)
        
        self.prompt_template = PromptTemplate(
            template="""Answer the question based on the provided context. If you cannot find 
            the answer in the context, say "I cannot answer this question based on the provided context."
            
            Also answer the section of the document that answers the question.
            Context: {context}
            
            Question: {question}
            
            Answer in this format: 
            
            <answer>
            <answer based on the document to the question>
            </answer>
            
            <references>
            <the literal text of the document that answers the question>
            </references>
            
            Answer: """,
            input_variables=["context", "question"]
        )
    
    def _get_llm(self, provider: str):
        """Get LLM instance based on provider
        
        Args:
            provider: LLM provider name
        """
        
        if provider == "openai":
            return ChatOpenAI(
                model=self.model or "gpt-4.1-mini",
                temperature=0,
            )
        elif provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=self.model or "gemini-2.0-flash-lite",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def select_response(self, response: str) -> Dict:
        """Select the response from the LLM
        
        Args:
            response: Response from the LLM
        """
        response = re.sub(r"(<answer>)|(</answer>)|(</references>)", "", response)
        return response.split("<references>")
    
    def generate_answer(self, question: str, relevant_docs: List[str]) -> Dict:
        """Generate answer based on question and relevant documents
        
        Args:
            question: User's question
            relevant_docs: List of relevant document contents
            
        Returns:
            Dict containing answer and sources
        """
        if not relevant_docs:
            return {
                "answer": "No relevant context found in the documents. Please try a different question or upload relevant documents.",
                "references": []
            }
            
        # Combine relevant documents into context
        context = "\n\n".join(relevant_docs)
        
        
        # Generate prompt
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        logging.info(f"Prompt: {prompt}")
        
        # Get response from LLM
        response = self.llm.invoke(prompt)
        
        logging.info(f"Response: {response.content}")
        final_response = self.select_response(response.content)            
        
        if (len(final_response) < 2):
            return {
                "answer": "I am sorry, I cannot answer this question based on the provided context.",
                "references": []
            }
            
        return {
            "answer": final_response[0],
            "references": final_response[1]
        }