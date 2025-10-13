# -*- coding: utf-8 -*-
"""
Custom AI Provider Integration
Supports OpenAI, Anthropic, and Google Gemini
"""

import os
import sys
import json
import logging
import asyncio
from typing import Optional, Dict, Any
import aiohttp
from openai import AsyncOpenAI
import anthropic

# Ensure UTF-8 encoding for all I/O operations
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)


async def retry_with_exponential_backoff(
    func,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0
):
    """
    Retry a function with exponential backoff
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
    
    Returns:
        Result of the function call
    
    Raises:
        Exception: Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            
            # Check if it's a retryable error
            is_rate_limit = 'rate' in error_str or '429' in error_str
            is_server_error = '500' in error_str or '502' in error_str or '503' in error_str or '504' in error_str
            is_timeout = 'timeout' in error_str or 'timed out' in error_str
            is_connection = 'connection' in error_str
            
            if not (is_rate_limit or is_server_error or is_timeout or is_connection):
                # Not a retryable error, raise immediately
                raise
            
            if attempt == max_retries:
                # Last attempt, raise the exception
                logger.error(f"All {max_retries} retry attempts failed: {str(e)}")
                raise
            
            # Calculate delay with exponential backoff
            wait_time = min(delay, max_delay)
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)
            delay *= exponential_base
    
    raise last_exception


class AIProvider:
    """Base class for AI providers"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def generate(self, prompt: str, system_message: str) -> str:
        raise NotImplementedError


class OpenAIProvider(AIProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.7):
        super().__init__(api_key)
        self.model = model
        self.temperature = temperature
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def generate(self, prompt: str, system_message: str) -> str:
        async def _generate():
            try:
                # Ensure strings are properly handled
                messages = [
                    {"role": "system", "content": str(system_message)},
                    {"role": "user", "content": str(prompt)}
                ]
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=16000
                )
                return response.choices[0].message.content
            except Exception as e:
                error_msg = str(e)
                # Improve error messages for token exhaustion
                if 'maximum context length' in error_msg.lower() or 'token' in error_msg.lower():
                    raise Exception(f"OpenAI token limit exceeded. Try reducing input size or using chunked generation. Details: {error_msg}")
                logger.error(f"OpenAI API error: {error_msg}")
                raise Exception(f"OpenAI generation failed: {error_msg}")
        
        return await retry_with_exponential_backoff(_generate)


class AnthropicProvider(AIProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, api_key: str, model: str = "claude-3-7-sonnet-20250219", temperature: float = 0.7):
        super().__init__(api_key)
        self.model = model
        self.temperature = temperature
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    async def generate(self, prompt: str, system_message: str) -> str:
        async def _generate():
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=16000,
                    temperature=self.temperature,
                    system=system_message,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
            except Exception as e:
                error_msg = str(e)
                # Improve error messages for token exhaustion
                if 'maximum context length' in error_msg.lower() or 'token' in error_msg.lower():
                    raise Exception(f"Anthropic token limit exceeded. Try reducing input size or using chunked generation. Details: {error_msg}")
                logger.error(f"Anthropic API error: {error_msg}")
                raise Exception(f"Anthropic generation failed: {error_msg}")
        
        return await retry_with_exponential_backoff(_generate)


class GeminiProvider(AIProvider):
    """Google Gemini provider"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-latest", temperature: float = 0.7):
        super().__init__(api_key)
        self.model = model
        self.temperature = temperature
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models"
    
    async def generate(self, prompt: str, system_message: str) -> str:
        async def _generate():
            try:
                # Combine system message and prompt for Gemini
                full_prompt = f"{system_message}\n\n{prompt}"
                
                url = f"{self.api_url}/{self.model}:generateContent?key={self.api_key}"
                
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": full_prompt
                        }]
                    }],
                    "generationConfig": {
                        "temperature": self.temperature,
                        "maxOutputTokens": 16000,
                    }
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"Gemini API error (status {response.status}): {error_text}")
                        
                        result = await response.json()
                        return result["candidates"][0]["content"]["parts"][0]["text"]
                        
            except Exception as e:
                error_msg = str(e)
                # Improve error messages for token exhaustion
                if 'token' in error_msg.lower() or 'limit' in error_msg.lower():
                    raise Exception(f"Gemini token limit exceeded. Try reducing input size or using chunked generation. Details: {error_msg}")
                logger.error(f"Gemini API error: {error_msg}")
                raise Exception(f"Gemini generation failed: {error_msg}")
        
        return await retry_with_exponential_backoff(_generate)


class AIProviderFactory:
    """Factory to create AI provider instances"""
    
    @staticmethod
    def create_provider(provider_name: str, api_key: str, model: Optional[str] = None, temperature: float = 0.7) -> AIProvider:
        """
        Create an AI provider instance
        
        Args:
            provider_name: 'openai', 'anthropic', or 'gemini'
            api_key: API key for the provider
            model: Optional model name (uses default if not provided)
            temperature: Temperature for generation (0.0-1.0)
        
        Returns:
            AIProvider instance
        """
        provider_map = {
            "openai": ("gpt-4o", OpenAIProvider),
            "anthropic": ("claude-3-7-sonnet-20250219", AnthropicProvider),
            "gemini": ("gemini-2.5-flash-latest", GeminiProvider)
        }
        
        if provider_name not in provider_map:
            raise ValueError(f"Unsupported provider: {provider_name}")
        
        default_model, provider_class = provider_map[provider_name]
        model_to_use = model or default_model
        
        return provider_class(api_key=api_key, model=model_to_use, temperature=temperature)


async def generate_with_ai(
    provider_name: str,
    api_key: str,
    prompt: str,
    system_message: str,
    model: Optional[str] = None,
    temperature: float = 0.7
) -> str:
    """
    Convenience function to generate content with any AI provider
    
    Args:
        provider_name: 'openai', 'anthropic', or 'gemini'
        api_key: API key for the provider
        prompt: User prompt
        system_message: System instruction
        model: Optional specific model to use
        temperature: Temperature for generation (0.0-1.0)
    
    Returns:
        Generated text response
    """
    provider = AIProviderFactory.create_provider(provider_name, api_key, model, temperature)
    return await provider.generate(prompt, system_message)

