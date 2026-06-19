from __future__ import annotations

import os
from .prompts import TISTORY_SEO_SYSTEM_PROMPT, HUMANIZE_SYSTEM_PROMPT


class LLMClient:
    def __init__(self, provider: str | None = None, api_key: str | None = None) -> None:
        self.provider: str = (provider or os.getenv("LLM_PROVIDER") or "gemini").lower()
        self.api_key: str | None = api_key
        self.client = None
        self._initialize()

    def _initialize(self) -> None:
        gemini_key = self.api_key or os.getenv("GEMINI_API_KEY")
        openai_key = self.api_key or os.getenv("OPENAI_API_KEY")

        # Fallback between providers if one key is missing but the other exists
        if self.provider == "gemini" and not gemini_key and openai_key:
            self.provider = "openai"
        elif self.provider == "openai" and not openai_key and gemini_key:
            self.provider = "gemini"

        if self.provider == "gemini":
            if not gemini_key:
                print("[LLMClient] GEMINI_API_KEY is not set. LLM is disabled.", flush=True)
                return
            try:
                from google import genai
                self.client = genai.Client(api_key=gemini_key)
            except ImportError:
                print("[LLMClient] Failed to import 'google-genai'. Please install google-genai.", flush=True)
        elif self.provider == "openai":
            if not openai_key:
                print("[LLMClient] OPENAI_API_KEY is not set. LLM is disabled.", flush=True)
                return
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=openai_key)
            except ImportError:
                print("[LLMClient] Failed to import 'openai'. Please install openai.", flush=True)
        else:
            print(f"[LLMClient] Unknown provider '{self.provider}'. LLM is disabled.", flush=True)

    def is_enabled(self) -> bool:
        return self.client is not None

    def generate_full_post(self, text: str) -> str:
        if not self.is_enabled():
            return text

        text_strip = text.strip()
        if not text_strip:
            return text

        try:
            if self.provider == "gemini":
                # For google-genai SDK, use gemini-2.5-flash
                # Added retry logic for 5 RPM free tier limit and transient errors (503, etc.)
                import time
                from google.genai.errors import APIError
                
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        response = self.client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=text_strip,
                            config=dict(
                                system_instruction=TISTORY_SEO_SYSTEM_PROMPT,
                                temperature=0.7,
                            ),
                        )
                        return response.text.strip()
                    except APIError as e:
                        if e.code in (429, 500, 503, 504) and attempt < max_retries - 1:
                            sleep_time = (2 ** attempt) * 5
                            print(f"[LLMClient] Temporary API error ({e.code}) in generate_full_post. Sleeping {sleep_time}s before retry {attempt+1}/{max_retries}...", flush=True)
                            time.sleep(sleep_time)
                        else:
                            raise e
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": TISTORY_SEO_SYSTEM_PROMPT},
                        {"role": "user", "content": text_strip},
                    ],
                    temperature=0.7,
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLMClient] API call failed: {e}. Falling back to original text.", flush=True)
            return text
        return text

    def humanize(self, text: str) -> str:
        if not self.is_enabled():
            return text

        text_strip = text.strip()
        if not text_strip:
            return text

        try:
            if self.provider == "gemini":
                import time
                from google.genai.errors import APIError
                
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        response = self.client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=text_strip,
                            config=dict(
                                system_instruction=HUMANIZE_SYSTEM_PROMPT,
                                temperature=0.7,
                            ),
                        )
                        return response.text.strip()
                    except APIError as e:
                        if e.code in (429, 500, 503, 504) and attempt < max_retries - 1:
                            sleep_time = (2 ** attempt) * 5
                            print(f"[LLMClient] Temporary API error ({e.code}) in humanize. Sleeping {sleep_time}s before retry {attempt+1}/{max_retries}...", flush=True)
                            time.sleep(sleep_time)
                        else:
                            raise e
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": HUMANIZE_SYSTEM_PROMPT},
                        {"role": "user", "content": text_strip},
                    ],
                    temperature=0.7,
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLMClient] humanize API call failed: {e}. Falling back to original text.", flush=True)
            return text
        return text

