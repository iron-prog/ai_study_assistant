"""
LLM-Powered Question Generator
=============================

This module integrates with real LLMs to generate intelligent, 
contextual questions for any topic.
"""

import requests
import json
import random
from typing import List, Dict, Optional
import time
from config import HF_API_KEY

class LLMQuestionGenerator:
    """Generates intelligent questions using Groq API"""
    
    def __init__(self, api_key: str = "", provider: str = "groq"):
        # Accept runtime key; fallback to env-configured key
        self.api_key = api_key or HF_API_KEY
        self.provider = "groq"
        self.base_urls = {
            # Groq API endpoint
            "groq": "https://api.groq.com/openai/v1/chat/completions",
        }
        
        # Use multiple models for better reliability
        self.models = {
            "groq": "llama-3.3-70b-versatile",
        }
    
    def generate_questions(self, topic: str, num_questions: int = 5, 
                          difficulty: str = "intermediate", 
                          grade_level: str = "high school") -> List[Dict]:
        """Generate intelligent questions using LLM"""
        
        if not self.api_key:
            print(f"❌ No API key provided, using fallback questions for topic: {topic}")
            return self._fallback_questions(topic, num_questions)
        
        prompt = self._create_prompt(topic, num_questions, difficulty, grade_level)
        
        try:
            print(f"🤖 Calling LLM API for topic: {topic}")
            response = self._call_llm_api(prompt)
            print(f"✅ Received response from LLM API")
            questions = self._parse_response(response)
            print(f"📄 Parsed {len(questions)} questions")
            
            # Check if we got real questions or fallback questions
            if questions and len(questions) > 0:
                first_question = questions[0].get('text', '').lower()
                if 'important concept' in first_question or 'fundamental principle' in first_question:
                    print(f"⚠️  Detected fallback questions for topic '{topic}'")
                else:
                    print(f"✅ Got real questions for topic '{topic}'")
            
            # Fallback if parsing fails or returns empty
            if not questions:
                print(f"⚠️  No questions parsed, using fallback for topic: {topic}")
                return self._fallback_questions(topic, num_questions)
            
            result = questions[:num_questions]  # Ensure we don't exceed requested number
            print(f"✅ Returning {len(result)} questions for topic: {topic}")
            return result

        except Exception as e:
            print(f"❌ LLM API Error for topic '{topic}': {e}")
            print(f"🔄 Using fallback questions for topic: {topic}")
            return self._fallback_questions(topic, num_questions)
    
    def _create_prompt(self, topic: str, num_questions: int, difficulty: str, grade_level: str) -> str:
        """Create a detailed prompt for question generation"""
        
        prompt = f"""
You are an expert teacher and educational content creator. Create {num_questions} high-quality multiple choice questions specifically about "{topic}" for {grade_level} students.

CRITICAL REQUIREMENTS:
1. Questions must be SPECIFIC to "{topic}" - not generic
2. Difficulty level: {difficulty}
3. Each question should test different aspects of {topic}
4. Include realistic examples, calculations, or scenarios
5. Make distractors plausible but clearly wrong
6. Questions should be practical and applicable

TOPIC-SPECIFIC GUIDELINES:
- For MATH topics: Include calculations, formulas, or problem-solving
- For SCIENCE topics: Include experiments, processes, or scientific concepts
- For PROGRAMMING topics: Include code examples, syntax, or logic
- For HISTORY topics: Include dates, events, or cause-effect relationships
- For LANGUAGE topics: Include grammar, vocabulary, or literary concepts

EXAMPLES OF GOOD QUESTIONS:
- Math: "What is the derivative of f(x) = x³ - 2x² + 5x - 1?"
- Science: "In photosynthesis, what gas is released as a byproduct?"
- Programming: "Which Python function is used to read user input?"

Return ONLY a valid JSON array in this exact format:
[
    {{
        "question": "Specific question about {topic} with concrete details",
        "options": [
            "Correct answer with specific details",
            "Plausible but wrong answer",
            "Another wrong but related answer", 
            "Clearly incorrect distractor"
        ],
        "correct_answer": 0,
        "explanation": "Detailed explanation of why the correct answer is right",
        "difficulty": 0.6,
        "cognitive_level": "apply"
    }}
]

IMPORTANT:
- Return ONLY the JSON array, no other text
- Make questions SPECIFIC to "{topic}" with concrete details
- Include realistic examples, numbers, or scenarios
- Ensure correct_answer is 0, 1, 2, or 3
- Make explanations educational and helpful
"""
        return prompt
    
    def _call_llm_api(self, prompt: str) -> str:
        """Call Groq API only"""
        return self._call_groq_api(prompt)
    
    def _call_groq_api(self, prompt: str) -> str:
        """Call Groq API"""
        url = self.base_urls["groq"]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.models["groq"],
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.7
        }
        
        print(f"🌐 Calling Groq API with model: {self.models['groq']}")
        print(f"🔑 API Key length: {len(self.api_key)}")
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            print(f"📡 Response Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ API Error: {response.text}")
                response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"✅ API call successful! Response length: {len(content)}")
            return content
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Request Error: {e}")
            raise
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
            raise
    
    def _parse_response(self, response_text: str) -> List[Dict]:
        """Parse LLM response and extract questions"""
        try:
            # Clean the response text
            response_text = response_text.strip()
            print(f"🔍 Raw response: {response_text[:200]}...")
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                # Find the actual JSON content
                lines = response_text.split('\n')
                start_idx = 0
                end_idx = len(lines)
                
                for i, line in enumerate(lines):
                    if line.strip().startswith('```'):
                        if start_idx == 0:
                            start_idx = i + 1
                        else:
                            end_idx = i
                            break
                            
                response_text = '\n'.join(lines[start_idx:end_idx])
                print(f"📝 Cleaned from markdown: {response_text[:200]}...")
            
            # Find JSON array in response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                print("❌ No JSON array found in response")
                print(f"Response text: {response_text}")
                raise ValueError("No JSON array found in response")
            
            json_text = response_text[start_idx:end_idx]
            print(f"📝 Extracted JSON: {json_text[:200]}...")
            
            # Handle truncated JSON by trying to fix incomplete entries
            if not json_text.endswith(']'):
                # Try to fix truncated JSON
                last_brace = json_text.rfind('}')
                if last_brace > 0:
                    json_text = json_text[:last_brace + 1] + ']'
                    print(f"🔧 Fixed truncated JSON")
            
            questions = json.loads(json_text)
            print(f"✅ Parsed {len(questions)} questions from JSON")
            
            # Validate and clean questions
            cleaned_questions = []
            for i, q in enumerate(questions):
                if self._validate_question(q):
                    cleaned_q = self._clean_question(q)
                    cleaned_questions.append(cleaned_q)
                    print(f"✅ Question {i+1}: {cleaned_q['text'][:50]}...")
                else:
                    print(f"❌ Invalid question {i+1}: {q}")
            
            return cleaned_questions
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"Raw response: {response_text}")
            return []
        except Exception as e:
            print(f"❌ Error parsing LLM response: {e}")
            return []
    
    def _validate_question(self, question: Dict) -> bool:
        """Validate that a question has all required fields"""
        required_fields = ["question", "options", "correct_answer"]
        return all(field in question for field in required_fields)
    
    def _clean_question(self, question: Dict) -> Dict:
        """Clean and standardize question format"""
        return {
            "text": question["question"],
            "options": question["options"],
            "correct_answer": int(question["correct_answer"]),
            "difficulty": float(question.get("difficulty", 0.5)),
            "cognitive_level": question.get("cognitive_level", "understand"),
            "explanation": question.get("explanation", "No explanation provided")
        }
    
    def _fallback_questions(self, topic: str, num_questions: int) -> List[Dict]:
        """Fallback questions when LLM is not available or generates poor quality"""
        print(f"🔄 Generating fallback questions for topic: {topic}")
        
        # Topic-specific fallback questions
        topic_lower = topic.lower()
        
        if any(word in topic_lower for word in ["math", "calculus", "algebra", "differentiation", "derivative"]):
            print(f"📘 Using math fallback questions for topic: {topic}")
            return self._math_fallback_questions(topic, num_questions)
        elif any(word in topic_lower for word in ["science", "biology", "chemistry", "physics", "photosynthesis"]):
            print(f"🔬 Using science fallback questions for topic: {topic}")
            return self._science_fallback_questions(topic, num_questions)
        elif any(word in topic_lower for word in ["programming", "python", "code", "function", "variable"]):
            print(f"💻 Using programming fallback questions for topic: {topic}")
            return self._programming_fallback_questions(topic, num_questions)
        else:
            print(f"📚 Using general fallback questions for topic: {topic}")
            return self._general_fallback_questions(topic, num_questions)
    
    def _math_fallback_questions(self, topic: str, num_questions: int) -> List[Dict]:
        """Math-specific fallback questions"""
        questions = []
        
        templates = [
            {
                "text": f"What is the derivative of f(x) = x² + 3x - 1?",
                "options": ["2x + 3", "2x - 3", "x + 3", "x² + 3"],
                "correct_answer": 0,
                "cognitive_level": "apply"
            },
            {
                "text": f"Which rule is used to find the derivative of a product of two functions?",
                "options": ["Product rule", "Chain rule", "Quotient rule", "Power rule"],
                "correct_answer": 0,
                "cognitive_level": "remember"
            },
            {
                "text": f"What is the derivative of sin(x)?",
                "options": ["cos(x)", "-cos(x)", "-sin(x)", "tan(x)"],
                "correct_answer": 0,
                "cognitive_level": "remember"
            }
        ]
        
        for i in range(num_questions):
            template = templates[i % len(templates)]
            questions.append({
                "text": template["text"],
                "options": template["options"],
                "correct_answer": template["correct_answer"],
                "difficulty": 0.5 + (i * 0.1),
                "cognitive_level": template["cognitive_level"],
                "explanation": f"This is a fundamental concept in {topic}"
            })
        
        return questions
    
    def _science_fallback_questions(self, topic: str, num_questions: int) -> List[Dict]:
        """Science-specific fallback questions"""
        questions = []
        
        templates = [
            {
                "text": f"What is the chemical formula for water?",
                "options": ["H₂O", "H₂O₂", "CO₂", "O₂"],
                "correct_answer": 0,
                "cognitive_level": "remember"
            },
            {
                "text": f"In photosynthesis, what gas is absorbed by plants?",
                "options": ["Carbon dioxide", "Oxygen", "Nitrogen", "Hydrogen"],
                "correct_answer": 0,
                "cognitive_level": "remember"
            },
            {
                "text": f"What is the process by which plants convert sunlight into energy?",
                "options": ["Photosynthesis", "Respiration", "Transpiration", "Digestion"],
                "correct_answer": 0,
                "cognitive_level": "understand"
            }
        ]
        
        for i in range(num_questions):
            template = templates[i % len(templates)]
            questions.append({
                "text": template["text"],
                "options": template["options"],
                "correct_answer": template["correct_answer"],
                "difficulty": 0.5 + (i * 0.1),
                "cognitive_level": template["cognitive_level"],
                "explanation": f"This is a fundamental concept in {topic}"
            })
        
        return questions
    
    def _programming_fallback_questions(self, topic: str, num_questions: int) -> List[Dict]:
        """Programming-specific fallback questions"""
        questions = []
        
        templates = [
            {
                "text": f"Which Python function is used to read user input?",
                "options": ["input()", "read()", "get()", "scan()"],
                "correct_answer": 0,
                "cognitive_level": "remember"
            },
            {
                "text": f"What is the correct syntax to define a function in Python?",
                "options": ["def function_name():", "function function_name():", "define function_name():", "func function_name():"],
                "correct_answer": 0,
                "cognitive_level": "remember"
            },
            {
                "text": f"Which data type is used to store a sequence of characters in Python?",
                "options": ["String", "Integer", "Float", "Boolean"],
                "correct_answer": 0,
                "cognitive_level": "remember"
            }
        ]
        
        for i in range(num_questions):
            template = templates[i % len(templates)]
            questions.append({
                "text": template["text"],
                "options": template["options"],
                "correct_answer": template["correct_answer"],
                "difficulty": 0.5 + (i * 0.1),
                "cognitive_level": template["cognitive_level"],
                "explanation": f"This is a fundamental concept in {topic}"
            })
        
        return questions
    
    def _general_fallback_questions(self, topic: str, num_questions: int) -> List[Dict]:
        """General fallback questions"""
        return [
            {
                "text": f"What is the most important concept in {topic}?",
                "options": [
                    f"The fundamental principle of {topic}",
                    f"A secondary aspect of {topic}",
                    f"An unrelated concept",
                    f"A common misconception about {topic}"
                ],
                "correct_answer": 0,
                "difficulty": 0.5,
                "cognitive_level": "remember",
                "explanation": f"This is a fundamental concept in {topic}"
            }
        ] * num_questions