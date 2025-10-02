"""
AI Weak Area Predictor for Students
====================================

This module provides an AI-powered system to:
1. Generate intelligent assessment tests based on student topics
2. Predict weak areas through performance analysis
3. Provide personalized improvement suggestions
4. Track progress and adapt learning paths

Author: AI Assistant
Date: 2024
"""

import os
import json
import time
import uuid
import random
import sqlite3
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import requests
from llm_question_generator import LLMQuestionGenerator
from config import HF_API_KEY
from con import groq_api_key
from advanced_recommendations import create_recommendation_engine

# =============================
# CONFIGURATION
# =============================
st.set_page_config(page_title="AI Weak Area Predictor", layout="wide")
DB_PATH = "weak_area_predictor.db"
# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or groq_api_key

# Learning parameters
MASTERY_THRESHOLD = 0.7
WEAK_THRESHOLD = 0.4
IMPROVEMENT_THRESHOLD = 0.1

# Adaptive difficulty parameters
DIFFICULTY_ADJUSTMENT_FACTOR = 0.1
MIN_DIFFICULTY = 0.1
MAX_DIFFICULTY = 0.95
CONFIDENCE_WEIGHT = 0.3
PERFORMANCE_WEIGHT = 0.7

# Simple resource catalog mapping topic keywords to concrete chapters/sections
RESOURCE_CATALOG = [
    {
        'keywords': ['calculus', 'differentiation', 'derivative', 'integral'],
        'entries': [
            {
                'source': 'Khan Academy',
                'title': 'Differential calculus',
                'chapter': 'Chapter 2: Differentiation',
                'sections': ['2.1 The derivative', '2.2 Rules of differentiation', '2.3 Product/Quotient/Chain rules'],
                'url': 'https://www.khanacademy.org/math/differential-calculus'
            },
            {
                'source': 'Stewart Calculus',
                'title': 'Calculus: Early Transcendentals',
                'chapter': 'Chapter 3: Differentiation Rules',
                'sections': ['3.1 Derivatives of Polynomials', '3.3 Product & Quotient Rules', '3.4 Chain Rule'],
                'url': 'https://stewartcalculus.com'
            }
        ]
    },
    {
        'keywords': ['photosynthesis', 'biology', 'plant'],
        'entries': [
            {
                'source': 'OpenStax Biology',
                'title': 'Biology 2e',
                'chapter': 'Chapter 8: Photosynthesis',
                'sections': ['8.1 Energy and Metabolism', '8.2 Photosynthesis Overview', '8.3 Light-Dependent Reactions'],
                'url': 'https://openstax.org/details/books/biology-2e'
            },
            {
                'source': 'Crash Course',
                'title': 'Photosynthesis',
                'chapter': 'Episode 8',
                'sections': ['Chloroplasts', 'Light reactions', 'Calvin cycle'],
                'url': 'https://www.youtube.com/watch?v=sQK3Yr4Sc_k'
            }
        ]
    },
    {
        'keywords': ['python', 'programming', 'function', 'loop', 'variable'],
        'entries': [
            {
                'source': 'Automate the Boring Stuff',
                'title': 'Automate the Boring Stuff with Python',
                'chapter': 'Chapter 3: Functions',
                'sections': ['def statements', 'Parameters & arguments', 'Return values'],
                'url': 'https://automatetheboringstuff.com/'
            },
            {
                'source': 'Real Python',
                'title': 'Loops in Python',
                'chapter': 'Article',
                'sections': ['for loops', 'while loops', 'break/continue'],
                'url': 'https://realpython.com/python-for-loop/'
            }
        ]
    }
]

@dataclass
class Topic:
    name: str
    difficulty_level: float
    prerequisites: List[str]
    learning_objectives: List[str]

@dataclass
class Question:
    id: str
    topic: str
    text: str
    options: List[str]
    correct_answer: int
    difficulty: float
    cognitive_level: str  # 'remember', 'understand', 'apply', 'analyze', 'evaluate', 'create'
    time_limit: int  # seconds

@dataclass
class StudentResponse:
    question_id: str
    selected_answer: int
    time_taken: float
    is_correct: bool
    confidence_level: float = 0.5
    difficulty_rating: float = 0.5
    timestamp: Optional[datetime] = None

@dataclass
class WeakArea:
    topic: str
    weakness_score: float
    common_mistakes: List[str]
    improvement_suggestions: List[str]
    recommended_resources: List[str]
    practice_questions: List[str]

@dataclass
class GamificationProfile:
    student_id: str
    total_points: int
    current_level: int
    study_streak: int
    badges_earned: List[str]
    achievements: List[str]

@dataclass
class LearningPattern:
    student_id: str
    topic_id: str
    learning_curve_slope: float
    forgetting_rate: float
    optimal_review_interval: int
    learning_style_scores: Dict[str, float]
    attention_span: float
    cognitive_load_threshold: float

@dataclass
class StudySession:
    session_id: str
    student_id: str
    topic_id: str
    start_time: datetime
    end_time: datetime
    questions_answered: int
    correct_answers: int
    average_confidence: float
    session_quality_score: float

# =============================
# DATABASE SETUP
# =============================
def setup_database():
    """Initialize the database with comprehensive schema"""
    # Check if we should use PostgreSQL
    database_url = os.getenv('DATABASE_URL', '')
    if database_url and database_url.startswith('postgresql'):
        # Use PostgreSQL
        try:
            import psycopg2
            # Parse database URL
            # For simplicity, we'll use sqlite for now but this shows where to add PostgreSQL support
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        except ImportError:
            print("psycopg2 not installed, falling back to SQLite")
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    else:
        # Use SQLite
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    
    schema = """
    -- Students table
    CREATE TABLE IF NOT EXISTS students (
        student_id TEXT PRIMARY KEY,
        name TEXT,
        grade_level TEXT,
        learning_style TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Topics table
    CREATE TABLE IF NOT EXISTS topics (
        topic_id TEXT PRIMARY KEY,
        name TEXT UNIQUE,
        subject_area TEXT,
        difficulty_level REAL,
        prerequisites TEXT, -- JSON array
        learning_objectives TEXT, -- JSON array
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Questions table
    CREATE TABLE IF NOT EXISTS questions (
        question_id TEXT PRIMARY KEY,
        topic_id TEXT,
        text TEXT,
        options TEXT, -- JSON array
        correct_answer INTEGER,
        difficulty REAL,
        cognitive_level TEXT,
        time_limit INTEGER,
        quality_score REAL DEFAULT 0.0,
        feedback_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (topic_id) REFERENCES topics(topic_id)
    );
    
    -- Assessments table
    CREATE TABLE IF NOT EXISTS assessments (
        assessment_id TEXT PRIMARY KEY,
        student_id TEXT,
        title TEXT,
        topics TEXT, -- JSON array of topic_ids
        total_questions INTEGER,
        time_limit INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES students(student_id)
    );
    
    -- Student responses table
    CREATE TABLE IF NOT EXISTS student_responses (
        response_id TEXT PRIMARY KEY,
        assessment_id TEXT,
        question_id TEXT,
        student_id TEXT,
        selected_answer INTEGER,
        time_taken REAL,
        is_correct BOOLEAN,
        confidence_level REAL, -- 0.0 to 1.0
        difficulty_rating REAL, -- Student's perceived difficulty
        hint_used BOOLEAN DEFAULT FALSE,
        attempts INTEGER DEFAULT 1,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (assessment_id) REFERENCES assessments(assessment_id),
        FOREIGN KEY (question_id) REFERENCES questions(question_id),
        FOREIGN KEY (student_id) REFERENCES students(student_id)
    );
    
    -- Weak areas analysis table
    CREATE TABLE IF NOT EXISTS weak_areas (
        analysis_id TEXT PRIMARY KEY,
        student_id TEXT,
        topic_id TEXT,
        weakness_score REAL,
        common_mistakes TEXT, -- JSON array
        improvement_suggestions TEXT, -- JSON array
        recommended_resources TEXT, -- JSON array
        analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES students(student_id),
        FOREIGN KEY (topic_id) REFERENCES topics(topic_id)
    );
    
    -- Progress tracking table
    CREATE TABLE IF NOT EXISTS progress_tracking (
        progress_id TEXT PRIMARY KEY,
        student_id TEXT,
        topic_id TEXT,
        mastery_level REAL,
        improvement_rate REAL,
        last_assessed TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES students(student_id),
        FOREIGN KEY (topic_id) REFERENCES topics(topic_id)
    );
    
    -- Gamification table
    CREATE TABLE IF NOT EXISTS gamification (
        gamification_id TEXT PRIMARY KEY,
        student_id TEXT,
        total_points INTEGER DEFAULT 0,
        current_level INTEGER DEFAULT 1,
        study_streak INTEGER DEFAULT 0,
        last_study_date DATE,
        badges_earned TEXT, -- JSON array of badge names
        achievements TEXT, -- JSON array of achievements
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES students(student_id)
    );
    
    -- Learning patterns table
    CREATE TABLE IF NOT EXISTS learning_patterns (
        pattern_id TEXT PRIMARY KEY,
        student_id TEXT,
        topic_id TEXT,
        learning_curve_slope REAL,
        forgetting_rate REAL,
        optimal_review_interval INTEGER, -- days
        learning_style_score TEXT, -- JSON with visual/auditory/kinesthetic scores
        attention_span REAL,
        cognitive_load_threshold REAL,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES students(student_id),
        FOREIGN KEY (topic_id) REFERENCES topics(topic_id)
    );
    
    -- Study sessions table
    CREATE TABLE IF NOT EXISTS study_sessions (
        session_id TEXT PRIMARY KEY,
        student_id TEXT,
        topic_id TEXT,
        start_time TIMESTAMP,
        end_time TIMESTAMP,
        questions_answered INTEGER,
        correct_answers INTEGER,
        average_confidence REAL,
        session_quality_score REAL,
        FOREIGN KEY (student_id) REFERENCES students(student_id),
        FOREIGN KEY (topic_id) REFERENCES topics(topic_id)
    );
    """
    
    conn.executescript(schema)
    conn.commit()
    return conn

# =============================
# AI QUESTION GENERATION
# =============================
class AIQuestionGenerator:
    """Generates intelligent questions using Groq API"""
    
    def __init__(self, api_key: Optional[str] = None, provider: str = "groq"):
        self.llm_generator = LLMQuestionGenerator(api_key or "", provider)
        self.api_key = api_key
        self.provider = provider
    
    def generate_questions(self, topic: str, num_questions: int = 5, 
                          difficulty: str = "intermediate",
                          cognitive_levels: Optional[List[str]] = None,
                          grade_level: str = "high school") -> List[Dict]:
        """Generate questions for a specific topic using Groq API"""
        
        print(f"🤖 Calling {self.provider.upper()} API for topic: {topic}")
        print(f"📊 Generating {num_questions} questions at {difficulty} level")
        
        # Generate questions using Groq API
        questions = self.llm_generator.generate_questions(
            topic=topic,
            num_questions=num_questions,
            difficulty=difficulty,
            grade_level=grade_level
        )
        
        print(f"✅ Successfully generated {len(questions)} questions from {self.provider.upper()}")
        return questions

# =============================
# WEAK AREA ANALYSIS
# =============================
class WeakAreaAnalyzer:
    """Analyzes student performance to identify weak areas"""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
    
    def analyze_student_performance(self, student_id: str) -> List[WeakArea]:
        """Analyze student performance and identify weak areas"""
        # Get all responses for the student
        responses_df = self._get_student_responses(student_id)
        
        if responses_df.empty:
            return []
        
        # Calculate performance metrics by topic
        topic_performance = self._calculate_topic_performance(responses_df)

        # Build composite weakness scores and adaptive thresholds
        topic_scores = {}
        for topic_id, m in topic_performance.items():
            mistake_type_score = self._compute_mistake_type_score(m)
            # Composite score: higher means weaker
            # w1 accuracy, w2 time inefficiency, w3 mistake types
            w1, w2, w3 = 0.6, 0.2, 0.2
            score = (
                w1 * (1.0 - m.get('accuracy', 0.0)) +
                w2 * max(0.0, 1.0 - min(1.0, m.get('time_efficiency', 1.0))) +
                w3 * mistake_type_score
            )
            topic_scores[topic_id] = score
            m['composite_weakness_score'] = float(score)
            m['mistake_type_score'] = float(mistake_type_score)

        # Adaptive threshold: topics above 75th percentile of scores, and with enough data
        scores_series = pd.Series(topic_scores)
        if not scores_series.empty:
            adaptive_cutoff = float(scores_series.quantile(0.75))
        else:
            adaptive_cutoff = 0.0

        weak_areas = []
        for topic_id, metrics in topic_performance.items():
            # Require minimum data for confidence
            has_enough_data = metrics.get('total_questions', 0) >= 5
            is_weak_adaptive = metrics.get('composite_weakness_score', 0) >= adaptive_cutoff
            # Fallback: if little data, only consider very low accuracy
            is_weak_low_accuracy = metrics.get('accuracy', 1) < (WEAK_THRESHOLD if has_enough_data else 0.3)
            if (has_enough_data and is_weak_adaptive) or is_weak_low_accuracy:
                weak_area = self._create_weak_area_analysis(
                    student_id, topic_id, metrics, responses_df
                )
                weak_areas.append(weak_area)

        return sorted(weak_areas, key=lambda x: x.weakness_score, reverse=True)
    
    def _get_student_responses(self, student_id: str) -> pd.DataFrame:
        """Get all student responses as DataFrame"""
        query = """
        SELECT sr.*, q.topic_id, q.difficulty, q.cognitive_level, q.correct_answer
        FROM student_responses sr
        JOIN questions q ON sr.question_id = q.question_id
        WHERE sr.student_id = ?
        """
        return pd.read_sql_query(query, self.conn, params=(student_id,))
    
    def _calculate_topic_performance(self, responses_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics by topic"""
        topic_metrics = {}
        
        for topic_id in responses_df['topic_id'].unique():
            topic_data = responses_df[responses_df['topic_id'] == topic_id]
            
            # Difficulty-weighted accuracy
            weights = topic_data['difficulty'].clip(lower=0.3, upper=1.0)
            weighted_accuracy = float(np.average(topic_data['is_correct'].astype(float), weights=weights)) if len(topic_data) else 0.0

            # Time metrics
            avg_time = float(topic_data['time_taken'].mean()) if len(topic_data) else 0.0
            median_time = float(topic_data['time_taken'].median()) if len(topic_data) else 0.0
            time_efficiency = float(median_time / max(avg_time, 1e-6)) if avg_time else 0.0

            # Difficulty bands
            bands = pd.cut(topic_data['difficulty'], bins=[0, 0.4, 0.7, 1.0], labels=['easy', 'medium', 'hard'])
            band_perf = topic_data.assign(band=bands).groupby('band', observed=False)['is_correct'].mean().fillna(0).to_dict()

            metrics = {
                'accuracy': float(topic_data['is_correct'].mean()),
                'weighted_accuracy': weighted_accuracy,
                'total_questions': int(len(topic_data)),
                'avg_time': avg_time,
                'median_time': median_time,
                'time_efficiency': time_efficiency,
                'difficulty_distribution': topic_data['difficulty'].round(1).value_counts().sort_index().to_dict(),
                'difficulty_band_performance': {str(k): float(v) for k, v in band_perf.items()},
                'cognitive_level_performance': {str(k): float(v) for k, v in topic_data.groupby('cognitive_level', observed=False)['is_correct'].mean().fillna(0).to_dict().items()},
                'common_mistakes': self._identify_common_mistakes(topic_data),
                'distractor_patterns': self._analyze_distractor_patterns(topic_data)
            }
            topic_metrics[topic_id] = metrics
        
        return topic_metrics

    def _compute_mistake_type_score(self, metrics: Dict) -> float:
        """Score mistake types: conceptual mistakes weigh higher than careless.
        Return in [0,1]. Uses proxies from metrics.
        """
        score = 0.0
        # If hard band performance is low, conceptual issues
        hard_perf = metrics.get('difficulty_band_performance', {}).get('hard', 1.0)
        if hard_perf < 0.4:
            score += 0.5
        # If cognitive levels above understand are weak, conceptual
        cog_perf = metrics.get('cognitive_level_performance', {})
        higher_levels = [lvl for lvl in cog_perf.keys() if lvl not in ('remember', 'understand')]
        if higher_levels:
            low_higher = [lvl for lvl in higher_levels if cog_perf.get(lvl, 1) < 0.5]
            if low_higher:
                score += 0.3
        # If time efficiency indicates rushing and low accuracy, careless
        if metrics.get('time_efficiency', 1) < 0.7 and metrics.get('accuracy', 1) < 0.6:
            score += 0.2
        return min(1.0, score)
    
    def _identify_common_mistakes(self, topic_data: pd.DataFrame) -> List[str]:
        """Identify common mistakes in a topic"""
        incorrect_responses = topic_data[topic_data['is_correct'] == False]
        
        if incorrect_responses.empty:
            return []
        
        # Analyze patterns in incorrect responses
        mistakes = []
        
        # Time-based mistakes (too fast/slow)
        avg_time = topic_data['time_taken'].mean()
        fast_mistakes = incorrect_responses[incorrect_responses['time_taken'] < avg_time * 0.5]
        if not fast_mistakes.empty:
            mistakes.append("Rushing through questions without careful consideration")
        
        # Difficulty-based mistakes
        hard_questions = incorrect_responses[incorrect_responses['difficulty'] > 0.7]
        if not hard_questions.empty:
            mistakes.append("Struggling with higher difficulty concepts")
        
        return mistakes

    def _analyze_distractor_patterns(self, topic_data: pd.DataFrame) -> Dict[str, int]:
        """Analyze over-selected wrong options to detect distractor traps."""
        if 'selected_answer' not in topic_data.columns or 'correct_answer' not in topic_data.columns:
            return {}
        incorrect = topic_data[topic_data['is_correct'] == False]
        if incorrect.empty:
            return {}
        counts = incorrect['selected_answer'].value_counts().to_dict()
        # keep only 0-3 indices
        return {str(k): int(v) for k, v in counts.items() if k in [0, 1, 2, 3]}
    
    def _create_weak_area_analysis(self, student_id: str, topic_id: str, 
                                 metrics: Dict, responses_df: pd.DataFrame) -> WeakArea:
        """Create detailed weak area analysis"""
        # Get topic name
        topic_name = self.conn.execute(
            "SELECT name FROM topics WHERE topic_id = ?", (topic_id,)
        ).fetchone()['name']
        
        # Calculate weakness score (0-1, higher = weaker)
        weakness_score = 1 - metrics['accuracy']
        
        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(topic_id, metrics)
        
        # Get recommended resources
        resources = self._get_recommended_resources(topic_id)
        
        # Generate practice questions
        practice_questions = self._generate_practice_questions(topic_id, metrics)
        
        return WeakArea(
            topic=topic_name,
            weakness_score=weakness_score,
            common_mistakes=metrics['common_mistakes'],
            improvement_suggestions=suggestions,
            recommended_resources=resources,
            practice_questions=practice_questions
        )
    
    def _generate_improvement_suggestions(self, topic_id: str, metrics: Dict) -> List[str]:
        """Generate personalized improvement suggestions"""
        suggestions = []
        
        # Accuracy-based suggestions
        if metrics.get('accuracy', 0) < 0.3:
            suggestions.append("Start with fundamentals: revisit key definitions/examples and solve 10 easy problems.")
            suggestions.append("Review prerequisite topics before advancing.")
        elif metrics.get('accuracy', 0) < 0.5:
            suggestions.append("Targeted practice: complete 15 medium problems with step-by-step solutions.")
            suggestions.append("For each mistake, write a brief note on the misconception.")
        
        # Weighted accuracy emphasizes difficulty mastery
        if metrics.get('weighted_accuracy', 1) < 0.45 and metrics.get('accuracy', 1) >= 0.5:
            suggestions.append("Focus on hard items: add 10 hard-level problems emphasizing multi-step reasoning.")

        # Time-based suggestions
        if metrics.get('avg_time', 0) > 300:  # 5 minutes per question
            suggestions.append("Use a 2-minute timer per item for the next 10 questions.")
            suggestions.append("Practice recognizing problem type quickly before solving.")
        elif metrics.get('time_efficiency', 1) > 1.2:
            suggestions.append("You're accurate but slow: prepare a checklist to streamline steps and reduce time.")
        elif metrics.get('time_efficiency', 1) < 0.7 and metrics.get('accuracy', 1) < 0.5:
            suggestions.append("Likely rushing: pause 10 seconds to outline your approach before answering.")
        
        # Cognitive level suggestions
        for level, performance in metrics.get('cognitive_level_performance', {}).items():
            if performance < 0.5:
                if level in ("remember", "understand"):
                    suggestions.append(f"Boost {level}: create flashcards and do 10 recall drills.")
                else:
                    suggestions.append(f"Improve {level}: review 5 worked examples, then solve 10 practice items.")

        # Difficulty band suggestions
        bands = metrics.get('difficulty_band_performance', {})
        if bands.get('hard', 1) < 0.4:
            suggestions.append("Hard-level plan: do 3 sets of 5 hard problems; annotate where you got stuck.")

        # Distractor patterns
        distractors = metrics.get('distractor_patterns', {})
        if distractors:
            common_idx = max(distractors, key=lambda k: distractors[k])
            suggestions.append(f"You often pick option {chr(65+int(common_idx))} when wrong. Double-check signs, units, and edge cases before finalizing.")
        
        return suggestions
    
    def _get_recommended_resources(self, topic_id: str) -> List[str]:
        """Get recommended learning resources for a topic"""
        # Try to map topic name to concrete chapters/sections
        topic_row = self.conn.execute("SELECT name FROM topics WHERE topic_id = ?", (topic_id,)).fetchone()
        topic_name = topic_row['name'] if topic_row else topic_id
        name_lower = topic_name.lower()

        for catalog in RESOURCE_CATALOG:
            if any(k in name_lower for k in catalog['keywords']):
                # Format detailed resource entries
                formatted = []
                for e in catalog['entries']:
                    sections = ", ".join(e.get('sections', [])[:3])
                    formatted.append(f"{e['source']} - {e['title']} | {e['chapter']} | Sections: {sections} | {e['url']}")
                return formatted

        # Fallback generic resources
        return [
            f"Overview chapter for {topic_name}",
            f"Practice set: 20 problems on {topic_name}",
            f"Video lecture: key concepts in {topic_name}"
        ]
    
    def _generate_practice_questions(self, topic_id: str, metrics: Dict) -> List[str]:
        """Generate practice questions for weak areas"""
        # This would integrate with the question generator
        return [
            f"Practice question 1 for {topic_id}",
            f"Practice question 2 for {topic_id}",
            f"Practice question 3 for {topic_id}"
        ]

# =============================
# ASSESSMENT CREATOR
# =============================
class AssessmentCreator:
    """Creates intelligent assessments based on student needs"""
    
    def __init__(self, conn: sqlite3.Connection, question_generator: AIQuestionGenerator):
        self.conn = conn
        self.question_generator = question_generator
    
    def create_adaptive_assessment(self, student_id: str, topics: List[str], 
                                 num_questions: int = 20) -> str:
        """Create an adaptive assessment based on student's weak areas"""
        assessment_id = f"ASSESS_{uuid.uuid4().hex[:8]}"
        
        # Get student's current mastery levels
        mastery_levels = self._get_student_mastery(student_id, topics)
        
        # Generate questions based on mastery levels
        questions = []
        for topic in topics:
            topic_id = self._get_or_create_topic(topic)
            
            # Determine difficulty based on mastery
            mastery = mastery_levels.get(topic_id, 0.5)
            difficulty_range = self._calculate_difficulty_range(mastery)
            
            # Generate questions for this topic
            # Distribute questions evenly among topics, but ensure at least 1 question per topic
            per_topic = max(1, num_questions // max(1, len(topics)))
            # Convert difficulty range to string
            if difficulty_range[1] <= 0.4:
                difficulty = "beginner"
            elif difficulty_range[1] <= 0.7:
                difficulty = "intermediate"
            else:
                difficulty = "advanced"
            
            topic_questions = self.question_generator.generate_questions(
                topic, 
                num_questions=per_topic,
                difficulty=difficulty
            )
            
            # DEBUG: Print the generated questions
            print(f"DEBUG: Generated {len(topic_questions)} questions for topic '{topic}':")
            for i, q in enumerate(topic_questions):
                text_preview = q.get('text', 'NO TEXT')[:50]
                print(f"  Question {i+1}: {text_preview}...")
            
            # Check if we got real questions or fallback questions
            if topic_questions and len(topic_questions) > 0:
                first_question = topic_questions[0].get('text', '').lower()
                if 'important concept' in first_question or 'fundamental principle' in first_question:
                    print(f"⚠️  WARNING: Got fallback questions for topic '{topic}'")
                else:
                    print(f"✅ Got real questions for topic '{topic}'")
            else:
                print(f"⚠️  WARNING: No questions generated for topic '{topic}'")
            
            # Store questions in database
            for q in topic_questions:
                question_id = f"Q_{uuid.uuid4().hex[:8]}"
                try:
                    print(f"Attempting to store question: {q['text'][:50]}...")
                    self.conn.execute("""
                        INSERT INTO questions (question_id, topic_id, text, options, 
                                            correct_answer, difficulty, cognitive_level, time_limit,
                                            quality_score, feedback_count, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (
                        question_id, topic_id, q['text'], 
                        json.dumps(q['options']), q['correct_answer'],
                        q['difficulty'], q['cognitive_level'], 120, 0.0, 0
                    ))
                    questions.append(question_id)
                    print(f"✅ Successfully stored question ID: {question_id}")
                except Exception as e:
                    print(f"❌ Failed to store question: {e}")
                    # Still add the question ID to the list to avoid breaking the flow
                    questions.append(question_id)
            
            # TEMPORARILY DISABLED: Collect features for model improvement
            # for q in topic_questions:
            #     try:
            #         from data_collector import DataCollector
            #         collector = DataCollector()
            #         features = collector.collect_question_generation_features(topic, q)
            #         collector.store_features('question_generation_features', features)
            #     except Exception as e:
            #         print(f"Warning: Could not collect features: {e}")
        
        # Create assessment record
        self.conn.execute("""
            INSERT INTO assessments (assessment_id, student_id, title, topics, 
                                  total_questions, time_limit)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            assessment_id, student_id, f"Adaptive Assessment - {datetime.now().strftime('%Y-%m-%d')}",
            json.dumps([self._get_or_create_topic(t) for t in topics]),
            len(questions), len(questions) * 120
        ))
        
        self.conn.commit()
        return assessment_id
    
    def _get_student_mastery(self, student_id: str, topics: List[str]) -> Dict[str, float]:
        """Get student's current mastery levels for topics"""
        mastery_levels = {}
        
        for topic in topics:
            topic_id = self._get_or_create_topic(topic)
            
            # Get recent performance for this topic
            query = """
                SELECT AVG(is_correct) as accuracy
                FROM student_responses sr
                JOIN questions q ON sr.question_id = q.question_id
                WHERE sr.student_id = ? AND q.topic_id = ?
                AND sr.timestamp > datetime('now', '-30 days')
            """
            result = self.conn.execute(query, (student_id, topic_id)).fetchone()
            
            mastery_levels[topic_id] = result['accuracy'] if result['accuracy'] else 0.5
        
        return mastery_levels
    
    def _get_or_create_topic(self, topic_name: str) -> str:
        """Get or create a topic in the database"""
        # Check if topic exists
        result = self.conn.execute(
            "SELECT topic_id FROM topics WHERE name = ?", (topic_name,)
        ).fetchone()
        
        if result:
            return result['topic_id']
        
        # Create new topic
        topic_id = f"TOPIC_{uuid.uuid4().hex[:8]}"
        self.conn.execute("""
            INSERT INTO topics (topic_id, name, subject_area, difficulty_level, 
                              prerequisites, learning_objectives)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            topic_id, topic_name, "General", 0.5,
            json.dumps([]), json.dumps([f"Understand {topic_name}"])
        ))
        self.conn.commit()
        return topic_id
    
    def _calculate_difficulty_range(self, mastery: float) -> Tuple[float, float]:
        """Calculate appropriate difficulty range based on mastery level"""
        if mastery < 0.3:
            return (0.2, 0.5)  # Easy to medium
        elif mastery < 0.7:
            return (0.4, 0.7)  # Medium difficulty
        else:
            return (0.6, 0.9)  # Medium to hard

# =============================
# PROGRESS TRACKER
# =============================
class ProgressTracker:
    """Tracks and visualizes student progress"""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
    
    def update_progress(self, student_id: str, assessment_id: str):
        """Update student progress after assessment"""
        # Get assessment responses
        responses = self.conn.execute("""
            SELECT sr.*, q.topic_id, q.difficulty
            FROM student_responses sr
            JOIN questions q ON sr.question_id = q.question_id
            WHERE sr.assessment_id = ?
        """, (assessment_id,)).fetchall()
        
        # Calculate mastery levels by topic
        topic_mastery = {}
        for response in responses:
            topic_id = response['topic_id']
            if topic_id not in topic_mastery:
                topic_mastery[topic_id] = {'correct': 0, 'total': 0}
            
            topic_mastery[topic_id]['total'] += 1
            if response['is_correct']:
                topic_mastery[topic_id]['correct'] += 1
        
        # Update progress tracking
        for topic_id, stats in topic_mastery.items():
            mastery_level = stats['correct'] / stats['total']
            
            # Check if progress record exists
            existing = self.conn.execute("""
                SELECT progress_id FROM progress_tracking 
                WHERE student_id = ? AND topic_id = ?
            """, (student_id, topic_id)).fetchone()
            
            if existing:
                # Update existing record
                self.conn.execute("""
                    UPDATE progress_tracking 
                    SET mastery_level = ?, last_assessed = CURRENT_TIMESTAMP
                    WHERE student_id = ? AND topic_id = ?
                """, (mastery_level, student_id, topic_id))
            else:
                # Create new record
                progress_id = f"PROG_{uuid.uuid4().hex[:8]}"
                self.conn.execute("""
                    INSERT INTO progress_tracking 
                    (progress_id, student_id, topic_id, mastery_level, 
                     improvement_rate, last_assessed)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (progress_id, student_id, topic_id, mastery_level, 0.0))
        
        self.conn.commit()
    
    def get_progress_chart(self, student_id: str) -> go.Figure:
        """Generate progress visualization"""
        # Get progress data
        query = """
            SELECT t.name as topic_name, pt.mastery_level, pt.last_assessed
            FROM progress_tracking pt
            JOIN topics t ON pt.topic_id = t.topic_id
            WHERE pt.student_id = ?
            ORDER BY pt.last_assessed DESC
        """
        data = pd.read_sql_query(query, self.conn, params=(student_id,))
        
        if data.empty:
            return go.Figure()
        
        # Create progress chart
        fig = px.bar(
            data, 
            x='topic_name', 
            y='mastery_level',
            title='Student Mastery Levels by Topic',
            labels={'mastery_level': 'Mastery Level', 'topic_name': 'Topic'},
            color='mastery_level',
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(
            yaxis=dict(range=[0, 1]),
            showlegend=False
        )
        
        return fig

# =============================
# LEARNING ANALYZER
# =============================
class LearningAnalyzer:
    """Analyzes learning patterns and adapts to student needs"""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
    
    def analyze_learning_patterns(self, student_id: str) -> Dict:
        """Analyze student's learning patterns over time"""
        # Get student's response history
        query = """
        SELECT sr.*, q.topic_id, q.difficulty, q.cognitive_level
        FROM student_responses sr
        JOIN questions q ON sr.question_id = q.question_id
        WHERE sr.student_id = ?
        ORDER BY sr.timestamp
        """
        responses_df = pd.read_sql_query(query, self.conn, params=(student_id,))
        
        if responses_df.empty:
            return {}
        
        # Calculate learning velocity (improvement rate over time)
        responses_df['date'] = pd.to_datetime(responses_df['timestamp'])
        daily_performance = responses_df.groupby(responses_df['date'].dt.date)['is_correct'].mean()
        
        # Calculate overall improvement
        if len(daily_performance) > 1:
            improvement_rate = (daily_performance.iloc[-1] - daily_performance.iloc[0]) / len(daily_performance)
        else:
            improvement_rate = 0.0
        
        # Identify preferred cognitive levels
        cognitive_preferences = responses_df.groupby('cognitive_level')['is_correct'].mean().to_dict()
        
        # Identify optimal difficulty range
        difficulty_performance = responses_df.groupby('difficulty')['is_correct'].mean()
        optimal_difficulty = difficulty_performance.idxmax() if not difficulty_performance.empty else 0.5
        
        return {
            'improvement_rate': float(improvement_rate),
            'cognitive_preferences': cognitive_preferences,
            'optimal_difficulty': float(optimal_difficulty),
            'total_questions_answered': len(responses_df),
            'overall_accuracy': float(responses_df['is_correct'].mean())
        }
    
    def recommend_adaptive_strategy(self, student_id: str) -> Dict:
        """Recommend adaptive learning strategies based on patterns"""
        patterns = self.analyze_learning_patterns(student_id)
        
        recommendations = {
            'difficulty_adjustment': 'maintain',
            'focus_topics': [],
            'practice_strategy': 'mixed'
        }
        
        # Adjust difficulty based on improvement rate
        if patterns.get('improvement_rate', 0) < 0.01:
            recommendations['difficulty_adjustment'] = 'decrease'
        elif patterns.get('improvement_rate', 0) > 0.05:
            recommendations['difficulty_adjustment'] = 'increase'
        
        # Recommend practice strategy
        cognitive_prefs = patterns.get('cognitive_preferences', {})
        if cognitive_prefs.get('remember', 0) < 0.8 and cognitive_prefs.get('understand', 0) < 0.7:
            recommendations['practice_strategy'] = 'foundation_first'
        elif cognitive_prefs.get('apply', 0) > 0.8 or cognitive_prefs.get('analyze', 0) > 0.7:
            recommendations['practice_strategy'] = 'advanced_focus'
        
        return recommendations

# =============================
# MAIN APPLICATION
# =============================
def main():
    """Main Streamlit application"""
    st.title("🧠 AI Weak Area Predictor")
    st.markdown("Intelligent assessment and improvement system for students")
    
    # Health check endpoint
    if st.query_params.get("healthz"):
        st.write("OK")
        st.stop()
    
    # Initialize database and components
    conn = setup_database()
    
    # Groq API Configuration
    st.sidebar.subheader("🤖 Groq API Configuration")
    
    # Use Groq API key from config
    GROQ_API_KEY = os.getenv("GROQ_API_KEY") or groq_api_key
    
    if not GROQ_API_KEY:
        st.sidebar.write("**Status:** ⚠️ API Key Needed")
        st.sidebar.write("**Provider:** Groq (Fast LLMs)")
        st.sidebar.write("**Get API Key:** https://console.groq.com/keys")
        st.sidebar.write("**Free Tier:** Generous rate limits")
        
        # Use fallback mode
        question_generator = AIQuestionGenerator(None, "groq")
        st.warning("⚠️ Please add your Groq API key to use real AI questions!")
    else:
        st.sidebar.write("**Status:** ✅ Groq API Key Configured")
        st.sidebar.write(f"**Provider:** Groq (Fast LLMs)")
        st.sidebar.write(f"**API Key:** {GROQ_API_KEY[:20]}...")
        
        # Initialize Groq-powered question generator
        question_generator = AIQuestionGenerator(GROQ_API_KEY, "groq")
        st.success("🚀 Connected to Groq API! Ready to generate intelligent questions.")
    
    analyzer = WeakAreaAnalyzer(conn)
    learning_analyzer = LearningAnalyzer(conn)
    assessment_creator = AssessmentCreator(conn, question_generator)
    progress_tracker = ProgressTracker(conn)
    
    # Initialize session state
    if 'student_id' not in st.session_state:
        st.session_state.student_id = f"STUDENT_{uuid.uuid4().hex[:8]}"
    
    # Sidebar for student setup
    with st.sidebar:
        st.subheader("Student Setup")
        student_name = st.text_input("Student Name", "John Doe")
        grade_level = st.selectbox("Grade Level", ["Elementary", "Middle School", "High School", "College"])
        learning_style = st.selectbox("Learning Style", ["Visual", "Auditory", "Kinesthetic", "Mixed"])
        
        if st.button("Update Student Profile"):
            conn.execute("""
                INSERT OR REPLACE INTO students (student_id, name, grade_level, learning_style)
                VALUES (?, ?, ?, ?)
            """, (st.session_state.student_id, student_name, grade_level, learning_style))
            conn.commit()
            st.success("Student profile updated!")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Assessment", "📊 Weak Areas", "📈 Progress", "🎯 Recommendations", "🧠 Learning Patterns"])
    
    with tab5:
        st.subheader("Learning Pattern Analysis")
        
        if st.button("Analyze Learning Patterns"):
            with st.spinner("Analyzing your learning patterns..."):
                patterns = learning_analyzer.analyze_learning_patterns(st.session_state.student_id)
                recommendations = learning_analyzer.recommend_adaptive_strategy(st.session_state.student_id)
                
                if patterns:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Overall Accuracy", f"{patterns.get('overall_accuracy', 0):.2f}")
                        st.metric("Questions Answered", patterns.get('total_questions_answered', 0))
                    
                    with col2:
                        st.metric("Improvement Rate", f"{patterns.get('improvement_rate', 0):.3f}")
                        st.metric("Optimal Difficulty", f"{patterns.get('optimal_difficulty', 0.5):.2f}")
                    
                    with col3:
                        st.write("**Recommended Strategy:**")
                        st.write(f"Difficulty: {recommendations.get('difficulty_adjustment', 'maintain')}")
                        st.write(f"Practice: {recommendations.get('practice_strategy', 'mixed')}")
                    
                    st.write("**Cognitive Preferences:**")
                    cognitive_prefs = patterns.get('cognitive_preferences', {})
                    for level, accuracy in cognitive_prefs.items():
                        st.progress(accuracy, f"{level.title()}: {accuracy:.2f}")
                else:
                    st.info("Not enough data to analyze learning patterns. Complete some assessments first.")
    
    with tab1:
        st.subheader("Create Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            topics_input = st.text_area(
                "Topics (one per line)", 
                "Mathematics\nScience\nEnglish\nHistory"
            )
            # Add control for questions per topic
            questions_per_topic = st.slider("Questions per Topic", 1, 20, 5)
            num_questions = st.slider("Total Questions (approximate)", 5, 50, 20)
        
        with col2:
            difficulty_preference = st.select_slider(
                "Difficulty Preference", 
                options=["Beginner", "Intermediate", "Advanced"],
                value="Intermediate"
            )
            time_limit = st.slider("Time Limit (minutes)", 10, 120, 30)
        
        if st.button("Generate Assessment"):
            topics = [t.strip() for t in topics_input.split('\n') if t.strip()]
            
            with st.spinner("Generating assessment..."):
                assessment_id = assessment_creator.create_adaptive_assessment(
                    st.session_state.student_id, topics, num_questions
                )
                st.session_state.assessment_id = assessment_id
                st.session_state.answers = {}  # Initialize answers
                st.success(f"Assessment created! ID: {assessment_id}")
        
        # Display assessment questions if available
        if 'assessment_id' in st.session_state:
            st.subheader("Take Assessment")
            
            # Get questions for this specific assessment
            questions_query = """
                SELECT q.*, t.name as topic_name
                FROM questions q
                JOIN topics t ON q.topic_id = t.topic_id
                JOIN assessments a ON a.assessment_id = ?
                WHERE q.topic_id IN (
                    SELECT value as topic_id
                    FROM json_each(a.topics)
                )
                AND q.text NOT LIKE '%important concept%' 
                AND q.text NOT LIKE '%fundamental principle%'
                ORDER BY q.created_at DESC
                LIMIT 50
            """
            
            try:
                questions_df = pd.read_sql_query(questions_query, conn, params=(st.session_state.assessment_id,))
                
                # Debug info
                st.write(f"📊 Debug: Found {len(questions_df)} questions in database")
                
                if not questions_df.empty:
                    st.write(f"**Assessment ID:** {st.session_state.assessment_id}")
                    st.write(f"**Total Questions:** {len(questions_df)}")
                    
                    # Show first question as preview
                    first_q = questions_df.iloc[0]
                    st.write(f"📝 Preview: {first_q['text'][:100]}...")
                    
                    # Initialize answers if not exists
                    if 'answers' not in st.session_state:
                        st.session_state.answers = {}
                    
                    # Display questions
                    for idx, row in questions_df.iterrows():
                        st.write(f"**Question {idx + 1}:** {row['text']}")
                        st.write(f"*Topic: {row['topic_name']} | Difficulty: {row['difficulty']:.2f}*")
                        
                        options = json.loads(row['options'])
                        selected = st.radio(
                            "Select your answer:",
                            options,
                            key=f"q_{row['question_id']}",
                            index=st.session_state.answers.get(row['question_id'], 0)
                        )
                        
                        # Store answer
                        st.session_state.answers[row['question_id']] = options.index(selected)
                    
                    # Submit button
                    if st.button("Submit Assessment"):
                        # Record responses
                        for idx, row in questions_df.iterrows():
                            selected_idx = st.session_state.answers.get(row['question_id'], 0)
                            is_correct = selected_idx == row['correct_answer']
                            
                            response_id = f"RESP_{uuid.uuid4().hex[:8]}"
                            time_taken = 60.0  # This should be dynamically calculated
                            conn.execute("""
                                INSERT INTO student_responses 
                                (response_id, assessment_id, question_id, student_id, 
                                 selected_answer, time_taken, is_correct, confidence_level, difficulty_rating, hint_used, attempts)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                response_id, st.session_state.assessment_id, 
                                row['question_id'], st.session_state.student_id,
                                selected_idx, time_taken, is_correct, 0.5, 0.5, False, 1
                            ))
                        
                        conn.commit()
                        st.success("Assessment submitted successfully!")
                        st.session_state.answers = {}  # Clear answers
                
            except Exception as e:
                st.error(f"Error loading assessment: {e}")
    
    with tab2:
        st.subheader("Weak Area Analysis")
        
        if st.button("Analyze Weak Areas"):
            with st.spinner("Analyzing performance..."):
                weak_areas = analyzer.analyze_student_performance(st.session_state.student_id)
                
                if weak_areas:
                    st.write("### Identified Weak Areas:")

                    for i, weak_area in enumerate(weak_areas):
                        with st.expander(f"🔴 {weak_area.topic} (Score: {weak_area.weakness_score:.2f})"):
                            # Fetch latest metrics from DB for display
                            topic_row = self_conn = None
                            # Display key metrics
                            st.write("**Key Metrics:**")
                            colA, colB, colC = st.columns(3)
                            with colA:
                                st.metric("Accuracy", f"{(1-weak_area.weakness_score):.2f}")
                            with colB:
                                # Weighted accuracy not on WeakArea; infer from analyzer if needed
                                st.write("Weighted accuracy, time, bands shown below")
                            with colC:
                                st.write("")

                            # Improvement suggestions
                            st.write("**Improvement Suggestions:**")
                            for suggestion in weak_area.improvement_suggestions:
                                st.write(f"• {suggestion}")

                            # Common mistakes and resources
                            st.write("**Common Mistakes:**")
                            for mistake in weak_area.common_mistakes:
                                st.write(f"• {mistake}")

                            st.write("**Recommended Resources:**")
                            for resource in weak_area.recommended_resources:
                                st.write(f"• {resource}")
                else:
                    st.info("No weak areas identified. Great job!")
    
    with tab3:
        st.subheader("Progress Tracking")
        
        if st.button("Update Progress"):
            if 'assessment_id' in st.session_state:
                progress_tracker.update_progress(st.session_state.student_id, st.session_state.assessment_id)
                st.success("Progress updated!")
        
        # Display progress chart
        progress_chart = progress_tracker.get_progress_chart(st.session_state.student_id)
        if progress_chart.data:
            st.plotly_chart(progress_chart, use_container_width=True)
        else:
            st.info("No progress data available. Complete an assessment first.")
    
    with tab4:
        st.subheader("🎆 Advanced Personalized Recommendations")
        
        # Initialize advanced recommendation engine
        if 'rec_engine' not in st.session_state:
            st.session_state.rec_engine = create_recommendation_engine(conn)
        
        # Get student's weak areas for recommendations
        weak_areas_list = [vars(weak_area) for weak_area in analyzer.analyze_student_performance(st.session_state.student_id)]
        
        if weak_areas_list:
            # Generate advanced recommendations
            with st.spinner("🤖 AI is analyzing your learning patterns..."):
                recommendations = st.session_state.rec_engine.generate_personalized_recommendations(
                    st.session_state.student_id, weak_areas_list
                )
            
            # Display comprehensive recommendations
            st.write("### 🎯 AI-Powered Study Strategy")
            
            # Create tabs for different recommendation types
            rec_tab1, rec_tab2, rec_tab3, rec_tab4, rec_tab5, rec_tab6 = st.tabs([
                "🚑 Immediate", "📅 Study Plan", "📚 Resources", 
                "🏋️ Practice", "👥 Peer Insights", "🛛 Learning Path"
            ])
            
            with rec_tab1:
                st.write("#### 🚑 Immediate Actions (Do Today!)")
                for action in recommendations['immediate_actions']:
                    st.write(f"• {action}")
            
            with rec_tab2:
                st.write("#### 📅 Personalized Study Plan")
                for plan_item in recommendations['study_plan']:
                    st.write(f"• {plan_item}")
            
            with rec_tab3:
                st.write("#### 📚 Curated Learning Resources")
                for resource in recommendations['resources']:
                    st.write(f"• {resource}")
            
            with rec_tab4:
                st.write("#### 🏋️ Adaptive Practice Exercises")
                for exercise in recommendations['practice_exercises']:
                    st.write(f"• {exercise}")
            
            with rec_tab5:
                st.write("#### 👥 Peer Learning Insights")
                for insight in recommendations['peer_insights']:
                    st.write(f"• {insight}")
            
            with rec_tab6:
                st.write("#### 🛛 Adaptive Learning Path")
                for path_item in recommendations['adaptive_path']:
                    st.write(f"• {path_item}")
            
            # Add success metrics and motivation
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🎯 Improvement Potential", "85%", "+15%")
            with col2:
                st.metric("🔥 Study Streak Goal", "7 days", "+2 days")
            with col3:
                st.metric("🏆 Mastery Timeline", "2 weeks", "-3 days")
            
            # Gamification elements
            st.write("### 🎮 Achievement Unlocks")
            achievements = [
                "🌟 Complete 3 immediate actions → Unlock 'Quick Learner' badge",
                "📚 Finish Week 1 study plan → Unlock 'Consistent Student' badge",
                "🏆 Master weak area → Unlock 'Weakness Crusher' badge"
            ]
            
            for achievement in achievements:
                st.write(f"• {achievement}")
                
        else:
            st.info(" YYS Amazing! No weak areas detected.")
            st.balloons()
            
            # Recommendations for high performers
            st.write("### 🏆 Excellence Pathway")
            excellence_recs = [
                "🚀 Challenge Mode: Try competition-level problems",
                "👥 Mentor Others: Teaching reinforces your mastery",
                "🌍 Explore Advanced Topics: Expand your knowledge horizons",
                "🏅 Set Stretch Goals: Aim for 95%+ accuracy in new topics"
            ]
            
            for rec in excellence_recs:
                st.write(f"• {rec}")

if __name__ == "__main__":
    main()