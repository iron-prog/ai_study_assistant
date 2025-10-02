# AI Weak Area Predictor

An intelligent system that generates assessments, detects weak areas with adaptive analytics, and provides actionable, chapter-level study recommendations using Groq's fast LLMs.

## Features

### 🧠 Core Functionality
- **Intelligent Assessment Generation**: Adaptive tests from student topics using Groq's Llama 3.3 70B
- **Weak Area Detection (Enhanced)**: Composite score + adaptive thresholds to identify real gaps
- **Personalized Recommendations**: Actionable steps driven by mistake types and difficulty bands
- **Progress Tracking**: Visualizes mastery across topics over time
- **Adaptive Learning**: Difficulty adjusts based on performance

### 📊 Analysis Capabilities (Upgraded)
- **Composite Weakness Score**: Combines (1 − accuracy), time efficiency, and mistake-type score
- **Adaptive Thresholds**: Per-student 75th-percentile cutoff for weak topics
- **Difficulty-Weighted Accuracy**: Harder questions carry more weight
- **Time Efficiency**: Median vs average time to detect rushing or struggle
- **Difficulty Bands**: Easy / Medium / Hard performance breakdown
- **Cognitive Levels**: remember / understand / apply / analyze / evaluate / create
- **Distractor Patterns**: Detects over-selected wrong options (trap choices)
- **Resource Recommendations (Chapters/Sections)**: Topic-matched chapters with key sections and URLs

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone or download the project
2. Install requirements:
```bash
pip install -r requirements.txt
```

### Environment Variables
Set your Groq API key (recommended):
```bash
export GROQ_API_KEY="your_groq_api_key"
```

## Usage

### Advanced Version (with Groq)
```bash
streamlit run weak_area_predictor.py
```
Notes:
- The app reads your key from `GROQ_API_KEY` or `study_project/config.py` (GROQ_API_KEY fallback).
- Complete at least one assessment before analyzing weak areas.

### Workflow
1) Student Setup → create/update profile
2) Assessment Creation → enter topics, set size and difficulty
   - The app generates questions using Groq's Llama 3.3 70B
   - Ensures at least 5 questions per topic for reliable analysis
3) Take Assessment → answer questions; responses are logged with timing
4) Weak Area Analysis (Enhanced)
   - Computes composite score per topic
   - Uses adaptive thresholds to select weak topics
   - Surfaces: weighted accuracy, time efficiency, band and cognitive performance
   - Detects distractor traps and common mistake patterns
   - Recommends concrete chapters/sections with URLs
5) Recommendations & Progress → view plan and track mastery over time

## Database Schema

SQLite tables:
- `students`: Student profiles
- `topics`: Topics and difficulty levels
- `questions`: Generated questions with metadata
- `assessments`: Assessment sessions and configuration
- `student_responses`: Responses including correctness and timing
- `weak_areas`: Reserved for analysis results (optional persistence)

## Configuration

### Learning Parameters
- `MASTERY_THRESHOLD`: 0.7 (mastery baseline)
- `WEAK_THRESHOLD`: 0.4 (fallback for sparse data)
- `IMPROVEMENT_THRESHOLD`: 0.1 (10% improvement)
- Adaptive cutoff: 75th-percentile composite score per student

### Question Generation
- Provider: Groq (Fast LLMs)
- Model: `llama-3.3-70b-versatile`
- Endpoint: `https://api.groq.com/openai/v1/chat/completions`

## Troubleshooting

1. API Key not connected
   - Ensure `GROQ_API_KEY` is exported in the shell running Streamlit
   - Restart the app after setting the key
2. No weak areas detected
   - Complete an assessment first; ensure ≥ 5 questions per topic
   - Increase topic diversity and difficulty spread
3. Low question quality
   - Use clearer, more specific topic names
   - Regenerate assessment with higher difficulty preference

## Customization

- Edit `_generate_improvement_suggestions` in `weak_area_predictor.py` to tune feedback
- Extend `RESOURCE_CATALOG` to map topics to your textbooks/courses
- Adjust metric weights in `_compute_mistake_type_score` and composite score

## What's New
- Groq API integration for ultra-fast question generation
- Composite weakness scoring with adaptive thresholds
- Difficulty-weighted accuracy and time-efficiency metrics
- Difficulty-band and cognitive-level breakdowns
- Distractor pattern analysis
- Topic-mapped chapter/section resources with URLs
- Minimum 5 questions per topic for reliability

## Future Enhancements

### Planned Features
- [ ] Multi-language support
- [ ] Advanced analytics dashboard (trend lines, cohort comparisons)
- [ ] LMS integration
- [ ] Mobile app version
- [ ] Collaborative learning features
- [ ] Gamification elements

### Technical Improvements
- [ ] Simple ML model (e.g., logistic regression) for weak-topic probability
- [ ] Adaptive mastery models per topic (Bayesian Knowledge Tracing)
- [ ] Cohort-based adaptive thresholds by grade/subject
- [ ] Rich visuals for bands/cognitive levels over time
- [ ] Background jobs for scheduled reassessment suggestions
"""
AI Weak Area Predictor for Students - OPTIMIZED VERSION
=====================================================

This module provides an AI-powered system to:
1. Generate intelligent assessment tests based on student topics
2. Predict weak areas through performance analysis
3. Provide personalized improvement suggestions
4. Track progress and adapt learning paths

Optimizations:
- Improved database connection management with connection pooling
- Enhanced caching strategies for better performance
- Refactored class architecture for better separation of concerns
- Optimized SQL queries with proper indexing
- Better error handling and logging
- Reduced code duplication and improved maintainability
- Enhanced async processing capabilities
- Better memory management
"""