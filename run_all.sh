#!/bin/bash

# AI Weak Area Predictor - Complete System Launcher
# Perfect for GSoC demos and AI/ML company presentations

echo "🚀 Launching AI Weak Area Predictor - Complete System"
echo "=================================================="

# Activate virtual environment
echo "📦 Activating study_env virtual environment..."
source ../study_project/study_env/bin/activate

# Check if all dependencies are installed
echo "🔍 Checking dependencies..."
python -c "import streamlit, pandas, numpy, plotly, sklearn, xgboost, requests; print('✅ All dependencies installed')"

# Kill any existing streamlit processes
echo "🧹 Cleaning up existing processes..."
pkill -f streamlit 2>/dev/null || true

# Wait a moment for cleanup
sleep 2

echo "🎯 Starting applications..."

# Launch main application (port 8501)
echo "🧠 Starting Main AI Predictor (port 8501)..."
streamlit run weak_area_predictor.py --server.port=8501 &
MAIN_PID=$!

# Wait for main app to start
sleep 5

# Launch ML enhanced version (port 8502)  
echo "🤖 Starting ML Enhanced Dashboard (port 8502)..."
streamlit run ml_enhanced.py --server.port=8502 &
ML_PID=$!

# Wait for ML app to start
sleep 5

# Test ML models
echo "🧪 Testing ML Models..."
python -c "
import ml_model
try:
    engine = ml_model.create_ml_engine()
    print('✅ ML Engine initialized successfully')
except Exception as e:
    print(f'⚠️  ML Engine warning: {e}')
"

echo ""
echo "🎉 System Launch Complete!"
echo "=========================="
echo ""
echo "📱 Access Your Applications:"
echo "   Main AI Predictor:     http://localhost:8501"
echo "   ML Enhanced Dashboard: http://localhost:8502"
echo ""
echo "🎯 Perfect for demonstrating:"
echo "   ✅ AI-powered question generation (Groq API)"
echo "   ✅ Machine learning weak area prediction"
echo "   ✅ Advanced analytics and visualizations"
echo "   ✅ Production-ready architecture"
echo ""
echo "💡 For GSoC/AI-ML demos:"
echo "   1. Start with Main App (8501) - show user experience"
echo "   2. Switch to ML Dashboard (8502) - show technical depth"
echo "   3. Highlight real-time AI question generation"
echo "   4. Demonstrate ML model predictions"
echo ""
echo "🛑 To stop all services, run: ./stop_all.sh"
echo ""
echo "Process IDs: Main=$MAIN_PID, ML=$ML_PID"