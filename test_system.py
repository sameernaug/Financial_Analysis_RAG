"""
System Testing Script for Financial RAG System
Verifies all components are working correctly
"""

import os
import sys
import json
from datetime import datetime

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

def test_system():
    """Test all system components"""
    print("🧪 Testing Financial RAG System...")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Data Acquisition
    print("\n1. Testing Data Acquisition...")
    try:
        from scripts.data_acquisition import FinancialDataAcquisition
        acquirer = FinancialDataAcquisition()
        acquirer.run_data_acquisition()
        
        # Check if data files exist
        data_dir = "data"
        if os.path.exists(data_dir):
            market_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
            if market_files:
                print("✅ Data acquisition test passed")
                tests_passed += 1
            else:
                print("❌ No market data files found")
        else:
            print("❌ Data directory not found")
    except Exception as e:
        print(f"❌ Data acquisition failed: {str(e)}")
    
    # Test 2: Data Processing
    print("\n2. Testing Data Processing...")
    try:
        from scripts.data_processor import FinancialDataProcessor
        processor = FinancialDataProcessor()
        processor.run_processing()
        
        # Check for processed chunks
        if os.path.exists("data/chunks"):
            chunk_files = [f for f in os.listdir("data/chunks") if f.endswith('.json')]
            if chunk_files:
                print("✅ Data processing test passed")
                tests_passed += 1
            else:
                print("❌ No chunk files found")
        else:
            print("❌ Chunks directory not found")
    except Exception as e:
        print(f"❌ Data processing failed: {str(e)}")
    
    # Test 3: Vector Store
    print("\n3. Testing Vector Store...")
    try:
        from scripts.vector_store import FinancialVectorStore
        vector_store = FinancialVectorStore()
        
        # Test collection creation
        vector_store.setup_collections()
        
        # Test query
        results = vector_store.query_all_collections("investment", limit=3)
        if results:
            print("✅ Vector store test passed")
            tests_passed += 1
        else:
            print("❌ Vector store query failed")
    except Exception as e:
        print(f"❌ Vector store test failed: {str(e)}")
    
    # Test 4: RAG Pipeline
    print("\n4. Testing RAG Pipeline...")
    try:
        from scripts.rag_pipeline import FinancialRAGPipeline
        pipeline = FinancialRAGPipeline()
        
        # Test risk calculation
        risk_metrics = pipeline.calculate_risk_metrics("AAPL", 30)
        if risk_metrics and 'volatility' in risk_metrics:
            print("✅ RAG pipeline test passed")
            tests_passed += 1
        else:
            print("❌ RAG pipeline risk calculation failed")
    except Exception as e:
        print(f"❌ RAG pipeline test failed: {str(e)}")
    
    # Test 5: Streamlit App
    print("\n5. Testing Streamlit Application...")
    try:
        import streamlit
        
        # Check if app.py exists
        if os.path.exists("app.py"):
            print("✅ Streamlit app test passed")
            tests_passed += 1
        else:
            print("❌ app.py not found")
    except Exception as e:
        print(f"❌ Streamlit app test failed: {str(e)}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! System is ready to use.")
        print("\nTo start the application:")
        print("streamlit run app.py")
    else:
        print("⚠️ Some tests failed. Check the logs above.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    test_system()