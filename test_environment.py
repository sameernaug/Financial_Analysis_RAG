import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

# Test basic imports
try:
    import yfinance as yf
    print("✓ yfinance imported successfully")
except ImportError as e:
    print("✗ yfinance import failed:", e)

try:
    import pandas as pd
    print("✓ pandas imported successfully")
except ImportError as e:
    print("✗ pandas import failed:", e)

try:
    import chromadb
    print("✓ chromadb imported successfully")
except ImportError as e:
    print("✗ chromadb import failed:", e)

print("Environment test complete")