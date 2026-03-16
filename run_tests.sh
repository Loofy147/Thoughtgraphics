#!/bin/bash
echo "--- RUNNING ALGORITHM TESTS ---"
python -m pytest test_v2.py

echo -e "\n--- RUNNING INTEGRATION TESTS ---"
python test_integration.py
