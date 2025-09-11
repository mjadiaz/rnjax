#!/bin/bash
set -e

echo "Running ER graphs..."
python er_graphs.py

echo "Running SF graphs..."
python sf_graphs.py

echo "Running M graphs..."
python m_graphs.py

echo "Running analysis data script..."
python python read_analyse_parallel.py

echo "All graph scripts completed successfully!"
