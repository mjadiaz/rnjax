#!/bin/bash
set -e

# python extract_metrics.py --base-dir save_control/ER_sparse_stdp --local
# python extract_metrics.py --base-dir save_control/ER_sparse_stdp_control --local --control

echo "Running SF graphs..."
python sf_graphs.py

python extract_metrics.py --base-dir save_control/SF_balanced_dense_stdp --local
python extract_metrics.py --base-dir save_control/SF_balanced_dense_stdp_control --local --control
python extract_metrics.py --base-dir save_control/SF_balanced_intermediate_stdp --local
python extract_metrics.py --base-dir save_control/SF_balanced_intermediate_stdp_control --local --control


echo "Running SF graphs..."
python m_graphs.py

python extract_metrics.py --base-dir save_control/SBM_dense_4_stdp --local
python extract_metrics.py --base-dir save_control/SBM_dense_4_stdp_control --local --control
python extract_metrics.py --base-dir save_control/SBM_intermediate_4_stdp --local
python extract_metrics.py --base-dir save_control/SBM_intermediate_4_stdp_control --local --control

echo "Running ER graphs..."
python er_graphs.py

python extract_metrics.py --base-dir save_control/ER_dense_stdp --local
python extract_metrics.py --base-dir save_control/ER_dense_stdp_control --local --control
python extract_metrics.py --base-dir save_control/ER_intermediate_stdp --local
python extract_metrics.py --base-dir save_control/ER_intermediate_stdp_control --local --control

#echo "Running analysis data script..."
#python python read_analyse_parallel.py

echo "All graph scripts completed successfully!"
