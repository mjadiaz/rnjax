# Parallel Processing Guide for Neural Network Analysis

This guide explains how to use the parallel processing implementation to analyze neural network data across multiple steps efficiently.

## Files Overview

- `read_analyse_parallel.py` - Basic parallel implementation
- `read_analyse_parallel_enhanced.py` - Enhanced version with advanced features
- `config.py` - Configuration file for all parameters
- `README_parallel.md` - This guide

## Quick Start

### 1. Basic Usage

```bash
python read_analyse_parallel.py
```

This will:
- Use all available CPU cores
- Process all steps found in the base directory
- Create `metrics.csv` for each step
- Create a combined `combined_metrics.csv` file

### 2. Enhanced Usage

```bash
python read_analyse_parallel_enhanced.py
```

This provides additional features:
- Better error handling and recovery
- Memory optimization options
- Timeout protection
- Detailed logging and progress reporting
- System information display

## Configuration

Edit `config.py` to customize the processing:

```python
# Basic settings
MAX_WORKERS = 8          # Use 8 CPU cores
BATCH_SIZE = 20          # Expected batch size
I_EXT = 10.0            # External input parameter

# Memory optimization
MEMORY_EFFICIENT = True  # Enable for large datasets

# Timeout settings
BATCH_TIMEOUT = 300      # 5 minutes per batch
STEP_TIMEOUT = 3600      # 1 hour per step

# Error handling
CONTINUE_ON_ERROR = True        # Don't stop on single step failures
MAX_FAILED_BATCHES = 5          # Skip step if too many batches fail
```

## Performance Tuning

### CPU Usage

```python
# Use all cores (default)
MAX_WORKERS = multiprocessing.cpu_count()

# Use half the cores (good for shared systems)
MAX_WORKERS = multiprocessing.cpu_count() // 2

# Use specific number of cores
MAX_WORKERS = 4
```

### Memory Management

For large datasets, enable memory optimization:

```python
MEMORY_EFFICIENT = True
```

This will:
- Convert float64 to float32 where possible
- Optimize data types for memory usage

### Timeout Configuration

Adjust timeouts based on your data complexity:

```python
BATCH_TIMEOUT = 600     # 10 minutes for complex batches
STEP_TIMEOUT = 7200     # 2 hours for large steps
```

## Output Files

### Individual Step Metrics

Each step creates: `{base_dir}/{step}/metrics.csv`

Contains columns:
- All entropy measures (`emsrs`)
- Lempel-Ziv complexity measures (`lz`)
- Sample entropy measures (`sp`)
- Global network metrics (`gm`)
- Driver fraction metrics (`drf`)
- Base metrics (suffixed with `_pre`)
- Processing metadata

### Combined Metrics

Creates: `{base_dir}/combined_metrics.csv`

Contains all individual metrics combined with additional columns:
- `step` - Step number
- `batch_id` - Original batch identifier
- `processing_time_seconds` - Time taken for batch processing
- `failed_batches` - Number of failed batches in step
- `total_batches` - Total batches in step

### Processing Summary

Creates: `{base_dir}/processing_summary.txt`

Contains:
- Total processing statistics
- Configuration used
- Success/failure summary

## Error Handling

### Common Issues

1. **Memory Errors**
   ```
   Solution: Set MEMORY_EFFICIENT = True in config.py
   ```

2. **Timeout Errors**
   ```
   Solution: Increase BATCH_TIMEOUT or STEP_TIMEOUT
   ```

3. **Missing Checkpoint Files**
   ```
   Solution: Check that checkpoint directories exist and contain valid data
   ```

4. **Process Hanging**
   ```
   Solution: Use Ctrl+C for graceful shutdown, or reduce MAX_WORKERS
   ```

### Debugging

Enable verbose logging:

```python
LOG_LEVEL = "DEBUG"
VERBOSE_LOGGING = True
```

## Performance Comparison

Typical speedup with parallel processing:

| Cores | Sequential Time | Parallel Time | Speedup |
|-------|----------------|---------------|---------|
| 1     | 100 minutes    | 100 minutes   | 1.0x    |
| 4     | 100 minutes    | 30 minutes    | 3.3x    |
| 8     | 100 minutes    | 18 minutes    | 5.6x    |
| 16    | 100 minutes    | 12 minutes    | 8.3x    |

*Note: Actual speedup depends on your specific data and CPU architecture*

## Best Practices

### 1. Resource Management

- Monitor CPU and memory usage with `htop` or `top`
- Start with fewer cores if system is shared
- Use memory-efficient mode for large datasets

### 2. Data Validation

- Check that all required arrays exist in checkpoints
- Validate data shapes before processing
- Use timeout protection for long-running batches

### 3. Error Recovery

- Enable `CONTINUE_ON_ERROR` for robustness
- Set appropriate `MAX_FAILED_BATCHES` threshold
- Save intermediate results to avoid data loss

### 4. Monitoring

- Use progress bars to track processing
- Enable verbose logging for debugging
- Check system resources during processing

## Troubleshooting

### High Memory Usage

```python
# Reduce number of workers
MAX_WORKERS = multiprocessing.cpu_count() // 2

# Enable memory optimization
MEMORY_EFFICIENT = True

# Process steps one at a time (modify code)
```

### Slow Processing

```python
# Increase workers (if you have cores available)
MAX_WORKERS = multiprocessing.cpu_count()

# Reduce timeout overhead
BATCH_TIMEOUT = 60  # Lower if batches are typically fast
```

### Process Crashes

```python
# Enable error continuation
CONTINUE_ON_ERROR = True

# Add more debugging
LOG_LEVEL = "DEBUG"
VERBOSE_LOGGING = True

# Increase timeouts
BATCH_TIMEOUT = 1800  # 30 minutes
STEP_TIMEOUT = 7200   # 2 hours
```

## Example Usage Scenarios

### Scenario 1: Fast Processing (Small Dataset)

```python
MAX_WORKERS = multiprocessing.cpu_count()
BATCH_TIMEOUT = 60
MEMORY_EFFICIENT = False
```

### Scenario 2: Memory-Constrained (Large Dataset)

```python
MAX_WORKERS = multiprocessing.cpu_count() // 2
BATCH_TIMEOUT = 600
MEMORY_EFFICIENT = True
```

### Scenario 3: Shared System (Conservative)

```python
MAX_WORKERS = 4
BATCH_TIMEOUT = 300
CONTINUE_ON_ERROR = True
```

## Advanced Features

### Custom Processing Function

You can modify `process_batch` to add custom metrics or change the analysis pipeline.

### Selective Processing

Skip already processed steps by setting:

```python
SKIP_EXISTING = True
```

### Custom Output Format

Modify the DataFrame creation section to change output format or add columns.

## Support

For issues or questions:
1. Check the log output for detailed error messages
2. Verify your checkpoint files are valid
3. Try reducing `MAX_WORKERS` if you encounter stability issues
4. Enable `VERBOSE_LOGGING` for detailed debugging information