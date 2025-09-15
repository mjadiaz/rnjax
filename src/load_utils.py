# Utility functions for loading saved attack data
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Set


def validate_base_dir(base_dir: Union[str, Path]) -> Path:
    """
    Validate that the base directory exists and is a directory.

    Args:
        base_dir: Path to the base directory

    Returns:
        Path object for the base directory

    Raises:
        ValueError: If the base directory doesn't exist or isn't a directory
    """
    base_path = Path(base_dir).resolve()
    if not base_path.exists():
        raise ValueError(f"Base directory does not exist: {base_path}")
    if not base_path.is_dir():
        raise ValueError(f"Path is not a directory: {base_path}")
    return base_path


def list_available_steps(base_dir: Union[str, Path]) -> List[int]:
    """
    List all available step directories in the base directory.

    Args:
        base_dir: Path to the base directory

    Returns:
        List of step indices (as integers) sorted in ascending order
    """
    base_path = validate_base_dir(base_dir)
    steps = []

    for item in base_path.iterdir():
        if item.is_dir() and item.name.isdigit():
            steps.append(int(item.name))

    return sorted(steps)


def load_metadata(base_dir: Union[str, Path], step: int) -> Dict:
    """
    Load metadata for a specific step.

    Args:
        base_dir: Path to the base directory
        step: Step index to load metadata from

    Returns:
        Dictionary containing the metadata

    Raises:
        ValueError: If the step directory or metadata file doesn't exist
    """
    base_path = validate_base_dir(base_dir)
    step_dir = base_path / str(step)

    if not step_dir.exists() or not step_dir.is_dir():
        raise ValueError(f"Step directory does not exist: {step_dir}")

    metadata_path = step_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"Metadata file does not exist: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return metadata


def list_available_arrays(base_dir: Union[str, Path], step: int) -> List[str]:
    """
    List all available arrays in a specific step directory.

    Args:
        base_dir: Path to the base directory
        step: Step index to list arrays from

    Returns:
        List of array names (without the .npy extension)

    Raises:
        ValueError: If the step directory doesn't exist
    """
    base_path = validate_base_dir(base_dir)
    step_dir = base_path / str(step)

    if not step_dir.exists() or not step_dir.is_dir():
        raise ValueError(f"Step directory does not exist: {step_dir}")

    array_names = []
    for item in step_dir.iterdir():
        if item.is_file() and item.suffix == '.npy':
            array_names.append(item.stem)

    return sorted(array_names)


def load_array(base_dir: Union[str, Path], step: int, array_name: str) -> np.ndarray:
    """
    Load a specific array from a step directory.

    Args:
        base_dir: Path to the base directory
        step: Step index to load the array from
        array_name: Name of the array to load (without .npy extension)

    Returns:
        NumPy array containing the loaded data

    Raises:
        ValueError: If the step directory or array file doesn't exist
    """
    base_path = validate_base_dir(base_dir)
    step_dir = base_path / str(step)

    if not step_dir.exists() or not step_dir.is_dir():
        raise ValueError(f"Step directory does not exist: {step_dir}")

    array_path = step_dir / f"{array_name}.npy"
    if not array_path.exists():
        raise ValueError(f"Array file does not exist: {array_path}")

    return np.load(array_path)


def load_step_data(base_dir: Union[str, Path], step: int,
                  arrays: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Load metadata and arrays from a specific step.

    Args:
        base_dir: Path to the base directory
        step: Step index to load data from
        arrays: List of array names to load. If None, load all available arrays.

    Returns:
        Dictionary containing metadata and loaded arrays

    Raises:
        ValueError: If the step directory, metadata file, or any requested array doesn't exist
    """
    base_path = validate_base_dir(base_dir)
    step_dir = base_path / str(step)

    if not step_dir.exists() or not step_dir.is_dir():
        raise ValueError(f"Step directory does not exist: {step_dir}")

    # Load metadata
    metadata = load_metadata(base_path, step)

    # Determine which arrays to load
    if arrays is None:
        arrays = list_available_arrays(base_path, step)

    # Load requested arrays
    loaded_arrays = {}
    for array_name in arrays:
        loaded_arrays[array_name] = load_array(base_path, step, array_name)

    # Combine metadata and arrays
    result = {
        "metadata": metadata,
        "arrays": loaded_arrays
    }

    return result


def load_multi_step_data(base_dir: Union[str, Path], steps: List[int],
                        arrays: Optional[List[str]] = None) -> Dict[int, Dict[str, Any]]:
    """
    Load data from multiple steps.

    Args:
        base_dir: Path to the base directory
        steps: List of step indices to load data from
        arrays: List of array names to load. If None, load all available arrays.

    Returns:
        Dictionary mapping step indices to their respective data

    Raises:
        ValueError: If any step directory, metadata file, or requested array doesn't exist
    """
    result = {}
    for step in steps:
        result[step] = load_step_data(base_dir, step, arrays)

    return result


def find_common_arrays(base_dir: Union[str, Path], steps: List[int]) -> Set[str]:
    """
    Find array names that are common across multiple steps.

    Args:
        base_dir: Path to the base directory
        steps: List of step indices to check

    Returns:
        Set of array names that exist in all specified steps

    Raises:
        ValueError: If any step directory doesn't exist
    """
    if not steps:
        return set()

    # Get arrays from the first step
    first_step_arrays = set(list_available_arrays(base_dir, steps[0]))

    # Find intersection with arrays from all other steps
    common_arrays = first_step_arrays
    for step in steps[1:]:
        step_arrays = set(list_available_arrays(base_dir, step))
        common_arrays &= step_arrays

    return common_arrays


def load_experiment_summary(base_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Load summary information about the experiment.

    Args:
        base_dir: Path to the base directory

    Returns:
        Dictionary containing experiment summary information
    """
    base_path = validate_base_dir(base_dir)
    steps = list_available_steps(base_path)

    if not steps:
        return {"steps": [], "metadata": {}}

    # Load metadata from the first step as a reference
    first_step_metadata = load_metadata(base_path, steps[0])

    # Get experiment type and parameters
    experiment_type = first_step_metadata.get('experiment_type', 'unknown')
    parameters = first_step_metadata.get('parameters', {})

    # Check if all steps have the same parameters
    parameters_consistent = True
    for step in steps[1:]:
        step_metadata = load_metadata(base_path, step)
        step_parameters = step_metadata.get('parameters', {})
        if step_parameters != parameters:
            parameters_consistent = False
            break

    # Get common arrays across all steps
    common_arrays = find_common_arrays(base_path, steps)

    return {
        "steps": steps,
        "experiment_type": experiment_type,
        "parameters": parameters,
        "parameters_consistent": parameters_consistent,
        "common_arrays": list(common_arrays),
        "first_timestamp": first_step_metadata.get('timestamp', 'unknown'),
        "num_steps": len(steps)
    }


def print_experiment_info(base_dir: Union[str, Path]) -> None:
    """
    Print a summary of the experiment information.

    Args:
        base_dir: Path to the base directory
    """
    try:
        summary = load_experiment_summary(base_dir)

        print(f"Experiment Summary for: {base_dir}")
        print("-" * 50)
        print(f"Experiment Type: {summary['experiment_type']}")
        print(f"Number of Steps: {summary['num_steps']}")
        print(f"First Timestamp: {summary['first_timestamp']}")
        print(f"Parameters Consistent: {summary['parameters_consistent']}")

        print("\nParameters:")
        for key, value in summary['parameters'].items():
            print(f"  {key}: {value}")

        print("\nCommon Arrays:")
        for array_name in summary['common_arrays']:
            print(f"  - {array_name}")

        print(f"\nAvailable Steps: {summary['steps']}")

    except ValueError as e:
        print(f"Error: {e}")
