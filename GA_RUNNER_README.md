# Genetic Algorithm Runner - Person 4

## Overview
This module implements a complete genetic algorithm engine for optimizing sports tournament schedules. It includes selection, crossover, mutation, and elitism mechanisms, along with comprehensive experiment capabilities.

## Files
- **ga_runner.py**: Main GA engine with full implementation
- **test_ga_runner.py**: Quick test script for basic functionality
- **experiment_results.json**: JSON output of experiment results (generated after running)
- **experiment_results.csv**: CSV summary of experiment results (generated after running)

## Features

### 1. Selection Methods
- **Tournament Selection**: Selects best individual from random tournament
- **Roulette Wheel Selection**: Probability proportional to fitness
- **Rank Selection**: Selection based on rank rather than absolute fitness

### 2. Crossover Methods
- **Single-Point Crossover**: Splits parents at one point
- **Two-Point Crossover**: Splits parents at two points

### 3. Mutation Methods
- **Swap Mutation**: Exchanges two random matches
- **Change Venue Mutation**: Changes venue of a random match
- **Change Time Mutation**: Changes time of a random match

### 4. Elitism
- Preserves best individuals across generations
- Configurable elitism count

### 5. Experiment Framework
- Test different parameter combinations
- Multiple runs per configuration for statistical validity
- Automatic result recording and comparison

## Usage

### Basic Usage
```python
from ga_runner import run_genetic_algorithm

# Run GA with default parameters
best_schedule, best_fitness, history = run_genetic_algorithm(
    population_size=50,
    generations=100,
    mutation_rate=0.1,
    crossover_rate=0.8,
    elitism_count=2,
    selection_method='tournament',
    tournament_size=3,
    crossover_method='single_point',
    verbose=True
)
```

### Running Experiments
```python
from ga_runner import run_experiments, compare_results, save_results

# Define experiment configurations
experiments = [
    {
        'population_size': 50,
        'generations': 100,
        'mutation_rate': 0.1,
        'selection_method': 'tournament'
    },
    # ... more configurations
]

# Run experiments
results = run_experiments(experiments, num_runs=3)

# Compare and save results
compare_results(results)
save_results(results, 'experiment_results.json')
```

### Command Line
Simply run the script:
```bash
python ga_runner.py
```

This will run a comprehensive set of experiments testing:
- Different population sizes (20, 30, 50, 100)
- Different mutation rates (0.05, 0.1, 0.3)
- Different generation counts (25, 50, 100)
- Different selection methods (tournament, roulette, rank)
- Different crossover methods (single-point, two-point)
- Best combination configuration

## Parameters

### Genetic Algorithm Parameters
- **population_size**: Number of individuals in population (default: 50)
- **generations**: Number of generations to evolve (default: 100)
- **mutation_rate**: Probability of mutation 0.0-1.0 (default: 0.1)
- **crossover_rate**: Probability of crossover 0.0-1.0 (default: 0.8)
- **elitism_count**: Number of best individuals to preserve (default: 2)

### Selection Parameters
- **selection_method**: 'tournament', 'roulette', or 'rank' (default: 'tournament')
- **tournament_size**: Size of tournament for tournament selection (default: 3)

### Crossover Parameters
- **crossover_method**: 'single_point' or 'two_point' (default: 'single_point')

## Output

### Console Output
- Progress updates every 10 generations
- Final best fitness score
- Experiment comparison table

### Files Generated
1. **experiment_results.json**: Complete results with full history
2. **experiment_results.csv**: Summary table for easy analysis

### Result Structure
```json
{
  "config": {
    "population_size": 50,
    "generations": 100,
    ...
  },
  "statistics": {
    "best_fitness_mean": 85.23,
    "best_fitness_std": 2.15,
    "best_fitness_max": 87.50,
    "best_fitness_min": 82.10,
    "avg_fitness_mean": 78.45
  },
  "runs": [...]
}
```

## Testing

Run the test script to verify basic functionality:
```bash
python test_ga_runner.py
```

## Performance Notes

- Larger populations and more generations = better results but longer runtime
- Mutation rate around 0.1-0.15 typically works well
- Tournament selection with size 3-5 is often effective
- Elitism helps preserve good solutions

## Integration

The GA runner integrates with existing project components:
- Uses `data/teams_venues_times.py` for team/venue/time data
- Uses `Match` class from schedule model
- Uses fitness function from fitness_function notebook
- Compatible with existing crossover/mutation operations

