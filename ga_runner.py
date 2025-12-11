"""
Genetic Algorithm Engine for Sports Tournament Scheduling
Person 4 - GA Engine + Experiments

This module implements a complete genetic algorithm for optimizing
sports tournament schedules with selection, crossover, mutation, and elitism.

INTEGRATED WITH OTHER TASKS:
- Uses schedule_generator from Task 1
- Uses genetic_operations from Task 3
- Uses fitness from Task 2
- No code duplication!
"""

import random
import copy
import json
import csv
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any
import sys
import os

# Import shared modules - Task 1, 2, 3
try:
    from data.models import Match
    from data.fitness import compute_fitness
    from data.teams_venues_times import teams, venues, match_times
    from data.schedule_generator import generate_weekly_schedule  # Task 1
    from data.genetic_operations import (  # Task 3
        single_point_crossover,
        two_point_crossover,
        uniform_crossover,
        swap_mutation,
        change_venue_mutation,
        change_time_mutation,
        tournament_selection,
        roulette_wheel_selection,
        rank_selection
    )
except ImportError:
    # If direct import fails, add to path
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    if data_path not in sys.path:
        sys.path.insert(0, data_path)
    from models import Match
    from fitness import compute_fitness
    from teams_venues_times import teams, venues, match_times
    from schedule_generator import generate_weekly_schedule
    from genetic_operations import (
        single_point_crossover,
        two_point_crossover,
        uniform_crossover,
        swap_mutation,
        change_venue_mutation,
        change_time_mutation,
        tournament_selection,
        roulette_wheel_selection,
        rank_selection
    )

# Tournament period
start_date = datetime(2025, 5, 1)
end_date = datetime(2026, 1, 31)

# Setup all dates
all_dates = []
current_date = start_date
while current_date <= end_date:
    all_dates.append(current_date)
    current_date += timedelta(days=1)


# generate_weekly_schedule is now imported from data.schedule_generator (Task 1)


# ==================== VALIDATION FUNCTIONS ====================

def validate_schedule(schedule, teams, min_rest_days=4):
    """
    Validate schedule constraints with comprehensive checks
    
    Returns:
    --------
    tuple: (is_valid: bool, errors: List[str])
    """
    errors = []
    expected_matches = len(teams) * (len(teams) - 1)  # 18 × 17 = 306
    
    # Check total match count
    if len(schedule) != expected_matches:
        errors.append(f"Expected {expected_matches} matches, got {len(schedule)}")
    
    # Check for duplicate matches (same teams, same date)
    seen_matches = {}
    for match in schedule:
        key = (match.team1, match.team2, match.date)
        if key in seen_matches:
            errors.append(f"Duplicate match: {match}")
        seen_matches[key] = True
    
    # Check for self-play
    for match in schedule:
        if match.team1 == match.team2:
            errors.append(f"Team playing itself: {match}")
    
    # Check match count per team (should be 34: 17 opponents × 2)
    team_match_count = {team: 0 for team in teams}
    for match in schedule:
        team_match_count[match.team1] += 1
        team_match_count[match.team2] += 1
    
    expected_per_team = (len(teams) - 1) * 2  # 17 × 2 = 34
    for team in teams:
        if team_match_count.get(team, 0) != expected_per_team:
            errors.append(f"Team {team} has {team_match_count.get(team, 0)} matches, expected {expected_per_team}")
    
    # Check home/away balance for each pair
    pair_matches = {}
    for match in schedule:
        pair = tuple(sorted([match.team1, match.team2]))
        if pair not in pair_matches:
            pair_matches[pair] = {'home': 0, 'away': 0}
        
        # Determine if team1 is home or away based on pair order
        if match.team1 == pair[0]:
            pair_matches[pair]['home'] += 1
        else:
            pair_matches[pair]['away'] += 1
    
    for pair, counts in pair_matches.items():
        if counts['home'] != 1 or counts['away'] != 1:
            errors.append(f"Pair {pair} has unbalanced home/away: home={counts['home']}, away={counts['away']} (expected 1 each)")
    
    # Check rest days
    last_played = {}
    for match in schedule:
        for team in [match.team1, match.team2]:
            if team in last_played:
                delta = (match.date - last_played[team]).days
                if delta < min_rest_days:
                    errors.append(f"Rest violation: {team} played {delta} days apart (min {min_rest_days} required)")
            last_played[team] = match.date
    
    # Check venue conflicts
    venue_slots = {}
    for match in schedule:
        key = (match.date, match.time, match.venue)
        if key in venue_slots:
            errors.append(f"Venue conflict: {match}")
        venue_slots[key] = True
    
    # Check consecutive home matches (no team should have 3+ consecutive home matches)
    team_schedule = {team: [] for team in teams}
    for match in schedule:
        team_schedule[match.team1].append(('home', match.date))
        team_schedule[match.team2].append(('away', match.date))
    
    for team, matches in team_schedule.items():
        matches.sort(key=lambda x: x[1])
        consecutive_home = 0
        for match_type, date in matches:
            if match_type == 'home':
                consecutive_home += 1
                if consecutive_home >= 3:
                    errors.append(f"Team {team} has {consecutive_home} consecutive home matches")
                    break
            else:
                consecutive_home = 0
    
    return len(errors) == 0, errors


def regenerate_missing_matches(schedule, teams, venues, match_times, all_dates, min_rest_days=4):
    """
    Regenerate missing matches to reach 306 matches
    """
    expected_matches = len(teams) * (len(teams) - 1)  # 306
    current_count = len(schedule)
    
    if current_count >= expected_matches:
        return schedule
    
    # Track existing matches
    existing_pairs = set()
    for match in schedule:
        existing_pairs.add((match.team1, match.team2))
    
    # Track venue/time usage per day
    day_info = {}
    for match in schedule:
        if match.date not in day_info:
            day_info[match.date] = {'count': 0, 'venues': set(), 'times': set()}
        day_info[match.date]['count'] += 1
        day_info[match.date]['venues'].add(match.venue)
        day_info[match.date]['times'].add(match.time)
    
    # Find missing pairings
    missing_pairings = []
    for t1 in teams:
        for t2 in teams:
            if t1 != t2 and (t1, t2) not in existing_pairs:
                missing_pairings.append((t1, t2))
    
    random.shuffle(missing_pairings)
    
    # Try to schedule missing matches
    for home, away in missing_pairings:
        if len(schedule) >= expected_matches:
            break
        
        # Find available slot
        for date in all_dates:
            if len(schedule) >= expected_matches:
                break
            
            if date not in day_info:
                day_info[date] = {'count': 0, 'venues': set(), 'times': set()}
            
            if day_info[date]['count'] < 2:  # Max 2 matches per day
                available_times = [t for t in match_times if t not in day_info[date]['times']]
                available_venues = [v for v in venues if v not in day_info[date]['venues']]
                
                if available_times and available_venues:
                    time = random.choice(available_times)
                    venue = random.choice(available_venues)
                    match = Match(home, away, date, time, venue)
                    schedule.append(match)
                    day_info[date]['count'] += 1
                    day_info[date]['venues'].add(venue)
                    day_info[date]['times'].add(time)
                    break
    
    return schedule


def repair_schedule(schedule, teams, venues, match_times, all_dates, min_rest_days=4, max_retries=5):
    """
    Repair invalid schedule by removing duplicates and fixing conflicts
    Includes retry mechanism and match regeneration
    
    Returns:
    --------
    List[Match]: Repaired schedule
    """
    best_repaired = schedule
    best_match_count = len(schedule)
    expected_matches = len(teams) * (len(teams) - 1)  # 306
    
    for retry in range(max_retries):
        # Remove duplicates and self-play matches
        seen = set()
        valid_schedule = []
        for match in schedule:
            key = (match.team1, match.team2, match.date)
            if key not in seen and match.team1 != match.team2:
                seen.add(key)
                valid_schedule.append(copy.deepcopy(match))
        
        # Remove venue conflicts (keep first occurrence)
        venue_slots = {}
        repaired = []
        for match in valid_schedule:
            key = (match.date, match.time, match.venue)
            if key not in venue_slots:
                venue_slots[key] = True
                repaired.append(match)
            else:
                # Try to fix by changing venue or time
                available_venues = [v for v in venues if v != match.venue]
                available_times = [t for t in match_times if t != match.time]
                
                fixed = False
                # Try changing venue first
                for new_venue in available_venues:
                    new_key = (match.date, match.time, new_venue)
                    if new_key not in venue_slots:
                        match.venue = new_venue
                        venue_slots[new_key] = True
                        repaired.append(match)
                        fixed = True
                        break
                
                # If venue change didn't work, try changing time
                if not fixed:
                    for new_time in available_times:
                        new_key = (match.date, new_time, match.venue)
                        if new_key not in venue_slots:
                            match.time = new_time
                            venue_slots[new_key] = True
                            repaired.append(match)
                            fixed = True
                            break
                
                # If still not fixed, try changing both venue and time
                if not fixed:
                    for new_venue in available_venues:
                        for new_time in available_times:
                            new_key = (match.date, new_time, new_venue)
                            if new_key not in venue_slots:
                                match.venue = new_venue
                                match.time = new_time
                                venue_slots[new_key] = True
                                repaired.append(match)
                                fixed = True
                                break
                        if fixed:
                            break
        
        # If we lost too many matches, try to regenerate
        if len(repaired) < expected_matches * 0.9:  # Lost more than 10%
            repaired = regenerate_missing_matches(repaired, teams, venues, match_times, all_dates, min_rest_days)
        
        # Keep track of best repair attempt
        if len(repaired) > best_match_count:
            best_repaired = repaired
            best_match_count = len(repaired)
        
        # If we have enough matches and it's valid, return
        if len(repaired) >= expected_matches * 0.95:  # At least 95% of expected
            repaired.sort(key=lambda m: m.date)
            return repaired
        
        # For next retry, use the repaired schedule as base
        schedule = repaired
    
    # Return best attempt
    best_repaired.sort(key=lambda m: m.date)
    return best_repaired


# ==================== SELECTION, CROSSOVER, MUTATION ====================
# All imported from data.genetic_operations (Task 3) - no duplication!

def apply_mutation(schedule, venues, match_times, mutation_rate=0.1):
    """
    Apply mutation based on mutation_rate
    Uses mutation functions from Task 3 (genetic_operations)
    
    The mutation_rate determines the probability that a mutation will occur.
    If mutation occurs, a random mutation type is chosen.
    """
    # Choose mutation type (can be weighted)
    mutation_type = random.choice(['swap', 'venue', 'time'])
    
    if mutation_type == 'swap':
        return swap_mutation(schedule, mutation_rate)
    elif mutation_type == 'venue':
        return change_venue_mutation(schedule, venues, mutation_rate)
    elif mutation_type == 'time':
        return change_time_mutation(schedule, match_times, mutation_rate)
    
    return copy.deepcopy(schedule)


# ==================== GENETIC ALGORITHM MAIN LOOP ====================

def run_genetic_algorithm(
    population_size=50,
    generations=100,
    mutation_rate=0.1,
    crossover_rate=0.8,
    elitism_count=2,
    selection_method='tournament',
    tournament_size=3,
    crossover_method='single_point',
    verbose=True
):
    """
    Run the complete genetic algorithm
    
    Parameters:
    -----------
    population_size : int
        Number of individuals in the population
    generations : int
        Number of generations to evolve
    mutation_rate : float
        Probability of mutation (0.0 to 1.0)
    crossover_rate : float
        Probability of crossover (0.0 to 1.0)
    elitism_count : int
        Number of best individuals to preserve
    selection_method : str
        'tournament', 'roulette', or 'rank'
    tournament_size : int
        Size of tournament for tournament selection
    crossover_method : str
        'single_point', 'two_point', or 'uniform'
    verbose : bool
        Print progress information
    
    Returns:
    --------
    best_schedule : List[Match]
        Best schedule found
    best_fitness : float
        Fitness of best schedule
    history : Dict
        Evolution history with fitness statistics
    """
    
    # Initialize population
    if verbose:
        print(f"Initializing population of {population_size} schedules...")
    
    population = [generate_weekly_schedule(teams, venues, all_dates, match_times, min_rest_days=4, start_date=start_date) 
                  for _ in range(population_size)]
    
    # Evaluate initial population
    fitness_scores = [compute_fitness(schedule) for schedule in population]
    
    # Track history
    history = {
        'best_fitness': [],
        'avg_fitness': [],
        'worst_fitness': [],
        'generation': []
    }
    
    best_fitness = max(fitness_scores)
    best_schedule = copy.deepcopy(population[fitness_scores.index(best_fitness)])
    
    if verbose:
        print(f"Initial best fitness: {best_fitness:.2f}")
    
    # Evolution loop
    for generation in range(generations):
        # Record statistics
        history['generation'].append(generation)
        history['best_fitness'].append(best_fitness)
        history['avg_fitness'].append(sum(fitness_scores) / len(fitness_scores))
        history['worst_fitness'].append(min(fitness_scores))
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individuals
        elite_indices = sorted(range(len(fitness_scores)), 
                               key=lambda i: fitness_scores[i], 
                               reverse=True)[:elitism_count]
        for idx in elite_indices:
            new_population.append(copy.deepcopy(population[idx]))
        
        # Generate remaining population
        while len(new_population) < population_size:
            # Selection
            if selection_method == 'tournament':
                parent1 = tournament_selection(population, fitness_scores, tournament_size)
                parent2 = tournament_selection(population, fitness_scores, tournament_size)
            elif selection_method == 'roulette':
                parent1 = roulette_wheel_selection(population, fitness_scores)
                parent2 = roulette_wheel_selection(population, fitness_scores)
            elif selection_method == 'rank':
                parent1 = rank_selection(population, fitness_scores)
                parent2 = rank_selection(population, fitness_scores)
            else:
                parent1 = tournament_selection(population, fitness_scores, tournament_size)
                parent2 = tournament_selection(population, fitness_scores, tournament_size)
            
            # Crossover
            if random.random() < crossover_rate:
                if crossover_method == 'two_point':
                    child = two_point_crossover(parent1, parent2)
                elif crossover_method == 'uniform':
                    child = uniform_crossover(parent1, parent2)
                else:
                    child = single_point_crossover(parent1, parent2)
            else:
                child = copy.deepcopy(parent1 if random.random() < 0.5 else parent2)
            
            # Mutation
            child = apply_mutation(child, venues, match_times, mutation_rate)
            
            # Validation and repair
            is_valid, errors = validate_schedule(child, teams, min_rest_days=4)
            if not is_valid:
                child = repair_schedule(child, teams, venues, match_times, all_dates, min_rest_days=4)
                # If still invalid or lost too many matches, use parent
                if len(child) < len(parent1) * 0.8:  # Lost too many matches
                    child = copy.deepcopy(parent1)
            
            new_population.append(child)
        
        # Update population
        population = new_population
        fitness_scores = [compute_fitness(schedule) for schedule in population]
        
        # Update best
        current_best = max(fitness_scores)
        if current_best > best_fitness:
            best_fitness = current_best
            best_schedule = copy.deepcopy(population[fitness_scores.index(current_best)])
        
        if verbose and (generation + 1) % 10 == 0:
            print(f"Generation {generation + 1}/{generations}: "
                  f"Best={best_fitness:.2f}, "
                  f"Avg={history['avg_fitness'][-1]:.2f}, "
                  f"Worst={history['worst_fitness'][-1]:.2f}")
    
    if verbose:
        print(f"\nFinal best fitness: {best_fitness:.2f}")
    
    return best_schedule, best_fitness, history


# ==================== EXPERIMENT RUNNER ====================

def run_experiments(experiment_configs, num_runs=3):
    """
    Run multiple experiments with different parameter configurations
    
    Parameters:
    -----------
    experiment_configs : List[Dict]
        List of parameter configurations to test
    num_runs : int
        Number of runs per configuration for averaging
    
    Returns:
    --------
    results : List[Dict]
        Results for each experiment configuration
    """
    results = []
    
    print("=" * 80)
    print("GENETIC ALGORITHM EXPERIMENTS")
    print("=" * 80)
    
    for exp_idx, config in enumerate(experiment_configs):
        print(f"\n{'='*80}")
        print(f"Experiment {exp_idx + 1}/{len(experiment_configs)}")
        print(f"Configuration: {config}")
        print(f"{'='*80}\n")
        
        run_results = []
        
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}...", end=" ")
            
            best_schedule, best_fitness, history = run_genetic_algorithm(
                population_size=config.get('population_size', 50),
                generations=config.get('generations', 100),
                mutation_rate=config.get('mutation_rate', 0.1),
                crossover_rate=config.get('crossover_rate', 0.8),
                elitism_count=config.get('elitism_count', 2),
                selection_method=config.get('selection_method', 'tournament'),
                tournament_size=config.get('tournament_size', 3),
                crossover_method=config.get('crossover_method', 'single_point'),
                verbose=False
            )
            
            run_results.append({
                'best_fitness': best_fitness,
                'final_avg_fitness': history['avg_fitness'][-1],
                'final_worst_fitness': history['worst_fitness'][-1],
                'history': history
            })
            
            print(f"Best fitness: {best_fitness:.2f}")
        
        # Calculate statistics
        best_fitnesses = [r['best_fitness'] for r in run_results]
        avg_fitnesses = [r['final_avg_fitness'] for r in run_results]
        
        result = {
            'config': config,
            'runs': run_results,
            'statistics': {
                'best_fitness_mean': sum(best_fitnesses) / len(best_fitnesses),
                'best_fitness_std': (sum((x - sum(best_fitnesses)/len(best_fitnesses))**2 
                                     for x in best_fitnesses) / len(best_fitnesses))**0.5,
                'best_fitness_max': max(best_fitnesses),
                'best_fitness_min': min(best_fitnesses),
                'avg_fitness_mean': sum(avg_fitnesses) / len(avg_fitnesses),
            }
        }
        
        results.append(result)
        
        print(f"\nStatistics:")
        print(f"  Best Fitness - Mean: {result['statistics']['best_fitness_mean']:.2f}, "
              f"Std: {result['statistics']['best_fitness_std']:.2f}, "
              f"Max: {result['statistics']['best_fitness_max']:.2f}, "
              f"Min: {result['statistics']['best_fitness_min']:.2f}")
    
    return results


def save_results(results, filename='experiment_results.json'):
    """Save experiment results to JSON file"""
    # Convert results to JSON-serializable format
    json_results = []
    for result in results:
        json_result = {
            'config': result['config'],
            'statistics': result['statistics'],
            'runs': []
        }
        # Only save final statistics for each run (not full history)
        for run in result['runs']:
            json_result['runs'].append({
                'best_fitness': run['best_fitness'],
                'final_avg_fitness': run['final_avg_fitness'],
                'final_worst_fitness': run['final_worst_fitness']
            })
        json_results.append(json_result)
    
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {filename}")


def save_results_csv(results, filename='experiment_results.csv'):
    """Save experiment results summary to CSV"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Experiment', 'Population Size', 'Generations', 'Mutation Rate',
            'Crossover Rate', 'Elitism Count', 'Selection Method',
            'Tournament Size', 'Crossover Method',
            'Best Fitness Mean', 'Best Fitness Std', 'Best Fitness Max', 'Best Fitness Min',
            'Avg Fitness Mean'
        ])
        
        for idx, result in enumerate(results):
            config = result['config']
            stats = result['statistics']
            writer.writerow([
                idx + 1,
                config.get('population_size', 'N/A'),
                config.get('generations', 'N/A'),
                config.get('mutation_rate', 'N/A'),
                config.get('crossover_rate', 'N/A'),
                config.get('elitism_count', 'N/A'),
                config.get('selection_method', 'N/A'),
                config.get('tournament_size', 'N/A'),
                config.get('crossover_method', 'N/A'),
                f"{stats['best_fitness_mean']:.2f}",
                f"{stats['best_fitness_std']:.2f}",
                f"{stats['best_fitness_max']:.2f}",
                f"{stats['best_fitness_min']:.2f}",
                f"{stats['avg_fitness_mean']:.2f}"
            ])
    
    print(f"Results summary saved to {filename}")


def compare_results(results):
    """Print a comparison of all experiment results"""
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS COMPARISON")
    print("=" * 80)
    
    # Sort by best fitness mean
    sorted_results = sorted(results, 
                           key=lambda x: x['statistics']['best_fitness_mean'], 
                           reverse=True)
    
    print(f"\n{'Exp':<5} {'Pop':<5} {'Gen':<5} {'Mut':<6} {'Sel':<12} {'Best Mean':<12} {'Best Max':<12} {'Best Min':<12}")
    print("-" * 80)
    
    for idx, result in enumerate(sorted_results):
        config = result['config']
        stats = result['statistics']
        print(f"{idx+1:<5} "
              f"{config.get('population_size', 'N/A'):<5} "
              f"{config.get('generations', 'N/A'):<5} "
              f"{config.get('mutation_rate', 'N/A'):<6.2f} "
              f"{config.get('selection_method', 'N/A'):<12} "
              f"{stats['best_fitness_mean']:<12.2f} "
              f"{stats['best_fitness_max']:<12.2f} "
              f"{stats['best_fitness_min']:<12.2f}")
    
    print("\n" + "=" * 80)
    print(f"Best Configuration: Experiment {sorted_results[0]['config']}")
    print(f"Best Fitness Achieved: {sorted_results[0]['statistics']['best_fitness_max']:.2f}")
    print("=" * 80)


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Define experiment configurations (reduced from 11 to 6 experiments)
    experiments = [
        # Baseline
        {
            'name': 'Baseline',
            'population_size': 30,
            'generations': 50,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elitism_count': 2,
            'selection_method': 'tournament',
            'tournament_size': 3,
            'crossover_method': 'single_point'
        },
        # Test population sizes
        {
            'name': 'Large Population',
            'population_size': 100,
            'generations': 50,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elitism_count': 2,
            'selection_method': 'tournament',
            'tournament_size': 3,
            'crossover_method': 'single_point'
        },
        # Test mutation rates
        {
            'name': 'High Mutation',
            'population_size': 50,
            'generations': 50,
            'mutation_rate': 0.3,
            'crossover_rate': 0.8,
            'elitism_count': 2,
            'selection_method': 'tournament',
            'tournament_size': 3,
            'crossover_method': 'single_point'
        },
        # Test selection methods
        {
            'name': 'Roulette Selection',
            'population_size': 50,
            'generations': 50,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elitism_count': 2,
            'selection_method': 'roulette',
            'tournament_size': 3,
            'crossover_method': 'single_point'
        },
        # Test crossover methods
        {
            'name': 'Two-Point Crossover',
            'population_size': 50,
            'generations': 50,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elitism_count': 2,
            'selection_method': 'tournament',
            'tournament_size': 3,
            'crossover_method': 'two_point'
        },
        # Best combination
        {
            'name': 'Best Combination',
            'population_size': 100,
            'generations': 100,
            'mutation_rate': 0.15,
            'crossover_rate': 0.8,
            'elitism_count': 5,
            'selection_method': 'tournament',
            'tournament_size': 5,
            'crossover_method': 'two_point'
        }
    ]
    
    # Run experiments (with fewer runs for faster execution)
    print("Starting experiments...")
    results = run_experiments(experiments, num_runs=3)
    
    # Compare and save results
    compare_results(results)
    save_results(results, 'experiment_results.json')
    save_results_csv(results, 'experiment_results.csv')
    
    print("\nExperiments completed!")

