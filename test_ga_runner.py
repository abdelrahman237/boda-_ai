"""
Quick test script for ga_runner.py
Tests basic functionality without running full experiments
"""

from datetime import datetime, timedelta
from ga_runner import (
    generate_weekly_schedule, 
    run_genetic_algorithm,
    teams, venues, match_times, all_dates
)
from data.fitness import compute_fitness

def test_basic_functionality():
    """Test basic GA functionality"""
    print("Testing basic GA functionality...")
    print("=" * 60)
    
    # Test 1: Generate initial schedule
    print("\n1. Testing schedule generation...")
    schedule = generate_weekly_schedule(teams, venues, all_dates, match_times)
    print(f"   Generated schedule with {len(schedule)} matches")
    
    # Test 2: Compute fitness
    print("\n2. Testing fitness computation...")
    fitness = compute_fitness(schedule)
    print(f"   Fitness score: {fitness:.2f}")
    
    # Test 3: Run short GA
    print("\n3. Testing GA with small parameters...")
    best_schedule, best_fitness, history = run_genetic_algorithm(
        population_size=10,
        generations=5,
        mutation_rate=0.1,
        verbose=True
    )
    print(f"   Final best fitness: {best_fitness:.2f}")
    print(f"   Best schedule has {len(best_schedule)} matches")
    
    print("\n" + "=" * 60)
    print("All tests passed!")

if __name__ == "__main__":
    test_basic_functionality()

