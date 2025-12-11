"""
Test script to verify that generate_weekly_schedule() produces exactly 306 matches
"""

from ga_runner import generate_weekly_schedule, teams, venues, match_times, all_dates
from data.fitness import compute_fitness
from ga_runner import validate_schedule

def test_match_count():
    """Test that schedule has exactly 306 matches"""
    print("Testing match count...")
    print("=" * 60)
    
    # Generate schedule
    schedule = generate_weekly_schedule(teams, venues, all_dates, match_times)
    
    expected = len(teams) * (len(teams) - 1)  # 18 × 17 = 306
    actual = len(schedule)
    
    print(f"Expected matches: {expected}")
    print(f"Actual matches: {actual}")
    
    if actual == expected:
        print("✅ PASS: Schedule has exactly 306 matches")
    else:
        print(f"❌ FAIL: Expected {expected}, got {actual}")
    
    # Test validation
    print("\nTesting validation...")
    is_valid, errors = validate_schedule(schedule, teams, min_rest_days=4)
    
    if is_valid:
        print("✅ PASS: Schedule is valid")
    else:
        print(f"❌ FAIL: Schedule has {len(errors)} validation errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    # Test fitness
    print("\nTesting fitness computation...")
    fitness = compute_fitness(schedule)
    print(f"Fitness score: {fitness:.2f}")
    
    print("\n" + "=" * 60)
    return actual == expected and is_valid

if __name__ == "__main__":
    success = test_match_count()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")


