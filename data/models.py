"""
Shared Models for Sports Tournament Scheduling
Contains common data structures used across all tasks
"""

from datetime import datetime


class Match:
    """Represents a single match in the tournament schedule"""
    def __init__(self, team1, team2, date, time, venue):
        self.team1 = team1
        self.team2 = team2
        self.date = date
        self.time = time
        self.venue = venue

    def __repr__(self):
        return f"{self.team1} vs {self.team2} on {self.date.strftime('%Y-%m-%d')} at {self.time} in {self.venue}"
    
    def __eq__(self, other):
        """Check if two matches are the same"""
        if not isinstance(other, Match):
            return False
        return (self.team1 == other.team1 and 
                self.team2 == other.team2 and 
                self.date == other.date and 
                self.time == other.time and 
                self.venue == other.venue)
    
    def __hash__(self):
        """Make Match hashable for use in sets"""
        return hash((self.team1, self.team2, self.date, self.time, self.venue))

