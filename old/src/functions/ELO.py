
import trueskill

class ELO:
    def __init__(self, mean: int = 1000, std: int = 100):
        self.mean = mean
        self.std = std
        self.ts = trueskill.TrueSkill(mu=mean, sigma=std, draw_probability=0.35)
    
    def create_rating(self, mean: int, std: int = 100):
        return self.ts.create_rating(mean, std)


    def calculate(self, rating1, rating2, score1, score2):
        """
        Rating update using TrueSkill directly to handle uncertainty.
        """
        # Determine rank (lower rank = better performance)
        if score1 > score2:
            ranks = [0, 1]  # Team1 won
        elif score1 < score2:
            ranks = [1, 0]  # Team2 won
        else:
            ranks = [0, 0]  # Draw
        
        # Format ratings for TrueSkill's rate method
        # TrueSkill expects a list of lists, where each inner list is a team
        teams = [[rating1], [rating2]]
        
        # Update ratings using TrueSkill
        new_ratings = self.ts.rate(teams, ranks)
        
        # Extract the updated ratings
        updated_rating1 = new_ratings[0][0]
        updated_rating2 = new_ratings[1][0]
        
        return updated_rating1, updated_rating2
    
    def init_rating(self):
        return self.ts.create_rating()