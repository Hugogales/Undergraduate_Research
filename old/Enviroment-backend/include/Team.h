#pragma once
#include <vector>
#include "Agent.h"
#include "Goal.h"

class Team {
public:
    Team(int team_size);

    std::vector<Agent>& getAgents();
    std::vector<float> getPositions() const;
    void reset();

    Goal& getGoal();
    const Goal& getGoal() const;

    // Method to update agents with forces (for future CUDA integration)
    void applyForces(const std::vector<Vector2D>& forces);

private:
    std::vector<Agent> agents;
    Goal goal;
};
