#include "Team.h"

// Constructor
Team::Team(int team_size) 
    : goal(0.0f, 0.0f) { // Initialize Goal with default position
    for (int i = 0; i < team_size; ++i) {
        agents.emplace_back(i);
    }
}

// Getters
std::vector<Agent>& Team::getAgents() { 
    return agents; 
}

// Get positions of all agents as a flat vector (x1, y1, x2, y2, ...)
std::vector<float> Team::getPositions() const {
    std::vector<float> positions;
    positions.reserve(agents.size() * 2);
    for (const auto& agent : agents) {
        const Vector2D& pos = agent.getPosition();
        positions.push_back(pos.x);
        positions.push_back(pos.y);
    }
    return positions;
}

// Reset all agents
void Team::reset() {
    for (auto& agent : agents) {
        agent.reset();
    }
}

// Getters for Goal
Goal& Team::getGoal() { 
    return goal; 
}

const Goal& Team::getGoal() const { 
    return goal; 
}

// Apply forces to agents
void Team::applyForces(const std::vector<Vector2D>& forces) {
    for (size_t i = 0; i < agents.size() && i < forces.size(); ++i) {
        agents[i].applyForce(forces[i]);
    }
}
