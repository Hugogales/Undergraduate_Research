#pragma once
#include <vector>
#include "Team.h"
#include "Ball.h"
#include "Parameters.h"
#include "Network.h"

class Environment {
public:
    Environment(const Parameters& params);
    ~Environment();

    void reset();
    void step();
    bool checkCollision(const Agent& a1, const Agent& a2) const;
    bool checkCollision(const Agent& agent, const Ball& ballObj) const;
    bool getStates(std::vector<std::vector<float>>& outStates) const;
    bool getRewards(std::vector<std::vector<float>>& outRewards) const;

private:
    Parameters params;
    Team team1;
    Team team2;
    Ball ball;
    std::vector<std::vector<float>> states;
    std::vector<std::vector<float>> rewards;

    // Network
    Network* network;

    // Callbacks for Network
    std::string serializeState();
    void handleInput(const std::string& input);

    void handleCollision(Agent& a1, Agent& a2);
    void handleCollision(Agent& agent, Ball& ballObj);
    void collectState();
    void collectRewards();

    // Private helper methods
    void checkForGoal();
    std::vector<Vector2D> getTeam1Positions() const;
    std::vector<Vector2D> getTeam2Positions() const;
};
