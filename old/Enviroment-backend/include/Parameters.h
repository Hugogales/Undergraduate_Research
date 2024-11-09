#pragma once
#include "Vector2D.h"
#include <vector>

class Parameters {
public:
    Parameters();

    int team_size;
    int max_steps;

    std::vector<Vector2D> teamPositions;
    Vector2D ballPosition;
    float ballRadius;
    float ballFriction;
    float ballMass;

    float agentRadius;
    float agentMass;

    float positionReward;
    float goalReward;

    Vector2D goalsPosition;
};
