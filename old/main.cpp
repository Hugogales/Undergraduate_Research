#include "Environment.h"
#include "Parameters.h"
#include <iostream>

int main() {
    Parameters params;
    params.team_size = 3;
    params.max_steps = 1000;
    params.ballRadius = 3.0f;
    params.ballFriction = 1.0f;
    params.ballMass = 1.0f;
    params.agentRadius = 5.0f;
    params.agentMass = 70.0f;
    params.positionReward = 0.0f;
    params.goalReward = 100.0f;
    params.goalsPosition = Vector2D(0.0f, 0.0f);

    Environment env(params);

    env.reset();

    // Simulation loop
    for (int step = 0; step < params.max_steps; ++step) {
        env.step();
    }

    return 0;
}
