#include "Parameters.h"

// Constructor with default values
Parameters::Parameters() 
    : team_size(3), max_steps(1000), 
      ballRadius(3.0f), ballFriction(1.0f), ballMass(1.0f),
      agentRadius(5.0f), agentMass(70.0f), 
      positionReward(0.0f), goalReward(100.0f),
      goalsPosition(0.0f, 0.0f) {}
