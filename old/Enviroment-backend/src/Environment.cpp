#include "Environment.h"
#include <json/json.h> // Using jsoncpp for JSON serialization
#include <sstream>
#include <iostream>
#include <cmath>

// Constructor
Environment::Environment(const Parameters& params)
    : params(params), team1(params.team_size), team2(params.team_size), ball() 
{
    // Initialize Network on port 12345
    network = new Network(12345, 
        std::bind(&Environment::serializeState, this),
        std::bind(&Environment::handleInput, this, std::placeholders::_1)
    );
    network->start();
}

// Destructor
Environment::~Environment() {
    network->stop();
    delete network;
}

// Reset the environment
void Environment::reset() {
    team1.reset();
    team2.reset();
    ball.reset();
    states.clear();
    rewards.clear();
}

// Step the environment by one timestep
void Environment::step() {
    // 1. Update positions of all agents and the ball
    for (auto& agent : team1.getAgents()) {
        agent.updatePosition();
    }
    for (auto& agent : team2.getAgents()) {
        agent.updatePosition();
    }
    ball.updatePosition();
    ball.applyFriction();

    // 2. Check for collisions between agents
    for (auto& agent1 : team1.getAgents()) {
        for (auto& agent2 : team2.getAgents()) {
            if (checkCollision(agent1, agent2)) {
                handleCollision(agent1, agent2);
            }
        }
    }

    // 3. Check for collisions between agents and the ball
    for (auto& agent : team1.getAgents()) {
        if (checkCollision(agent, ball)) {
            handleCollision(agent, ball);
        }
    }
    for (auto& agent : team2.getAgents()) {
        if (checkCollision(agent, ball)) {
            handleCollision(agent, ball);
        }
    }

    // 4. Check for goals
    checkForGoal();

    // 5. Collect state and reward information
    collectState();
    collectRewards();
}

// Check collision between two agents
bool Environment::checkCollision(const Agent& a1, const Agent& a2) const {
    Vector2D diff = a1.getPosition() - a2.getPosition();
    float distanceSquared = diff.x * diff.x + diff.y * diff.y;
    float radiusSum = a1.getRadius() + a2.getRadius();
    return distanceSquared <= (radiusSum * radiusSum);
}

// Check collision between an agent and the ball
bool Environment::checkCollision(const Agent& agent, const Ball& ballObj) const {
    Vector2D diff = agent.getPosition() - ballObj.getPosition();
    float distanceSquared = diff.x * diff.x + diff.y * diff.y;
    float radiusSum = agent.getRadius() + ballObj.getRadius();
    return distanceSquared <= (radiusSum * radiusSum);
}

// Handle collision between two agents
void Environment::handleCollision(Agent& a1, Agent& a2) {
    // Elastic collision response
    Vector2D pos1 = a1.getPosition();
    Vector2D pos2 = a2.getPosition();
    Vector2D vel1 = a1.getMomentum();
    Vector2D vel2 = a2.getMomentum();

    Vector2D delta = pos1 - pos2;
    float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);

    // Normalize delta
    Vector2D normal = (dist > 0.0f) ? Vector2D(delta.x / dist, delta.y / dist) : Vector2D(0.0f, 0.0f);

    // Relative velocity
    Vector2D relativeVel = vel1 - vel2;

    // Velocity along the normal
    float velAlongNormal = relativeVel.x * normal.x + relativeVel.y * normal.y;

    // Do not resolve if velocities are separating
    if (velAlongNormal > 0)
        return;

    // Calculate restitution (1.0 for elastic collisions)
    float restitution = 1.0f;

    // Calculate impulse scalar
    float invMass1 = 1.0f / a1.getMass();
    float invMass2 = 1.0f / a2.getMass();
    float impulseScalar = -(1 + restitution) * velAlongNormal;
    impulseScalar /= invMass1 + invMass2;

    // Apply impulse
    Vector2D impulse = normal * impulseScalar;
    a1.setMomentum(Vector2D(a1.getMomentum().x + impulse.x * invMass1,
                           a1.getMomentum().y + impulse.y * invMass1));
    a2.setMomentum(Vector2D(a2.getMomentum().x - impulse.x * invMass2,
                           a2.getMomentum().y - impulse.y * invMass2));
}

// Handle collision between an agent and the ball
void Environment::handleCollision(Agent& agent, Ball& ballObj) {
    // Elastic collision response
    Vector2D posA = agent.getPosition();
    Vector2D posB = ballObj.getPosition();
    Vector2D velA = agent.getMomentum();
    Vector2D velB = ballObj.getMomentum();

    Vector2D delta = posA - posB;
    float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);

    // Normalize delta
    Vector2D normal = (dist > 0.0f) ? Vector2D(delta.x / dist, delta.y / dist) : Vector2D(0.0f, 0.0f);

    // Relative velocity
    Vector2D relativeVel = velA - velB;

    // Velocity along the normal
    float velAlongNormal = relativeVel.x * normal.x + relativeVel.y * normal.y;

    // Do not resolve if velocities are separating
    if (velAlongNormal > 0)
        return;

    // Calculate restitution (1.0 for elastic collisions)
    float restitution = 1.0f;

    // Calculate impulse scalar
    float invMassA = 1.0f / agent.getMass();
    float invMassB = 1.0f / ballObj.getMass();
    float impulseScalar = -(1 + restitution) * velAlongNormal;
    impulseScalar /= invMassA + invMassB;

    // Apply impulse
    Vector2D impulse = normal * impulseScalar;
    agent.setMomentum(Vector2D(agent.getMomentum().x + impulse.x * invMassA,
                              agent.getMomentum().y + impulse.y * invMassA));
    ballObj.setMomentum(Vector2D(ballObj.getMomentum().x - impulse.x * invMassB,
                                 ballObj.getMomentum().y - impulse.y * invMassB));
}

// Serialize the current state to JSON
std::string Environment::serializeState() {
    Json::Value root;

    // Serialize Team1
    for (const auto& agent : team1.getAgents()) {
        Json::Value agentJson;
        agentJson["x"] = agent.getPosition().x;
        agentJson["y"] = agent.getPosition().y;
        root["team1"].append(agentJson);
    }

    // Serialize Team2
    for (const auto& agent : team2.getAgents()) {
        Json::Value agentJson;
        agentJson["x"] = agent.getPosition().x;
        agentJson["y"] = agent.getPosition().y;
        root["team2"].append(agentJson);
    }

    // Serialize Ball
    root["ball"]["x"] = ball.getPosition().x;
    root["ball"]["y"] = ball.getPosition().y;

    // Convert JSON to string
    Json::StreamWriterBuilder writer;
    std::string output = Json::writeString(writer, root);
    return output;
}

// Handle input received from Unity
void Environment::handleInput(const std::string& input) {
    // Parse input commands, e.g., "agent_id:force_x:force_y"
    // Example input: "1:5:-3" -> Agent 1 receives force (5, -3)
    std::stringstream ss(input);
    std::string token;
    std::vector<std::string> tokens;

    while (std::getline(ss, token, ':')) {
        tokens.push_back(token);
    }

    if (tokens.size() == 3) {
        int agent_id = std::stoi(tokens[0]);
        float force_x = std::stof(tokens[1]);
        float force_y = std::stof(tokens[2]);

        // Determine which team the agent belongs to
        // Assuming agent IDs are unique across teams or handle accordingly
        // For simplicity, apply to Team1
        if (agent_id >= 0 && agent_id < team1.getAgents().size()) {
            team1.getAgents()[agent_id].applyForce(Vector2D(force_x, force_y));
        }
        else if (agent_id >= team1.getAgents().size() && 
                 agent_id < team1.getAgents().size() + team2.getAgents().size()) {
            int adjusted_id = agent_id - team1.getAgents().size();
            team2.getAgents()[adjusted_id].applyForce(Vector2D(force_x, force_y));
        }
    }
}

// Collect state information
void Environment::collectState() {
    // Collect positions of all agents and the ball
    std::vector<float> currentState;

    // Team1
    std::vector<float> team1Positions = team1.getPositions();
    currentState.insert(currentState.end(), team1Positions.begin(), team1Positions.end());

    // Team2
    std::vector<float> team2Positions = team2.getPositions();
    currentState.insert(currentState.end(), team2Positions.begin(), team2Positions.end());

    // Ball
    const Vector2D& ballPos = ball.getPosition();
    currentState.push_back(ballPos.x);
    currentState.push_back(ballPos.y);

    states.emplace_back(currentState);
}

// Collect reward information
void Environment::collectRewards() {
    // Placeholder: Implement reward logic
    // For now, assign 0 rewards
    std::vector<float> currentReward;
    currentReward.assign(params.team_size * 2 + 2, 0.0f); // team1 + team2 + ball
    rewards.emplace_back(currentReward);
}

// Check for goals
void Environment::checkForGoal() {
    // Define goal areas (e.g., left and right edges)
    const float FIELD_WIDTH = 100.0f;
    const float GOAL_THRESHOLD = 1.0f; // Distance from the edge to count as a goal

    if (ball.getPosition().x < 0.0f - GOAL_THRESHOLD) {
        // Team2 scores
        std::cout << "Team2 scored a goal!" << std::endl;
        rewards.emplace_back(generateGoalReward(2));
        reset();
    }
    if (ball.getPosition().x > FIELD_WIDTH + GOAL_THRESHOLD) {
        // Team1 scores
        std::cout << "Team1 scored a goal!" << std::endl;
        rewards.emplace_back(generateGoalReward(1));
        reset();
    }
}

// Generate goal rewards
std::vector<float> Environment::generateGoalReward(int scoring_team) {
    std::vector<float> reward(params.team_size * 2 + 2, 0.0f); // team1 + team2 + ball
    if (scoring_team == 1) {
        // Reward for team1 agents
        for (int i = 0; i < params.team_size; ++i) {
            reward[i * 2] = params.goalReward; // Example: x position reward
            reward[i * 2 + 1] = params.goalReward;
        }
    }
    else {
        // Reward for team2 agents
        for (int i = 0; i < params.team_size; ++i) {
            reward[(params.team_size + i) * 2] = params.goalReward;
            reward[(params.team_size + i) * 2 + 1] = params.goalReward;
        }
    }
    return reward;
}

// Get states
bool Environment::getStates(std::vector<std::vector<float>>& outStates) const {
    if (states.empty()) return false;
    outStates = states;
    return true;
}

// Get rewards
bool Environment::getRewards(std::vector<std::vector<float>>& outRewards) const {
    if (rewards.empty()) return false;
    outRewards = rewards;
    return true;
}

// Helper methods to get team positions
std::vector<Vector2D> Environment::getTeam1Positions() const {
    std::vector<Vector2D> positions;
    for (const auto& agent : team1.getAgents()) {
        positions.emplace_back(agent.getPosition());
    }
    return positions;
}

std::vector<Vector2D> Environment::getTeam2Positions() const {
    std::vector<Vector2D> positions;
    for (const auto& agent : team2.getAgents()) {
        positions.emplace_back(agent.getPosition());
    }
    return positions;
}
