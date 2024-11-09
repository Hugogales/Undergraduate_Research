#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Environment.h"
#include "Parameters.h"

namespace py = pybind11;

// Define the GameState structure to match the JSON format sent from C++
struct AgentState {
    float x;
    float y;
};

struct BallState {
    float x;
    float y;
};

struct GameState {
    std::vector<AgentState> team1;
    std::vector<AgentState> team2;
    BallState ball;
};

PYBIND11_MODULE(multi_agent_env, m) {
    py::class_<Vector2D>(m, "Vector2D")
        .def(py::init<float, float>(), py::arg("x") = 0.0f, py::arg("y") = 0.0f)
        .def_readwrite("x", &Vector2D::x)
        .def_readwrite("y", &Vector2D::y)
        .def("toVector", &Vector2D::toVector);

    py::class_<Parameters>(m, "Parameters")
        .def(py::init<>())
        .def_readwrite("team_size", &Parameters::team_size)
        .def_readwrite("max_steps", &Parameters::max_steps)
        .def_readwrite("teamPositions", &Parameters::teamPositions)
        .def_readwrite("ballPosition", &Parameters::ballPosition)
        .def_readwrite("ballRadius", &Parameters::ballRadius)
        .def_readwrite("ballFriction", &Parameters::ballFriction)
        .def_readwrite("ballMass", &Parameters::ballMass)
        .def_readwrite("agentRadius", &Parameters::agentRadius)
        .def_readwrite("agentMass", &Parameters::agentMass)
        .def_readwrite("positionReward", &Parameters::positionReward)
        .def_readwrite("goalReward", &Parameters::goalReward)
        .def_readwrite("goalsPosition", &Parameters::goalsPosition);

    py::class_<Environment>(m, "Environment")
        .def(py::init<const Parameters&>())
        .def("reset", &Environment::reset)
        .def("step", &Environment::step)
        .def("getStates", &Environment::getStates)
        .def("getRewards", &Environment::getRewards);
}
