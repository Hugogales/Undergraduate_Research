cmake_minimum_required(VERSION 3.10)
project(MultiAgentRL)

set(CMAKE_CXX_STANDARD 17)

# Find Boost
find_package(Boost REQUIRED COMPONENTS system)

# Find jsoncpp
find_package(jsoncpp REQUIRED)

# Find pybind11
find_package(pybind11 REQUIRED)

# Include directories
include_directories(include)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${JSONCPP_INCLUDE_DIRS})

# Source files
file(GLOB SOURCES "src/*.cpp")

# Python bindings
pybind11_add_module(multi_agent_env bindings/bindings.cpp ${SOURCES})

# Link libraries
target_link_libraries(multi_agent_env PRIVATE Boost::system jsoncpp_lib)
