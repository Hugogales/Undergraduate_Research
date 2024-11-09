#pragma once
#include "Vector2D.h"

class Agent {
public:
    Agent(int id);

    void applyForce(const Vector2D& force);
    void updatePosition();
    void reset();

    const Vector2D& getPosition() const;
    float getRadius() const;
    float getMass() const;
    const Vector2D& getMomentum() const;
    void setMomentum(const Vector2D& new_momentum);
    void setPosition(const Vector2D& new_position);

private:
    int id;
    Vector2D position, momentum;
    float radius;
    float mass;
};
