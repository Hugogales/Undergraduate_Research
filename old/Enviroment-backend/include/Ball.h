#pragma once
#include "Vector2D.h"

class Ball {
public:
    Ball();

    void applyFriction();
    void updatePosition();
    void reset();

    const Vector2D& getPosition() const;
    float getRadius() const;
    float getMass() const;
    const Vector2D& getMomentum() const;
    void setMomentum(const Vector2D& new_momentum);
    void setPosition(const Vector2D& new_position);

private:
    Vector2D position, momentum;
    float friction;
    float radius;
    float mass;
};
