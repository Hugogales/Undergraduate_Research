#include "Ball.h"
#include <algorithm>

// Constructor
Ball::Ball() 
    : position(0.0f, 0.0f), momentum(0.0f, 0.0f), 
      friction(1.0f), radius(3.0f), mass(1.0f) {}

// Apply friction to ball's momentum
void Ball::applyFriction() {
    momentum.x = std::max(0.0f, momentum.x - friction);
    momentum.y = std::max(0.0f, momentum.y - friction);
}

// Update ball's position based on momentum
void Ball::updatePosition() { 
    position += momentum; 
}

// Reset ball's state
void Ball::reset() {
    position = Vector2D(0.0f, 0.0f);
    momentum = Vector2D(0.0f, 0.0f);
}

// Getters and Setters
const Vector2D& Ball::getPosition() const { 
    return position; 
}

float Ball::getRadius() const { 
    return radius; 
}

float Ball::getMass() const { 
    return mass; 
}

const Vector2D& Ball::getMomentum() const { 
    return momentum; 
}

void Ball::setMomentum(const Vector2D& new_momentum) { 
    momentum = new_momentum; 
}

void Ball::setPosition(const Vector2D& new_position) {
    position = new_position;
}
