#include "Goal.h"

// Constructor
Goal::Goal(float x, float y) : position(x, y) {}

// Getter for position
const Vector2D& Goal::getPosition() const {
    return position;
}
