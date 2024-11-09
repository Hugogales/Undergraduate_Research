#pragma once
#include "Vector2D.h"

class Goal {
public:
    Goal(float x = 0.0f, float y = 0.0f);

    const Vector2D& getPosition() const;

private:
    Vector2D position;
};
