#pragma once
#include <vector>

struct Vector2D {
    float x;
    float y;

    Vector2D(float x = 0.0f, float y = 0.0f);

    Vector2D operator+(const Vector2D& other) const;
    Vector2D& operator+=(const Vector2D& other);
    Vector2D operator-(const Vector2D& other) const;
    Vector2D& operator-=(const Vector2D& other);
    Vector2D operator*(float scalar) const;
    std::vector<float> toVector() const;
};
