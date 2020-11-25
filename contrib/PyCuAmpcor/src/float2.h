/*
 * @file  float2.h
 * @brief Define operators and functions on float2 (cuComplex) datatype
 *
 */

#ifndef __FLOAT2_H
#define __FLOAT2_H

#include <vector_types.h>

inline __host__ __device__ void zero(float2 &a) { a.x = 0.0f; a.y = 0.0f; }

// negative
inline __host__ __device__ float2 operator-(float2 &a)
{
    return make_float2(-a.x, -a.y);
}

// complex conjugate
inline __host__ __device__ float2 conjugate(float2 a)
{
    return make_float2(a.x, -a.y);
}

// addition
inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
}

inline __host__ __device__ float2 operator+(float2 a, float b)
{
    return make_float2(a.x + b, a.y);
}
inline __host__ __device__ void operator+=(float2 &a, float b)
{
    a.x += b;
}

// subtraction
inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2 &a, float2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ float2 operator-(float2 a, float b)
{
    return make_float2(a.x - b, a.y);
}
inline __host__ __device__ void operator-=(float2 &a, float b)
{
    a.x -= b;
}

// multiplication
inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x*b.x - a.y*b.y, a.y*b.x + a.x*b.y);
}
inline __host__ __device__ void operator*=(float2 &a, float2 b)
{
    a.x = a.x*b.x - a.y*b.y;
    a.y = a.y*b.x + a.x*b.y;
}
inline __host__ __device__ float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}
inline __host__ __device__ void operator*=(float2 &a, float b)
{
    a.x *= b;
    a.y *= b;
}
inline __host__ __device__ float2 operator*(float2 a, int b)
{
    return make_float2(a.x * b, a.y * b);
}
inline __host__ __device__ void operator*=(float2 &a, int b)
{
    a.x *= b;
    a.y *= b;
}
inline __host__ __device__ float2 complexMul(float2 a, float2 b)
{
    return a*b;
}
inline __host__ __device__ float2 complexMulConj(float2 a, float2 b)
{
    return make_float2(a.x*b.x + a.y*b.y, a.y*b.x - a.x*b.y);
}

inline __host__ __device__ float2 operator/(float2 a, float b)
{
    return make_float2(a.x / b, a.y / b);
}
inline __host__ __device__ void operator/=(float2 &a, float b)
{
    a.x /= b;
    a.y /= b;
}

// abs, arg
inline __host__ __device__ float complexAbs(float2 a)
{
    return sqrtf(a.x*a.x+a.y*a.y);
}
inline __host__ __device__ float complexArg(float2 a)
{
    return atan2f(a.y, a.x);
}

// make a complex number from phase
inline __host__ __device__ float2 complexExp(float arg)
{
    return make_float2(cosf(arg), sinf(arg));
}

#endif //__FLOAT2_H
// end of file