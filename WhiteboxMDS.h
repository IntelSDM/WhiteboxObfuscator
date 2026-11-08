#pragma once

#include "WhiteboxCipher.h"

#include <array>
#include <cstdint>
#include <random>
#include <utility>

inline uint8_t GFInv(uint8_t x)
{
    if (x == 0) return 0;
    uint8_t r = 1;
    uint8_t p = x;
    for (int i = 0; i < 8; ++i)
    {
        if ((254 >> i) & 1)
            r = GFMul(r, p);
        p = GFMul(p, p);
    }
    return r;
}

inline MDSMatrix InvertMDS(const MDSMatrix& m)
{
    uint8_t aug[4][8] = {};
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j) aug[i][j] = m[i][j];
        aug[i][4 + i] = 1;
    }
    for (int col = 0; col < 4; ++col)
    {
        int pivot = -1;
        for (int r = col; r < 4; ++r)
        {
            if (aug[r][col] != 0) { pivot = r; break; }
        }
        if (pivot < 0)
        {
            // singular input. return identity so the caller gets a deterministic (but wrong) matrix; callers that may feed none Cauchy input must IsMDS check first
            MDSMatrix i{};
            for (int k = 0; k < 4; ++k) i[k][k] = 1;
            return i;
        }
        if (pivot != col)
        {
            for (int j = 0; j < 8; ++j)
                std::swap(aug[col][j], aug[pivot][j]);
        }
        const uint8_t inv = GFInv(aug[col][col]);
        for (int j = 0; j < 8; ++j)
            aug[col][j] = GFMul(aug[col][j], inv);
        for (int r = 0; r < 4; ++r)
        {
            if (r == col) continue;
            const uint8_t factor = aug[r][col];
            if (factor == 0) continue;
            for (int j = 0; j < 8; ++j)
                aug[r][j] = static_cast<uint8_t>(aug[r][j] ^ GFMul(aug[col][j], factor));
        }
    }
    MDSMatrix inv{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            inv[i][j] = aug[i][4 + j];
    return inv;
}

inline uint8_t GFSubDet(const MDSMatrix& m, const int* rows, const int* cols, int k)
{
    if (k == 1) return m[rows[0]][cols[0]];
    uint8_t det = 0;
    int subrows[3];
    int subcols[3];
    for (int j = 0; j < k; ++j)
    {
        const uint8_t coeff = m[rows[0]][cols[j]];
        if (coeff == 0) continue;
        for (int i = 1; i < k; ++i) subrows[i - 1] = rows[i];
        int sc = 0;
        for (int jj = 0; jj < k; ++jj)
            if (jj != j) subcols[sc++] = cols[jj];
        const uint8_t sub = GFSubDet(m, subrows, subcols, k - 1);
        det = static_cast<uint8_t>(det ^ GFMul(coeff, sub));
    }
    return det;
}

inline int PopCount4(int mask)
{
    return ((mask >> 0) & 1) + ((mask >> 1) & 1) + ((mask >> 2) & 1) + ((mask >> 3) & 1);
}

inline bool IsMDS(const MDSMatrix& m)
{
    int rows[4];
    int cols[4];
    for (int k = 1; k <= 4; ++k)
    {
        for (int rmask = 0; rmask < 16; ++rmask)
        {
            if (PopCount4(rmask) != k) continue;
            int rc = 0;
            for (int i = 0; i < 4; ++i)
                if (rmask & (1 << i)) rows[rc++] = i;
            for (int cmask = 0; cmask < 16; ++cmask)
            {
                if (PopCount4(cmask) != k) continue;
                int cc = 0;
                for (int j = 0; j < 4; ++j)
                    if (cmask & (1 << j)) cols[cc++] = j;
                if (GFSubDet(m, rows, cols, k) == 0) return false;
            }
        }
    }
    return true;
}

inline MDSMatrix MakeCauchyMDS(std::mt19937_64& rng)
{
    std::array<uint8_t, 256> pool{};
    for (int i = 0; i < 256; ++i) pool[i] = static_cast<uint8_t>(i);
    for (int i = 0; i < 8; ++i)
    {
        const int j = i + static_cast<int>(rng() % static_cast<uint64_t>(256 - i));
        std::swap(pool[i], pool[j]);
    }
    const uint8_t xs[4] = {pool[0], pool[1], pool[2], pool[3]};
    const uint8_t ys[4] = {pool[4], pool[5], pool[6], pool[7]};
    MDSMatrix m{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            m[i][j] = GFInv(static_cast<uint8_t>(xs[i] ^ ys[j]));
    return m;
}

inline MDSMatrix MakeVandermondeMDS(std::mt19937_64& rng)
{
    constexpr int maxattempts = 64;
    for (int attempt = 0; attempt < maxattempts; ++attempt)
    {
        std::array<uint8_t, 255> pool{};
        for (int i = 0; i < 255; ++i)
            pool[i] = static_cast<uint8_t>(1 + i);
        for (int i = 0; i < 4; ++i)
        {
            const int j = i + static_cast<int>(rng() % static_cast<uint64_t>(255 - i));
            std::swap(pool[i], pool[j]);
        }
        const uint8_t alpha[4] = {pool[0], pool[1], pool[2], pool[3]};
        MDSMatrix m{};
        for (int i = 0; i < 4; ++i)
        {
            uint8_t power = 1;
            for (int j = 0; j < 4; ++j)
            {
                m[i][j] = power;
                power = GFMul(power, alpha[i]);
            }
        }
        if (IsMDS(m)) return m;
    }
    return MakeCauchyMDS(rng);
}

inline MDSMatrix MakeHadamardMDS(std::mt19937_64& rng)
{
    constexpr int maxattempts = 256;
    for (int attempt = 0; attempt < maxattempts; ++attempt)
    {
        std::array<uint8_t, 255> pool{};
        for (int i = 0; i < 255; ++i)
            pool[i] = static_cast<uint8_t>(1 + i);
        for (int i = 0; i < 4; ++i)
        {
            const int j = i + static_cast<int>(rng() % static_cast<uint64_t>(255 - i));
            std::swap(pool[i], pool[j]);
        }
        const uint8_t h[4] = {pool[0], pool[1], pool[2], pool[3]};
        MDSMatrix m{};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                m[i][j] = h[i ^ j];
        if (IsMDS(m)) return m;
    }
    return MakeCauchyMDS(rng);
}