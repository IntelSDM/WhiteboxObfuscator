#pragma once

#include "WhiteboxTables.h"

#include <cstdint>

constexpr uint64_t kMixerMul1 = 0xFF51AFD7ED558CCDULL;
constexpr uint64_t kMixerMul2 = 0xC4CEB9FE1A85EC53ULL;

constexpr uint32_t kRwSlotSize = 16;
constexpr uint32_t kRwOffsetMasked = 0;
constexpr uint32_t kRwOffsetAlpha = 8;

constexpr uint32_t kSeedOffset = kCipherDataSize;
constexpr uint32_t kCipherDataSizeWithSeed = kCipherDataSize + 8;

inline uint64_t StaffordMix13(uint64_t y)
{
    y ^= y >> 33;
    y *= kMixerMul1;
    y ^= y >> 33;
    y *= kMixerMul2;
    y ^= y >> 33;
    return y;
}

inline uint64_t NextAlpha(uint64_t alphacurr, uint64_t seedglobal)
{
    return StaffordMix13(alphacurr ^ seedglobal);
}