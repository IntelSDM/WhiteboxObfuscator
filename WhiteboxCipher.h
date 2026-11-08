#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

constexpr int kBaseRounds = 6;
constexpr int kMaxRounds = 8;
static_assert(kMaxRounds >= kBaseRounds, "kMaxRounds must accommodate kBaseRounds.");
constexpr int kBaseBlockBytes = 8;
constexpr int kMaxBlockBytes = 16;

inline uint8_t XTime(uint8_t x)
{
    return static_cast<uint8_t>((x << 1) ^ ((x & 0x80) ? 0x1B : 0));
}

inline uint8_t GFMul(uint8_t a, uint8_t b)
{
    uint8_t r = 0;
    for (int i = 0; i < 8; ++i)
    {
        if (b & 1) r ^= a;
        a = XTime(a);
        b >>= 1;
    }
    return r;
}

using ShiftPerm = std::array<uint8_t, kMaxBlockBytes>;

inline ShiftPerm IdentityShift(int blockbytes = kBaseBlockBytes)
{
    ShiftPerm s{};
    for (int i = 0; i < blockbytes; ++i)
        s[i] = static_cast<uint8_t>(i);
    return s;
}

inline ShiftPerm InvertShift(const ShiftPerm& s, int blockbytes = kBaseBlockBytes)
{
    ShiftPerm inv{};
    for (int i = 0; i < blockbytes; ++i)
        inv[s[i]] = static_cast<uint8_t>(i);
    return inv;
}

// must map at least one position in the first half to the second half (and vice versa). Otherwise MixColumns' column independence survives ShiftRows and the 64 bit block decomposes into two independent 32 bit ciphers
inline bool ShiftCrossesColumns(const ShiftPerm& s, int blockbytes = kBaseBlockBytes)
{
    const int half = blockbytes / 2;
    for (int i = 0; i < half; ++i)
        if (s[i] >= half) return true;
    return false;
}

inline ShiftPerm GenerateShift(std::mt19937_64& rng, int blockbytes = kBaseBlockBytes)
{
    ShiftPerm s{};
    for (;;)
    {
        for (int i = 0; i < blockbytes; ++i)
            s[i] = static_cast<uint8_t>(i);
        for (int i = blockbytes - 1; i > 0; --i)
        {
            const int j = static_cast<int>(rng() % static_cast<uint64_t>(i + 1));
            std::swap(s[i], s[j]);
        }
        if (ShiftCrossesColumns(s, blockbytes)) return s;
    }
}

inline ShiftPerm MakeAESAnalogShift()
{
    ShiftPerm s{};
    s[0] = 0; s[1] = 5; s[2] = 2; s[3] = 7;
    s[4] = 4; s[5] = 1; s[6] = 6; s[7] = 3;
    return s;
}

inline void ApplyShift(uint8_t* s, const ShiftPerm& shift, int blockbytes = kBaseBlockBytes)
{
    uint8_t tmp[kMaxBlockBytes];
    for (int i = 0; i < blockbytes; ++i) tmp[i] = s[shift[i]];
    for (int i = 0; i < blockbytes; ++i) s[i] = tmp[i];
}

inline void ApplyInvShift(uint8_t* s, const ShiftPerm& invshift, int blockbytes = kBaseBlockBytes)
{
    uint8_t tmp[kMaxBlockBytes];
    for (int i = 0; i < blockbytes; ++i) tmp[i] = s[invshift[i]];
    for (int i = 0; i < blockbytes; ++i) s[i] = tmp[i];
}

using SBox = std::array<uint8_t, 256>;

inline SBox GenerateSBox(std::mt19937_64& rng)
{
    SBox s;
    for (int i = 0; i < 256; ++i) s[i] = static_cast<uint8_t>(i);
    for (int i = 255; i > 0; --i)
    {
        const int j = static_cast<int>(rng() % static_cast<uint64_t>(i + 1));
        std::swap(s[i], s[j]);
    }
    return s;
}

inline SBox InvertSBox(const SBox& s)
{
    SBox inv{};
    for (int i = 0; i < 256; ++i) inv[s[i]] = static_cast<uint8_t>(i);
    return inv;
}

inline uint8_t RotateLeft8(uint8_t x, uint8_t k)
{
    k &= 7;
    return static_cast<uint8_t>((x << k) | (x >> (8 - k)));
}

inline uint8_t RotateRight8(uint8_t x, uint8_t k)
{
    k &= 7;
    return static_cast<uint8_t>((x >> k) | (x << (8 - k)));
}

inline uint64_t RotateStateBytesLeft(uint64_t state, uint8_t k)
{
    k &= 7;
    if (k == 0) return state;
    uint64_t out = 0;
    for (int b = 0; b < 8; ++b)
    {
        const uint8_t byte = static_cast<uint8_t>(state >> (b * 8));
        out |= static_cast<uint64_t>(RotateLeft8(byte, k)) << (b * 8);
    }
    return out;
}

using MDSMatrix = std::array<std::array<uint8_t, 4>, 4>;

constexpr MDSMatrix AESMDSMatrix = {{
    {0x02, 0x03, 0x01, 0x01},
    {0x01, 0x02, 0x03, 0x01},
    {0x01, 0x01, 0x02, 0x03},
    {0x03, 0x01, 0x01, 0x02},
}};

constexpr MDSMatrix AESInvMDSMatrix = {{
    {0x0E, 0x0B, 0x0D, 0x09},
    {0x09, 0x0E, 0x0B, 0x0D},
    {0x0D, 0x09, 0x0E, 0x0B},
    {0x0B, 0x0D, 0x09, 0x0E},
}};

inline void MixColumns32(uint8_t col[4], const MDSMatrix& mds)
{
    const uint8_t a0 = col[0];
    const uint8_t a1 = col[1];
    const uint8_t a2 = col[2];
    const uint8_t a3 = col[3];
    for (int r = 0; r < 4; ++r)
    {
        col[r] = static_cast<uint8_t>(
            GFMul(mds[r][0], a0) ^
            GFMul(mds[r][1], a1) ^
            GFMul(mds[r][2], a2) ^
            GFMul(mds[r][3], a3));
    }
}

inline void InvMixColumns32(uint8_t col[4], const MDSMatrix& invmds)
{
    const uint8_t a0 = col[0];
    const uint8_t a1 = col[1];
    const uint8_t a2 = col[2];
    const uint8_t a3 = col[3];
    for (int r = 0; r < 4; ++r)
    {
        col[r] = static_cast<uint8_t>(
            GFMul(invmds[r][0], a0) ^
            GFMul(invmds[r][1], a1) ^
            GFMul(invmds[r][2], a2) ^
            GFMul(invmds[r][3], a3));
    }
}

inline void PackU64LE(uint8_t out[8], uint64_t v)
{
    for (int i = 0; i < 8; ++i)
        out[i] = static_cast<uint8_t>(v >> (i * 8));
}

inline uint64_t UnpackU64LE(const uint8_t in[8])
{
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
        v |= static_cast<uint64_t>(in[i]) << (i * 8);
    return v;
}

inline uint64_t SPNEncrypt(uint64_t plaintext,
    const SBox& sbox,
    const std::array<uint64_t, kMaxRounds>& roundkeys,
    const ShiftPerm& shift,
    const MDSMatrix& mds,
    const uint8_t* bitrot = nullptr,
    const std::array<SBox, kMaxRounds>* perroundsboxes = nullptr,
    int rounds = kBaseRounds,
    const uint8_t* roundprogram = nullptr,
    const std::array<SBox, kMaxRounds>* gsboxesb = nullptr,
    const uint8_t* groutes = nullptr,
    int blockbytes = kBaseBlockBytes,
    uint64_t plaintexthi = 0,
    const std::array<uint64_t, kMaxRounds>* roundkeyshi = nullptr,
    uint64_t* ciphertexthi = nullptr)
{
    uint8_t s[kMaxBlockBytes] = {};
    PackU64LE(s, plaintext);
    if (blockbytes > 8) PackU64LE(s + 8, plaintexthi);

    for (int r = 0; r < rounds; ++r)
    {
        const uint8_t rt = (roundprogram != nullptr) ? roundprogram[r] : static_cast<uint8_t>(0);
        const SBox& roundsbox = (perroundsboxes != nullptr) ? (*perroundsboxes)[r] : sbox;

        if (rt == 1)
        {
            uint64_t state = UnpackU64LE(s);
            state ^= roundkeys[r];
            PackU64LE(s, state);
            if (blockbytes > 8 && roundkeyshi != nullptr)
            {
                uint64_t statehi = UnpackU64LE(s + 8);
                statehi ^= (*roundkeyshi)[r];
                PackU64LE(s + 8, statehi);
            }
        }
        else if (rt != 0 && rt != 2 && rt != 4 && rt != 5)
        {
            return 0;
        }

        if (rt == 4)
        {
        }
        else if (rt == 5 && gsboxesb != nullptr && groutes != nullptr)
        {
            const SBox& sboxb = (*gsboxesb)[r];
            const uint8_t route = groutes[r];
            for (int i = 0; i < blockbytes; ++i)
            {
                const bool useb = ((route >> i) & 1u) != 0;
                s[i] = useb ? sboxb[s[i]] : roundsbox[s[i]];
            }
        }
        else
        {
            for (int i = 0; i < blockbytes; ++i) s[i] = roundsbox[s[i]];
        }

        if (bitrot != nullptr && bitrot[r] != 0)
        {
            for (int i = 0; i < blockbytes; ++i)
                s[i] = RotateLeft8(s[i], bitrot[r]);
        }

        for (int c = 0; c < blockbytes / 4; ++c)
            MixColumns32(s + c * 4, mds);

        ApplyShift(s, shift, blockbytes);

        if (rt == 0 || rt == 4 || rt == 5)
        {
            uint64_t state = UnpackU64LE(s);
            state ^= roundkeys[r];
            PackU64LE(s, state);
            if (blockbytes > 8 && roundkeyshi != nullptr)
            {
                uint64_t statehi = UnpackU64LE(s + 8);
                statehi ^= (*roundkeyshi)[r];
                PackU64LE(s + 8, statehi);
            }
        }
    }

    if (blockbytes > 8 && ciphertexthi) *ciphertexthi = UnpackU64LE(s + 8);
    return UnpackU64LE(s);
}

inline uint64_t SPNDecrypt(uint64_t ciphertext,
    const SBox& invsbox,
    const std::array<uint64_t, kMaxRounds>& roundkeys,
    const ShiftPerm& invshift,
    const MDSMatrix& invmds,
    const uint8_t* bitrot = nullptr,
    const std::array<SBox, kMaxRounds>* perroundinvsboxes = nullptr,
    int rounds = kBaseRounds,
    const uint8_t* roundprogram = nullptr,
    const std::array<SBox, kMaxRounds>* ginvsboxesb = nullptr,
    const uint8_t* groutes = nullptr,
    int blockbytes = kBaseBlockBytes,
    uint64_t ciphertexthi = 0,
    const std::array<uint64_t, kMaxRounds>* roundkeyshi = nullptr,
    uint64_t* plaintexthi = nullptr)
{
    uint8_t s[kMaxBlockBytes] = {};
    PackU64LE(s, ciphertext);
    if (blockbytes > 8) PackU64LE(s + 8, ciphertexthi);

    for (int r = rounds - 1; r >= 0; --r)
    {
        const uint8_t rt = (roundprogram != nullptr) ? roundprogram[r] : static_cast<uint8_t>(0);
        const SBox& roundinvsbox = (perroundinvsboxes != nullptr) ? (*perroundinvsboxes)[r] : invsbox;

        if (rt == 0 || rt == 4 || rt == 5)
        {
            uint64_t state = UnpackU64LE(s);
            state ^= roundkeys[r];
            PackU64LE(s, state);
            if (blockbytes > 8 && roundkeyshi != nullptr)
            {
                uint64_t statehi = UnpackU64LE(s + 8);
                statehi ^= (*roundkeyshi)[r];
                PackU64LE(s + 8, statehi);
            }
        }
        else if (rt != 1 && rt != 2)
        {
            return 0;
        }

        ApplyInvShift(s, invshift, blockbytes);

        for (int c = 0; c < blockbytes / 4; ++c)
            InvMixColumns32(s + c * 4, invmds);

        if (bitrot != nullptr && bitrot[r] != 0)
        {
            for (int i = 0; i < blockbytes; ++i)
                s[i] = RotateRight8(s[i], bitrot[r]);
        }

        if (rt == 4)
        {
        }
        else if (rt == 5 && ginvsboxesb != nullptr && groutes != nullptr)
        {
            const SBox& invsboxb = (*ginvsboxesb)[r];
            const uint8_t route = groutes[r];
            for (int i = 0; i < blockbytes; ++i)
            {
                const bool useb = ((route >> i) & 1u) != 0;
                s[i] = useb ? invsboxb[s[i]] : roundinvsbox[s[i]];
            }
        }
        else
        {
            for (int i = 0; i < blockbytes; ++i) s[i] = roundinvsbox[s[i]];
        }

        if (rt == 1)
        {
            uint64_t state = UnpackU64LE(s);
            state ^= roundkeys[r];
            PackU64LE(s, state);
            if (blockbytes > 8 && roundkeyshi != nullptr)
            {
                uint64_t statehi = UnpackU64LE(s + 8);
                statehi ^= (*roundkeyshi)[r];
                PackU64LE(s + 8, statehi);
            }
        }
    }

    if (blockbytes > 8 && plaintexthi) *plaintexthi = UnpackU64LE(s + 8);
    return UnpackU64LE(s);
}