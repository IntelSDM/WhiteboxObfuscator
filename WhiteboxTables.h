#pragma once

#include "WhiteboxCipher.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

struct Encoding
{
    std::array<std::array<uint8_t, 256>, kMaxBlockBytes> perm;
    std::array<std::array<uint8_t, 256>, kMaxBlockBytes> invperm;

    uint64_t Apply(uint64_t state, int blockbytes = kBaseBlockBytes) const
    {
        uint64_t out = 0;
        for (int i = 0; i < blockbytes; ++i)
        {
            const uint8_t b = static_cast<uint8_t>(state >> (i * 8));
            out |= static_cast<uint64_t>(perm[i][b]) << (i * 8);
        }
        return out;
    }

    uint64_t Invert(uint64_t state, int blockbytes = kBaseBlockBytes) const
    {
        uint64_t out = 0;
        for (int i = 0; i < blockbytes; ++i)
        {
            const uint8_t b = static_cast<uint8_t>(state >> (i * 8));
            out |= static_cast<uint64_t>(invperm[i][b]) << (i * 8);
        }
        return out;
    }
};

inline Encoding GenerateEncoding(std::mt19937_64& rng, int blockbytes = kBaseBlockBytes)
{
    Encoding e{};
    for (int pos = 0; pos < blockbytes; ++pos)
    {
        for (int i = 0; i < 256; ++i) e.perm[pos][i] = static_cast<uint8_t>(i);
        for (int i = 255; i > 0; --i)
        {
            const int j = static_cast<int>(rng() % static_cast<uint64_t>(i + 1));
            std::swap(e.perm[pos][i], e.perm[pos][j]);
        }
        for (int i = 0; i < 256; ++i)
            e.invperm[pos][e.perm[pos][i]] = static_cast<uint8_t>(i);
    }
    return e;
}

struct EncodingSet
{
    Encoding inenc;
    std::array<Encoding, kMaxRounds> roundenc;
};

// populate ALL kMaxRounds slots unconditionally. At spec.rounds > kBaseRounds, roundenc[kBaseRounds..spec.rounds) would otherwise stay default initialised (byte identity, not a valid random perm) and the Tbox / XOR-table builders reading those slots would silently produce garbage
inline EncodingSet GenerateEncodings(std::mt19937_64& rng, int blockbytes = kBaseBlockBytes)
{
    EncodingSet s;
    s.inenc = GenerateEncoding(rng, blockbytes);
    for (int r = 0; r < kMaxRounds; ++r)
        s.roundenc[r] = GenerateEncoding(rng, blockbytes);
    return s;
}

inline uint8_t MDSCoeff(const MDSMatrix& mds, int outrow, int inrow)
{
    return mds[outrow][inrow];
}

using TBoxEntry = std::array<uint64_t, 2>;
using TBox = std::array<TBoxEntry, 256>;
using TBoxes = std::array<std::array<TBox, kMaxBlockBytes>, kMaxRounds>;
using XorByteTable = std::array<std::array<uint8_t, 256>, 256>;
using XorTables = std::array<std::array<XorByteTable, kMaxBlockBytes>, kMaxRounds>;

using ARKByteTable = std::array<uint8_t, 256>;
using ARKTables = std::array<std::array<ARKByteTable, kMaxBlockBytes>, kMaxRounds>;

inline void BuildTBoxesInto(TBoxes& t,
    const SBox& sbox,
    const std::array<uint64_t, kMaxRounds>& roundkeys,
    const EncodingSet& enc,
    const ShiftPerm& shift,
    const MDSMatrix& mds,
    const uint8_t* bitrot = nullptr,
    const std::array<SBox, kMaxRounds>* perroundsboxes = nullptr,
    int rounds = kBaseRounds,
    const uint8_t* roundprogram = nullptr,
    const uint8_t* fusion = nullptr,
    const std::array<SBox, kMaxRounds>* gsboxesb = nullptr,
    const uint8_t* groutes = nullptr,
    int blockbytes = kBaseBlockBytes,
    const std::array<uint64_t, kMaxRounds>* roundkeyshi = nullptr)
{
    const auto invshift = InvertShift(shift, blockbytes);

    for (int r = 0; r < rounds; ++r)
    {
        const Encoding& inenc = (r == 0) ? enc.inenc : enc.roundenc[r - 1];
        const Encoding& outenc = enc.roundenc[r];

        uint8_t rkbytes[kMaxBlockBytes] = {};
        PackU64LE(rkbytes, roundkeys[r]);
        if (roundkeyshi != nullptr && blockbytes > 8)
            PackU64LE(rkbytes + 8, (*roundkeyshi)[r]);

        const uint8_t rotamount = (bitrot != nullptr) ? bitrot[r] : 0;
        const SBox& roundsbox = (perroundsboxes != nullptr) ? (*perroundsboxes)[r] : sbox;

        const uint8_t rt = (roundprogram != nullptr) ? roundprogram[r] : static_cast<uint8_t>(0);
        if (rt != 0 && rt != 1 && rt != 2 && rt != 4 && rt != 5) return;

        const uint8_t fg = (fusion != nullptr) ? fusion[r] : static_cast<uint8_t>(0);
        if (fg == 2) return;
        // bug: ARKOut is Canonical-only — OrderSwap bakes ARK input-side, NoARK has no key; OrderSwap-ARKOut / NoARK-ARKOut would produce a stale ARK table against the actual round shape
        if (fg == 1 && rt != 0) return;

        for (int i = 0; i < blockbytes; ++i)
        {
            const int c = i / 4;
            const int j = i % 4;

            for (int x = 0; x < 256; ++x)
            {
                const uint8_t decoded = inenc.invperm[i][x];
                const uint8_t pre = (rt == 1) ? static_cast<uint8_t>(decoded ^ rkbytes[i]) : decoded;
                uint8_t subbed;
                if (rt == 4)
                {
                    subbed = pre;
                }
                else if (rt == 5 && gsboxesb != nullptr && groutes != nullptr)
                {
                    const uint8_t route = groutes[r];
                    const bool useb = ((route >> i) & 1u) != 0;
                    subbed = useb ? (*gsboxesb)[r][pre] : roundsbox[pre];
                }
                else
                {
                    subbed = roundsbox[pre];
                }
                if (rotamount != 0)
                    subbed = RotateLeft8(subbed, rotamount);

                uint8_t contribution[4];
                for (int b = 0; b < 4; ++b)
                {
                    const uint8_t coeff = MDSCoeff(mds, b, j);
                    contribution[b] = GFMul(coeff, subbed);
                }

                // bug: output-side ARK must fire for row-0 T-boxes only (j == 0) — if more than one row's T-box per column adds the round key, XOR cancellation makes it disappear; if none adds it, the cipher silently drops the key
                if ((rt == 0 || rt == 4 || rt == 5) && j == 0 && fg != 1)
                {
                    for (int b = 0; b < 4; ++b)
                    {
                        const int postshiftpos = invshift[c * 4 + b];
                        contribution[b] ^= rkbytes[postshiftpos];
                    }
                }

                uint64_t entrylo = 0, entryhi = 0;
                for (int b = 0; b < 4; ++b)
                {
                    const int postshiftpos = invshift[c * 4 + b];
                    const uint8_t encbyte = outenc.perm[postshiftpos][contribution[b]];
                    if (postshiftpos < 8)
                        entrylo |= static_cast<uint64_t>(encbyte) << (postshiftpos * 8);
                    else
                        entryhi |= static_cast<uint64_t>(encbyte) << ((postshiftpos - 8) * 8);
                }
                t[r][i][x][0] = entrylo;
                t[r][i][x][1] = entryhi;
            }
        }
    }
}

inline void BuildXorTablesInto(XorTables& x, const EncodingSet& enc, int rounds = kBaseRounds, int blockbytes = kBaseBlockBytes)
{
    for (int r = 0; r < rounds; ++r)
    {
        for (int p = 0; p < blockbytes; ++p)
        {
            const auto& perm = enc.roundenc[r].perm[p];
            const auto& invperm = enc.roundenc[r].invperm[p];
            for (int a = 0; a < 256; ++a)
            {
                for (int b = 0; b < 256; ++b)
                {
                    const uint8_t da = invperm[a];
                    const uint8_t db = invperm[b];
                    const uint8_t realxor = static_cast<uint8_t>(da ^ db);
                    x[r][p][a][b] = perm[realxor];
                }
            }
        }
    }
}

inline void BuildARKTablesInto(ARKTables& a,
    const std::array<uint64_t, kMaxRounds>& roundkeys,
    const EncodingSet& enc,
    const uint8_t* fusion = nullptr,
    int rounds = kBaseRounds,
    int blockbytes = kBaseBlockBytes,
    const std::array<uint64_t, kMaxRounds>* roundkeyshi = nullptr)
{
    for (int r = 0; r < rounds; ++r)
    {
        const uint8_t fg = (fusion != nullptr) ? fusion[r] : static_cast<uint8_t>(0);
        if (fg != 1) continue;

        uint8_t rkbytes[kMaxBlockBytes] = {};
        PackU64LE(rkbytes, roundkeys[r]);
        if (roundkeyshi != nullptr && blockbytes > 8)
            PackU64LE(rkbytes + 8, (*roundkeyshi)[r]);
        for (int p = 0; p < blockbytes; ++p)
        {
            const auto& perm = enc.roundenc[r].perm[p];
            const auto& invperm = enc.roundenc[r].invperm[p];
            for (int y = 0; y < 256; ++y)
                a[r][p][y] = perm[static_cast<uint8_t>(invperm[y] ^ rkbytes[p])];
        }
    }
}

inline uint64_t WbEncrypt(uint64_t plaintext,
    const TBoxes& t,
    const XorTables& x,
    const EncodingSet& enc,
    const ShiftPerm& shift,
    int rounds = kBaseRounds,
    uint8_t xortopology = 0,
    const uint8_t* fusion = nullptr,
    const ARKTables* arktables = nullptr,
    int blockbytes = kBaseBlockBytes,
    uint64_t plaintexthi = 0,
    uint64_t* ciphertexthi = nullptr)
{
    uint8_t contributingcol[kMaxBlockBytes];
    for (int p = 0; p < blockbytes; ++p)
        contributingcol[p] = static_cast<uint8_t>(shift[p] / 4);

    if (xortopology > 1) return 0;

    uint64_t statelo = enc.inenc.Apply(plaintext, 8);
    uint64_t statehi = 0;
    if (blockbytes > 8)
    {
        for (int i = 8; i < blockbytes; ++i)
        {
            const uint8_t b = static_cast<uint8_t>(plaintexthi >> ((i - 8) * 8));
            statehi |= static_cast<uint64_t>(enc.inenc.perm[i][b]) << ((i - 8) * 8);
        }
    }

    for (int r = 0; r < rounds; ++r)
    {
        uint64_t contriblo[kMaxBlockBytes];
        uint64_t contribhi[kMaxBlockBytes];
        for (int i = 0; i < blockbytes; ++i)
        {
            const uint8_t statebyte = (i < 8) ? static_cast<uint8_t>(statelo >> (i * 8)) : static_cast<uint8_t>(statehi >> ((i - 8) * 8));
            contriblo[i] = t[r][i][statebyte][0];
            contribhi[i] = t[r][i][statebyte][1];
        }

        uint64_t newlo = 0, newhi = 0;
        for (int p = 0; p < blockbytes; ++p)
        {
            const int c = contributingcol[p];
            auto extract = [&](int b) -> uint8_t
            {
                const uint64_t v = (p < 8) ? contriblo[c * 4 + b] : contribhi[c * 4 + b];
                const int shiftamt = (p < 8) ? (p * 8) : ((p - 8) * 8);
                return static_cast<uint8_t>(v >> shiftamt);
            };
            const uint8_t b0 = extract(0);
            const uint8_t b1 = extract(1);
            const uint8_t b2 = extract(2);
            const uint8_t b3 = extract(3);

            uint8_t step;
            if (xortopology == 0)
            {
                step = x[r][p][b0][b1];
                step = x[r][p][step][b2];
                step = x[r][p][step][b3];
            }
            else
            {
                const uint8_t s1 = x[r][p][b0][b1];
                const uint8_t s2 = x[r][p][b2][b3];
                step = x[r][p][s1][s2];
            }

            if (p < 8)
                newlo |= static_cast<uint64_t>(step) << (p * 8);
            else
                newhi |= static_cast<uint64_t>(step) << ((p - 8) * 8);
        }

        if (arktables != nullptr && fusion != nullptr && fusion[r] == 1)
        {
            uint64_t arkedlo = 0, arkedhi = 0;
            for (int p = 0; p < blockbytes; ++p)
            {
                const uint8_t y = (p < 8) ? static_cast<uint8_t>(newlo >> (p * 8)) : static_cast<uint8_t>(newhi >> ((p - 8) * 8));
                const uint8_t a = (*arktables)[r][p][y];
                if (p < 8)
                    arkedlo |= static_cast<uint64_t>(a) << (p * 8);
                else
                    arkedhi |= static_cast<uint64_t>(a) << ((p - 8) * 8);
            }
            newlo = arkedlo;
            newhi = arkedhi;
        }

        statelo = newlo;
        statehi = newhi;
    }

    uint64_t outlo = enc.roundenc[rounds - 1].Invert(statelo, 8);
    uint64_t outhi = 0;
    if (blockbytes > 8)
    {
        for (int i = 8; i < blockbytes; ++i)
        {
            const uint8_t b = static_cast<uint8_t>(statehi >> ((i - 8) * 8));
            outhi |= static_cast<uint64_t>(enc.roundenc[rounds - 1].invperm[i][b]) << ((i - 8) * 8);
        }
    }
    if (ciphertexthi) *ciphertexthi = outhi;
    return outlo;
}

constexpr uint32_t kInEncOffset = 0;
constexpr uint32_t kInEncSize = static_cast<uint32_t>(kMaxBlockBytes) * 256u;
constexpr uint32_t kOutEncInvOffset = kInEncOffset + kInEncSize;
constexpr uint32_t kOutEncInvSize = static_cast<uint32_t>(kMaxBlockBytes) * 256u;
constexpr uint32_t kTBoxOffset = kOutEncInvOffset + kOutEncInvSize;
constexpr uint32_t kTBoxEntryBytes = 16u;
constexpr uint32_t kTBoxSize = static_cast<uint32_t>(kMaxRounds) * static_cast<uint32_t>(kMaxBlockBytes) * 256u * kTBoxEntryBytes;
constexpr uint32_t kXorTblOffset = kTBoxOffset + kTBoxSize;
constexpr uint32_t kXorTblSize = static_cast<uint32_t>(kMaxRounds) * static_cast<uint32_t>(kMaxBlockBytes) * 256u * 256u;
constexpr uint32_t kARKTblOffset = kXorTblOffset + kXorTblSize;
constexpr uint32_t kARKTblSize = static_cast<uint32_t>(kMaxRounds) * static_cast<uint32_t>(kMaxBlockBytes) * 256u;
constexpr uint32_t kCipherDataSize = kARKTblOffset + kARKTblSize;

constexpr uint32_t kTBoxEntryStride = kTBoxEntryBytes;
constexpr uint32_t kTBoxRowStride = 256u * kTBoxEntryStride;
constexpr uint32_t kTBoxRoundStride = static_cast<uint32_t>(kMaxBlockBytes) * kTBoxRowStride;

constexpr uint32_t kXorTblTableStride = 256u * 256u;
constexpr uint32_t kXorTblRoundStride = static_cast<uint32_t>(kMaxBlockBytes) * kXorTblTableStride;

inline uint32_t TBoxDisp(int r, int inputpos)
{
    return kTBoxOffset + static_cast<uint32_t>(r) * kTBoxRoundStride + static_cast<uint32_t>(inputpos) * kTBoxRowStride;
}

inline uint32_t XorTblDisp(int r, int outputpos)
{
    return kXorTblOffset + static_cast<uint32_t>(r) * kXorTblRoundStride + static_cast<uint32_t>(outputpos) * kXorTblTableStride;
}

inline uint32_t ARKTblDisp(int r, int outputpos)
{
    return kARKTblOffset + static_cast<uint32_t>(r) * (static_cast<uint32_t>(kMaxBlockBytes) * 256u) + static_cast<uint32_t>(outputpos) * 256u;
}

inline uint32_t InEncDisp(int bytepos)
{
    return kInEncOffset + static_cast<uint32_t>(bytepos) * 256u;
}

inline uint32_t OutEncInvDisp(int bytepos)
{
    return kOutEncInvOffset + static_cast<uint32_t>(bytepos) * 256u;
}

// bug: rounds parameter must match BuildTBoxesInto's rounds and BuildXorTablesInto's round count — packing kBaseRounds slots at spec.rounds=8 leaves rounds 6 and 7 as zero-filled bytes and the runtime lookups for those rounds return zeros, producing garbage output
inline std::vector<uint8_t> PackWhiteboxCipherData(const EncodingSet& enc,
    const TBoxes& t,
    const XorTables& x,
    int rounds = kBaseRounds,
    const ARKTables* ark = nullptr,
    int blockbytes = kBaseBlockBytes)
{
    std::vector<uint8_t> data(kCipherDataSize);

    for (int p = 0; p < blockbytes; ++p)
        std::memcpy(data.data() + InEncDisp(p), enc.inenc.perm[p].data(), 256);

    for (int p = 0; p < blockbytes; ++p)
        std::memcpy(data.data() + OutEncInvDisp(p), enc.roundenc[rounds - 1].invperm[p].data(), 256);

    for (int r = 0; r < rounds; ++r)
    {
        for (int i = 0; i < blockbytes; ++i)
        {
            const uint32_t base = TBoxDisp(r, i);
            for (int xi = 0; xi < 256; ++xi)
            {
                const uint64_t vlo = t[r][i][xi][0];
                const uint64_t vhi = t[r][i][xi][1];
                const uint32_t ent = xi * kTBoxEntryStride;
                for (int b = 0; b < 8; ++b)
                    data[base + ent + b] = static_cast<uint8_t>(vlo >> (b * 8));
                for (int b = 0; b < 8; ++b)
                    data[base + ent + 8 + b] = static_cast<uint8_t>(vhi >> (b * 8));
            }
        }
    }

    for (int r = 0; r < rounds; ++r)
    {
        for (int p = 0; p < blockbytes; ++p)
        {
            const uint32_t base = XorTblDisp(r, p);
            for (int a = 0; a < 256; ++a)
                std::memcpy(data.data() + base + a * 256, x[r][p][a].data(), 256);
        }
    }

    if (ark != nullptr)
    {
        for (int r = 0; r < rounds; ++r)
            for (int p = 0; p < blockbytes; ++p)
                std::memcpy(data.data() + ARKTblDisp(r, p), (*ark)[r][p].data(), 256);
    }

    return data;
}