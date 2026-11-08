#pragma once

#include "WhiteboxCipher.h"

#include <cstdint>
#include <random>

enum class EncodingModule : uint8_t
{
    PerByteAffine = 0,
    PerColumnMB = 1,
    SelfEquivalence = 2,
    ImplicitEncoding = 3,
};

enum class RoundType : uint8_t
{
    Canonical = 0,
    OrderSwap = 1,
    NoARK = 2,
    MCFirst = 3,
    LinearOnly = 4,
    ParallelSBoxes = 5,
};

enum class MDSFamily : uint8_t
{
    Cauchy = 0,
    Vandermonde = 1,
    Hadamard = 2,
};

enum class ShiftKind : uint8_t
{
    Random = 0,
    AESAnalog = 1,
    BitRotBaked = 2,
    NibbleSwap = 3,
};

enum class FusionGranularity : uint8_t
{
    Full = 0,
    ARKOut = 1,
    ShiftOut = 2,
};

enum class XorTopology : uint8_t
{
    BalancedBinary = 0,
    LeftLeaning = 1,
    ThreeWayOnce = 2,
};

enum class ExtWrapKind : uint8_t
{
    BytePerm = 0,
    ByteAffine = 1,
    Layered = 2,
};

enum class SBoxScheme : uint8_t
{
    Shared = 0,
    PerRoundIndependent = 1,
};

enum class TableLayoutOrder : uint8_t
{
    TBoxesThenXor = 0,
    XorThenTBoxes = 1,
    Interleaved = 2,
    PerRoundPermuted = 3,
};

enum class BindingKind : uint8_t
{
    None = 0,
    ModeA = 1,
    ModeB = 2,
};

enum class BindingSource : uint8_t
{
    CBSPCRC = 0,
    SecurityCookie = 1,
    RAMFingerprint = 2,
    PathState = 3,
};

constexpr uint8_t kMaxSpecColumns = 4;

struct WbSpec
{
    uint8_t rounds;
    uint8_t blockbytes;
    uint8_t columns;
    uint8_t columnbytes;

    EncodingModule encodingmodule;
    RoundType roundprogram[kMaxRounds];
    XorTopology xortopology;
    MDSFamily mdsfamily;
    ShiftKind shiftkind;
    SBoxScheme sboxscheme;
    FusionGranularity fusion[kMaxRounds];

    uint8_t bitrot[kMaxRounds];
    uint8_t nibbleswapmask[kMaxRounds];

    ExtWrapKind extwrap;
    uint8_t extwrappositions[8];
    uint8_t extwrapcount;
    uint16_t extwrapfamilysize;

    uint16_t karroumidualprogram[kMaxRounds][kMaxSpecColumns];

    TableLayoutOrder tableorder;
    uint8_t physicalroundorder[kMaxRounds];
    uint8_t decoycount;

    BindingKind bindingkind;
    uint8_t bindround;
    BindingSource bindingsource;

    uint32_t instanceid;
    uint32_t specseed;

    static WbSpec MakeBaseline();
};

inline WbSpec WbSpec::MakeBaseline()
{
    WbSpec s{};
    s.rounds = static_cast<uint8_t>(kBaseRounds);
    s.blockbytes = static_cast<uint8_t>(kBaseBlockBytes);
    s.columnbytes = 4;
    s.columns = static_cast<uint8_t>(s.blockbytes / s.columnbytes);

    s.encodingmodule = EncodingModule::PerByteAffine;

    for (uint8_t r = 0; r < kMaxRounds; ++r)
        s.roundprogram[r] = RoundType::Canonical;

    s.xortopology = XorTopology::BalancedBinary;
    s.mdsfamily = MDSFamily::Cauchy;
    s.shiftkind = ShiftKind::Random;
    s.sboxscheme = SBoxScheme::Shared;

    for (uint8_t r = 0; r < kMaxRounds; ++r)
        s.fusion[r] = FusionGranularity::Full;

    for (uint8_t r = 0; r < kMaxRounds; ++r)
    {
        s.bitrot[r] = 0;
        s.nibbleswapmask[r] = 0;
    }

    s.extwrap = ExtWrapKind::BytePerm;
    s.extwrapcount = 2;
    s.extwrappositions[0] = 0;
    s.extwrappositions[1] = 7;
    for (int i = 2; i < 8; ++i) s.extwrappositions[i] = 0;
    s.extwrapfamilysize = 256;

    for (uint8_t r = 0; r < kMaxRounds; ++r)
        for (uint8_t c = 0; c < kMaxSpecColumns; ++c)
            s.karroumidualprogram[r][c] = 0;

    s.tableorder = TableLayoutOrder::TBoxesThenXor;
    for (uint8_t r = 0; r < kMaxRounds; ++r)
        s.physicalroundorder[r] = r;
    s.decoycount = 0;

    s.bindingkind = BindingKind::None;
    s.bindround = 0;
    s.bindingsource = BindingSource::CBSPCRC;

    s.instanceid = 0;
    s.specseed = 0;

    return s;
}

// NoARK must stay < 2 and never adjacent; LinearOnly must be interior only (never first or last round) and none adjacent to another LinearOnly. The boundary rounds must stay nonlinear, which LinearOnly isn't
inline bool IsRoundProgramLegal(const RoundType* program, int rounds)
{
    int ccount = 0;
    int fcount = 0;
    for (int r = 0; r < rounds; ++r)
    {
        if (program[r] == RoundType::NoARK)
        {
            ++ccount;
            if (r > 0 && program[r - 1] == RoundType::NoARK)
                return false;
        }
        if (program[r] == RoundType::LinearOnly)
        {
            ++fcount;
            if (r == 0 || r == rounds - 1)
                return false;
            if (program[r - 1] == RoundType::LinearOnly)
                return false;
        }
    }
    return ccount <= 2 && fcount <= 1;
}

inline void DrawRoundProgram(std::mt19937_64& rng, int rounds, RoundType* out)
{
    for (int r = 0; r < rounds; ++r)
        out[r] = (rng() & 1u) ? RoundType::OrderSwap : RoundType::Canonical;

    const int k = static_cast<int>(rng() % 3u);
    if (k >= 1)
    {
        const int p1 = static_cast<int>(rng() % static_cast<uint32_t>(rounds));
        out[p1] = RoundType::NoARK;
        if (k == 2)
        {
            int p2;
            do
            {
                p2 = static_cast<int>(rng() % static_cast<uint32_t>(rounds));
            } while (p2 == p1 || p2 == p1 - 1 || p2 == p1 + 1);
            out[p2] = RoundType::NoARK;
        }
    }

    if ((rng() & 1u) != 0u && rounds >= 3)
    {
        const int interiorspan = static_cast<int>(static_cast<uint32_t>(rounds) - 2u);
        int tries = 0;
        for (; tries < 8; ++tries)
        {
            const int pf = 1 + static_cast<int>(rng() % static_cast<uint32_t>(interiorspan));
            if (out[pf - 1] == RoundType::NoARK) continue;
            if (out[pf + 1] == RoundType::NoARK) continue;
            out[pf] = RoundType::LinearOnly;
            break;
        }
    }

    // only flip Canonical to ParallelSBoxes, never OrderSwap. OrderSwap's input ARK path doesn't compose cleanly with ParallelSBoxes's per position routing
    for (int r = 0; r < rounds; ++r)
    {
        if (out[r] == RoundType::Canonical && (rng() % 3u) == 0u)
            out[r] = RoundType::ParallelSBoxes;
    }
}