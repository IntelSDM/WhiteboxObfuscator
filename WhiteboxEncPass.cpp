#include "WhiteboxEncPass.h"

#include "WhiteboxCipher.h"
#include "WhiteboxMasking.h"
#include "WhiteboxMDS.h"
#include "WhiteboxSpec.h"
#include "WhiteboxTables.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <array>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

using namespace llvm;

cl::opt<bool> WhiteboxEnable("obfwhitebox", cl::init(true), cl::desc("Master Toggle"));

static constexpr uint32_t kMaxSites = 4096;

static constexpr uint8_t kExtWrapMaxCount = 8;
static constexpr uint8_t kExtWrapMaxDepth = 4;
static constexpr uint32_t kExtWrapFamilySizeMax = 1024;
static constexpr uint32_t kExtWrapRegionMaxBytes =
    static_cast<uint32_t>(kExtWrapMaxCount) * static_cast<uint32_t>(kExtWrapMaxDepth) * kExtWrapFamilySizeMax * 256u;

static constexpr uint32_t kPreencBankBytes =
    static_cast<uint32_t>(kMaxRounds) * 8u * 256u * 8u;
static constexpr uint32_t kInvPermsPerRoundBytes =
    static_cast<uint32_t>(kMaxRounds) * 8u * 256u;
static constexpr uint32_t kInvPermsPerRoundRowStride = 8u * 256u;
static constexpr uint32_t kGFMulTblBytes = 256u * 256u;
static constexpr uint32_t kFreshInvSBoxBytes = 256u;
static constexpr uint32_t kFreshRkBytes = static_cast<uint32_t>(kMaxRounds) * 8u;
static constexpr uint32_t kSBoxGBBytes = static_cast<uint32_t>(kMaxRounds) * 256u;
static constexpr uint32_t kInvSBoxGBBytes = static_cast<uint32_t>(kMaxRounds) * 256u;

struct WbExtWrapEntry
{
    std::array<uint8_t, 256> fwd;
};

struct WbModuleState
{
    WbSpec spec{};
    uint8_t rounds = 0;
    uint8_t blockbytes = 0;
    SBox sbox{};
    SBox invsbox{};
    std::array<uint64_t, kMaxRounds> roundkeys{};
    std::array<uint64_t, kMaxRounds> roundkeyshi{};
    ShiftPerm shift{};
    ShiftPerm invshift{};
    MDSMatrix mds{};
    MDSMatrix invmds{};
    EncodingSet enc{};
    std::vector<uint8_t> packedcipher;
    uint64_t seed = 0;
    uint64_t seedglobal = 0;

    std::array<uint8_t, kMaxRounds> bitrot{};
    std::array<uint8_t, kMaxRounds> roundprogram{};
    std::array<uint8_t, kMaxRounds> fusion{};
    bool anyarkout = false;
    std::array<SBox, kMaxRounds> perroundsboxes{};
    std::array<SBox, kMaxRounds> perroundinvsboxes{};
    bool hasperroundsboxes = false;

    uint8_t extwrapcount = 0;
    uint8_t extwrapdepth = 0;
    uint32_t extwrapfamilysize = 0;
    std::array<uint8_t, kExtWrapMaxCount> extwrappositions{};
    std::vector<WbExtWrapEntry> extfwdperms;

    uint32_t offextwrapregion = 0;
    uint32_t offpreencbank = 0;
    uint32_t offinvpermsperround = 0;
    uint32_t offgfmultbl = 0;
    uint32_t offfreshinvsbox = 0;
    uint32_t offfreshrk = 0;
    uint32_t offsboxgb = 0;
    uint32_t offinvsboxgb = 0;

    std::array<SBox, kMaxRounds> gsboxb{};
    std::array<SBox, kMaxRounds> ginvsboxb{};
    std::array<uint8_t, kMaxRounds> groute{};
    uint64_t groutepacked = 0;
    bool anyground = false;

    // ctor regen IR hardcodes 8-byte state throughout at bb=16 the fresh math pipeline would produce garbage, so BuildCipherState force-flips these off whenever it draws blockbytes > 8
    bool freshinenc = true;
    bool freshroundenc = true;
    bool freshciphermath = true;

    GlobalVariable* gvstatic = nullptr;
    GlobalVariable* gvdyn = nullptr;
    GlobalVariable* gvbct = nullptr;
    GlobalVariable* gvbetas = nullptr;
    GlobalVariable* gvrw0 = nullptr;
    Function* fncipher = nullptr;

    // live source for runtime cipherdata reads must be set once in RunWhiteboxEnc before EmitCipherFunction and site rewrite so both paths see the same base
    GlobalVariable* gvlive = nullptr;
};

struct WbSite
{
    Function* f = nullptr;
    Instruction* user = nullptr;
    unsigned opidx = 0;
    bool iscallarg = false;
    unsigned bitwidth = 0;
    uint64_t k = 0;
    uint32_t slotid = 0;
    uint64_t bct = 0;
    uint64_t betasite = 0;
    uint64_t alphainit = 0;
    uint64_t maskedinit = 0;
    uint64_t bcthi = 0;
    uint64_t betasitehi = 0;
    uint64_t alphainithi = 0;
    uint64_t maskedinithi = 0;
    uint64_t wideningpad = 0;
    std::array<std::array<uint16_t, kExtWrapMaxDepth>, kExtWrapMaxCount> familyindices{};
    bool extencon = false;
};

struct WbCandidate
{
    Instruction* user = nullptr;
    unsigned opidx = 0;
    bool iscallarg = false;
};

static inline size_t BctSlotBytes(const WbModuleState& s)
{
    return static_cast<size_t>(s.blockbytes);
}

static inline size_t BetasSlotBytes(const WbModuleState& s)
{
    return static_cast<size_t>(s.blockbytes);
}

static inline size_t Rw0SlotBytes(const WbModuleState& s)
{
    return static_cast<size_t>(s.blockbytes) * 2u;
}

static bool HasAnnotationTag(Function& f, const char* tag)
{
    Module* m = f.getParent();
    GlobalVariable* ga = m->getGlobalVariable("llvm.global.annotations");
    if (!ga) return false;
    ConstantArray* ca = dyn_cast<ConstantArray>(ga->getOperand(0));
    if (!ca) return false;
    for (unsigned i = 0; i < ca->getNumOperands(); ++i)
    {
        ConstantStruct* cs = dyn_cast<ConstantStruct>(ca->getOperand(i));
        if (!cs) continue;
        if (cs->getOperand(0)->stripPointerCasts() != &f) continue;
        auto* annstr = dyn_cast<GlobalVariable>(cs->getOperand(1)->stripPointerCasts());
        if (!annstr || !annstr->hasInitializer()) continue;
        auto* cds = dyn_cast<ConstantDataSequential>(annstr->getInitializer());
        if (!cds || !cds->isString()) continue;
        if (strcmp(cds->getAsString().str().c_str(), tag) == 0) return true;
    }
    return false;
}

static bool HasObfuscateAnnotation(Function& f)
{
    return HasAnnotationTag(f, "obfus");
}

static bool AnyFunctionAnnotated(Module& m)
{
    for (Function& f : m)
    {
        if (f.isDeclaration()) continue;
        if (HasObfuscateAnnotation(f)) return true;
    }
    return false;
}

static uint64_t PickSeed()
{
    std::random_device rd;
    uint64_t s = static_cast<uint64_t>(rd()) ^ (static_cast<uint64_t>(rd()) << 32);
    s ^= static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    return s;
}

static void DrawExtWrapAxes(WbModuleState& s, std::mt19937_64& rng)
{
    s.extwrapcount = static_cast<uint8_t>(1u + (rng() % static_cast<uint64_t>(kExtWrapMaxCount)));
    s.extwrapdepth = static_cast<uint8_t>(1u + (rng() % static_cast<uint64_t>(kExtWrapMaxDepth)));
    constexpr uint32_t familysizes[] = {64u, 128u, 256u, 512u, 1024u};
    s.extwrapfamilysize = familysizes[rng() % 5u];

    std::array<uint8_t, 8> allpositions = {0, 1, 2, 3, 4, 5, 6, 7};
    for (int i = 7; i > 0; --i)
    {
        const int j = static_cast<int>(rng() % static_cast<uint64_t>(i + 1));
        std::swap(allpositions[i], allpositions[j]);
    }
    s.extwrappositions.fill(0);
    for (uint8_t pidx = 0; pidx < s.extwrapcount; ++pidx)
        s.extwrappositions[pidx] = allpositions[pidx];
}

static void GenerateExtWrapFamilies(WbModuleState& s, std::mt19937_64& rng)
{
    const size_t totalentries = static_cast<size_t>(s.extwrapcount) * static_cast<size_t>(s.extwrapdepth) * static_cast<size_t>(s.extwrapfamilysize);
    s.extfwdperms.resize(totalentries);

    const size_t famstride = static_cast<size_t>(s.extwrapfamilysize);
    const size_t depthstride = famstride * s.extwrapdepth;
    for (uint8_t pidx = 0; pidx < s.extwrapcount; ++pidx)
    {
        for (uint8_t d = 0; d < s.extwrapdepth; ++d)
        {
            for (uint32_t e = 0; e < s.extwrapfamilysize; ++e)
            {
                const size_t fwdslot = static_cast<size_t>(pidx) * depthstride + static_cast<size_t>(d) * famstride + static_cast<size_t>(e);
                auto& entry = s.extfwdperms[fwdslot];
                for (int i = 0; i < 256; ++i)
                    entry.fwd[i] = static_cast<uint8_t>(i);
                for (int i = 255; i > 0; --i)
                {
                    const int j = static_cast<int>(rng() % static_cast<uint64_t>(i + 1));
                    std::swap(entry.fwd[i], entry.fwd[j]);
                }

                std::array<uint8_t, 256> inv{};
                for (int i = 0; i < 256; ++i)
                    inv[entry.fwd[i]] = static_cast<uint8_t>(i);
                const size_t base = static_cast<size_t>(s.offextwrapregion) + static_cast<size_t>(pidx) * depthstride * 256u + static_cast<size_t>(d) * famstride * 256u + static_cast<size_t>(e) * 256u;
                for (int i = 0; i < 256; ++i)
                    s.packedcipher[base + i] = inv[i];
            }
        }
    }
}

static void GeneratePreencBank(WbModuleState& s)
{
    const auto invshift = InvertShift(s.shift);
    const bool freshroundenc = s.freshroundenc;
    for (int r = 0; r < s.rounds; ++r)
    {
        uint8_t rkbytes[8];
        PackU64LE(rkbytes, s.roundkeys[r]);

        const uint8_t rotamount = s.bitrot[r];
        const auto& roundsbox = s.hasperroundsboxes ? s.perroundsboxes[r] : s.sbox;

        const uint8_t rt = s.roundprogram[r];
        const uint8_t fg = s.fusion[r];
        const bool isb = (rt == 1);
        const bool isf = (rt == 4);
        const bool isg = (rt == 5);
        const uint8_t groute = s.groute[r];
        const bool bakesoutputark = (rt == 0 || rt == 4 || rt == 5) && (fg != 1);

        for (int i = 0; i < 8; ++i)
        {
            const int c = i / 4;
            const int j = i % 4;

            for (int decoded = 0; decoded < 256; ++decoded)
            {
                const uint8_t presbox = isb ? static_cast<uint8_t>(decoded ^ rkbytes[i]) : static_cast<uint8_t>(decoded);
                uint8_t subbed;
                if (isf)
                    subbed = presbox;
                else if (isg && ((groute >> i) & 1u) != 0u)
                    subbed = s.gsboxb[r][presbox];
                else
                    subbed = roundsbox[presbox];
                if (rotamount != 0)
                    subbed = RotateLeft8(subbed, rotamount);

                uint64_t entry = 0;
                for (int b = 0; b < 4; ++b)
                {
                    const int postshiftpos = invshift[c * 4 + b];
                    const uint8_t coeff = s.mds[b][j];
                    uint8_t contribb = GFMul(coeff, subbed);
                    if (bakesoutputark && j == 0)
                        contribb ^= rkbytes[postshiftpos];
                    const uint8_t topack = freshroundenc ? contribb : s.enc.roundenc[r].perm[postshiftpos][contribb];
                    entry |= static_cast<uint64_t>(topack) << (postshiftpos * 8);
                }

                const size_t off = static_cast<size_t>(s.offpreencbank) + (static_cast<size_t>(r) * 8 + i) * 256 * 8 + static_cast<size_t>(decoded) * 8;
                for (int k = 0; k < 8; ++k)
                    s.packedcipher[off + k] = static_cast<uint8_t>(entry >> (k * 8));
            }
        }
    }
}

static uint64_t ApplyExtEncFwd(const WbModuleState& s, const std::array<std::array<uint16_t, kExtWrapMaxDepth>, kExtWrapMaxCount>& familyindices, uint64_t k)
{
    uint64_t result = k;
    const size_t famstride = static_cast<size_t>(s.extwrapfamilysize);
    const size_t depthstride = famstride * s.extwrapdepth;
    for (uint8_t pidx = 0; pidx < s.extwrapcount; ++pidx)
    {
        const uint8_t pos = s.extwrappositions[pidx];
        uint8_t byte = static_cast<uint8_t>((result >> (pos * 8u)) & 0xFFu);
        for (uint8_t d = 0; d < s.extwrapdepth; ++d)
        {
            const uint32_t fam = familyindices[pidx][d];
            byte = s.extfwdperms[static_cast<size_t>(pidx) * depthstride + static_cast<size_t>(d) * famstride + static_cast<size_t>(fam)].fwd[byte];
        }
        result &= ~(static_cast<uint64_t>(0xFFu) << (pos * 8u));
        result |= static_cast<uint64_t>(byte) << (pos * 8u);
    }
    return result;
}

static void BuildCipherState(WbModuleState& s)
{
    s.seed = PickSeed();
    std::mt19937_64 rng(s.seed);

    s.spec = WbSpec::MakeBaseline();
    s.spec.specseed = static_cast<uint32_t>(s.seed);

    constexpr unsigned roundsbase = static_cast<unsigned>(kBaseRounds);
    static_assert(static_cast<unsigned>(kMaxRounds) >= roundsbase, "kMaxRounds must be at least kBaseRounds");
    const unsigned roundsrange = static_cast<unsigned>(kMaxRounds) - roundsbase + 1u;
    s.rounds = static_cast<uint8_t>(roundsbase + (rng() % roundsrange));
    s.spec.rounds = s.rounds;

    constexpr unsigned blockbytesbase = static_cast<unsigned>(kBaseBlockBytes);
    constexpr unsigned blockbytescap = static_cast<unsigned>(kMaxBlockBytes);
    static_assert(blockbytescap >= blockbytesbase, "kMaxBlockBytes must be at least kBaseBlockBytes");
    const unsigned bbpick = static_cast<unsigned>(rng()) & 1u;
    s.blockbytes = static_cast<uint8_t>(bbpick ? blockbytescap : blockbytesbase);
    s.spec.blockbytes = s.blockbytes;
    s.spec.columns = static_cast<uint8_t>(s.blockbytes / 4);
    const bool bbwide = (static_cast<unsigned>(s.blockbytes) > blockbytesbase);

    // ctor regen IR hard-codes 8-byte state throughout at bb=16 the freshmath pipeline would produce garbage, so force disable here before anything downstream reads these flags
    if (bbwide)
    {
        s.freshinenc = false;
        s.freshroundenc = false;
        s.freshciphermath = false;
    }

    s.sbox = GenerateSBox(rng);
    s.invsbox = InvertSBox(s.sbox);
    for (int r = 0; r < s.rounds; ++r)
        s.roundkeys[r] = rng();
    s.roundkeyshi.fill(0);
    if (bbwide)
    {
        for (int r = 0; r < s.rounds; ++r)
            s.roundkeyshi[r] = rng();
    }

    {
        const uint32_t pick = static_cast<uint32_t>(rng()) % 3u;
        switch (pick)
        {
            case 0:
                s.spec.mdsfamily = MDSFamily::Cauchy;
                s.mds = MakeCauchyMDS(rng);
                break;
            case 1:
                s.spec.mdsfamily = MDSFamily::Vandermonde;
                s.mds = MakeVandermondeMDS(rng);
                break;
            case 2:
            default:
                s.spec.mdsfamily = MDSFamily::Hadamard;
                s.mds = MakeHadamardMDS(rng);
                break;
        }
        s.invmds = InvertMDS(s.mds);
    }

    s.bitrot.fill(0);
    {
        // AESAnalog is hardcoded to 8-byte state so at bb=16 restrict the draw to {Random, BitRotBaked} both go through GenerateShift which handles bb-sized perms
        const uint32_t pickmod = bbwide ? 2u : 3u;
        const uint32_t rawpick = static_cast<uint32_t>(rng()) % pickmod;
        const uint32_t pick = bbwide ? (rawpick == 1 ? 2u : 0u) : rawpick;
        switch (pick)
        {
            case 0:
                s.spec.shiftkind = ShiftKind::Random;
                s.shift = GenerateShift(rng, static_cast<int>(s.blockbytes));
                break;
            case 1:
                s.spec.shiftkind = ShiftKind::AESAnalog;
                s.shift = MakeAESAnalogShift();
                break;
            case 2:
            default:
                s.spec.shiftkind = ShiftKind::BitRotBaked;
                s.shift = GenerateShift(rng, static_cast<int>(s.blockbytes));
                for (int r = 0; r < s.rounds; ++r)
                {
                    s.bitrot[r] = static_cast<uint8_t>(rng() & 7u);
                    s.spec.bitrot[r] = s.bitrot[r];
                }
                break;
        }
    }
    s.invshift = InvertShift(s.shift, static_cast<int>(s.blockbytes));

    s.hasperroundsboxes = false;
    {
        const uint32_t pick = static_cast<uint32_t>(rng()) % 2u;
        if (pick == 1)
        {
            s.spec.sboxscheme = SBoxScheme::PerRoundIndependent;
            for (int r = 0; r < s.rounds; ++r)
            {
                s.perroundsboxes[r] = GenerateSBox(rng);
                s.perroundinvsboxes[r] = InvertSBox(s.perroundsboxes[r]);
            }
            s.hasperroundsboxes = true;
        }
    }

    s.roundprogram.fill(0);
    {
        RoundType tmp[kMaxRounds]{};
        DrawRoundProgram(rng, s.rounds, tmp);
        for (int r = 0; r < s.rounds; ++r)
        {
            s.roundprogram[r] = static_cast<uint8_t>(tmp[r]);
            s.spec.roundprogram[r] = tmp[r];
        }
    }

    s.gsboxb.fill(SBox{});
    s.ginvsboxb.fill(SBox{});
    s.groute.fill(0);
    s.anyground = false;
    s.groutepacked = 0;
    for (int r = 0; r < s.rounds; ++r)
    {
        if (s.roundprogram[r] == 5)
        {
            s.gsboxb[r] = GenerateSBox(rng);
            s.ginvsboxb[r] = InvertSBox(s.gsboxb[r]);
            s.groute[r] = static_cast<uint8_t>(rng() & 0xFFu);
            s.anyground = true;
        }
        s.groutepacked |= static_cast<uint64_t>(s.groute[r]) << (r * 8);
    }

    // ARKOut's ARK table is encoded with roundenc, so when roundenc regen is on without full fresh math the ARKTable would be stale against the regend roundenc; force Full fusion in that intermediate state
    const bool arkoutsafe = s.freshciphermath || !s.freshroundenc;
    s.fusion.fill(0);
    s.anyarkout = false;
    if (arkoutsafe)
    {
        for (int r = 0; r < s.rounds; ++r)
        {
            if (s.roundprogram[r] == 0)
            {
                const uint32_t pick = static_cast<uint32_t>(rng()) & 1u;
                s.fusion[r] = static_cast<uint8_t>(pick);
                if (pick == 1) s.anyarkout = true;
            }
            s.spec.fusion[r] = static_cast<FusionGranularity>(s.fusion[r]);
        }
    }

    s.enc = GenerateEncodings(rng, static_cast<int>(s.blockbytes));

    const uint8_t* bitrotptr = (s.spec.shiftkind == ShiftKind::BitRotBaked) ? s.bitrot.data() : nullptr;
    const std::array<SBox, kMaxRounds>* prsptr = s.hasperroundsboxes ? &s.perroundsboxes : nullptr;
    const uint8_t* rpptr = s.roundprogram.data();
    const uint8_t* fgptr = s.fusion.data();
    const std::array<SBox, kMaxRounds>* gsbptr = s.anyground ? &s.gsboxb : nullptr;
    const uint8_t* grtptr = s.anyground ? s.groute.data() : nullptr;
    const std::array<uint64_t, kMaxRounds>* rkhiptr = bbwide ? &s.roundkeyshi : nullptr;

    auto t = std::make_unique<TBoxes>();
    auto x = std::make_unique<XorTables>();
    BuildTBoxesInto(*t, s.sbox, s.roundkeys, s.enc, s.shift, s.mds, bitrotptr, prsptr, s.rounds, rpptr, fgptr, gsbptr, grtptr, static_cast<int>(s.blockbytes), rkhiptr);
    BuildXorTablesInto(*x, s.enc, s.rounds, static_cast<int>(s.blockbytes));

    std::unique_ptr<ARKTables> atables;
    if (s.anyarkout)
    {
        atables = std::make_unique<ARKTables>();
        BuildARKTablesInto(*atables, s.roundkeys, s.enc, fgptr, s.rounds, static_cast<int>(s.blockbytes), rkhiptr);
    }

    s.packedcipher = PackWhiteboxCipherData(s.enc, *t, *x, s.rounds, atables ? atables.get() : nullptr, static_cast<int>(s.blockbytes));

    s.packedcipher.resize(kCipherDataSizeWithSeed, 0);
    s.seedglobal = rng();
    for (int i = 0; i < 8; ++i)
        s.packedcipher[kSeedOffset + i] = static_cast<uint8_t>(s.seedglobal >> (i * 8));

    s.offextwrapregion = kCipherDataSizeWithSeed;
    DrawExtWrapAxes(s, rng);
    const uint32_t extwrapbytes = static_cast<uint32_t>(s.extwrapcount) * static_cast<uint32_t>(s.extwrapdepth) * s.extwrapfamilysize * 256u;
    s.offpreencbank = s.offextwrapregion + extwrapbytes;
    s.offinvpermsperround = s.offpreencbank + kPreencBankBytes;
    s.offgfmultbl = s.offinvpermsperround + kInvPermsPerRoundBytes;
    s.offfreshinvsbox = s.offgfmultbl + kGFMulTblBytes;
    s.offfreshrk = s.offfreshinvsbox + kFreshInvSBoxBytes;
    s.offsboxgb = s.offfreshrk + kFreshRkBytes;
    s.offinvsboxgb = s.offsboxgb + kSBoxGBBytes;

    s.packedcipher.resize(s.offextwrapregion + extwrapbytes, 0);
    GenerateExtWrapFamilies(s, rng);

    if (s.freshinenc)
    {
        const uint32_t total = (s.freshroundenc && s.freshciphermath) ? (s.offinvsboxgb + kInvSBoxGBBytes) : (s.offinvpermsperround + kInvPermsPerRoundBytes);
        s.packedcipher.resize(total, 0);
        GeneratePreencBank(s);

        if (s.freshroundenc && s.freshciphermath)
        {
            for (uint32_t a = 0; a < 256; ++a)
            {
                for (uint32_t b = 0; b < 256; ++b)
                    s.packedcipher[s.offgfmultbl + (a << 8) + b] = GFMul(static_cast<uint8_t>(a), static_cast<uint8_t>(b));
            }

            // nonParallelSBoxes rounds leave sboxgb slots zero on purpose safe only because groute byte is also zero at those rounds, so the routing-gated select in the Tbox rebuild IR never picks the zero slot
            if (s.anyground)
            {
                for (int r = 0; r < s.rounds; ++r)
                {
                    if (s.roundprogram[r] != 5) continue;
                    const size_t fwdbase = static_cast<size_t>(s.offsboxgb) + static_cast<size_t>(r) * 256u;
                    const size_t invbase = static_cast<size_t>(s.offinvsboxgb) + static_cast<size_t>(r) * 256u;
                    for (int i = 0; i < 256; ++i)
                    {
                        s.packedcipher[fwdbase + i] = s.gsboxb[r][i];
                        s.packedcipher[invbase + i] = s.ginvsboxb[r][i];
                    }
                }
            }
        }

        for (int r = 0; r < s.rounds; ++r)
        {
            const auto& invsrc = (r == 0) ? s.enc.inenc.invperm : s.enc.roundenc[r - 1].invperm;
            for (int i = 0; i < 8; ++i)
            {
                const size_t base = static_cast<size_t>(s.offinvpermsperround) + static_cast<size_t>(r) * kInvPermsPerRoundRowStride + static_cast<size_t>(i) * 256u;
                for (int xx = 0; xx < 256; ++xx)
                    s.packedcipher[base + xx] = invsrc[i][xx];
            }
        }
    }
}

static GlobalVariable* InstallByteArray(Module& m, StringRef name, size_t size, const uint8_t* bytesornull, bool isconstant)
{
    LLVMContext& c = m.getContext();
    ArrayType* ty = ArrayType::get(Type::getInt8Ty(c), size);
    Constant* init = bytesornull ? static_cast<Constant*>(ConstantDataArray::get(c, ArrayRef<uint8_t>(bytesornull, size))) : static_cast<Constant*>(ConstantAggregateZero::get(ty));
    auto* gv = new GlobalVariable(m, ty, isconstant, GlobalValue::InternalLinkage, init, name);
    gv->setAlignment(MaybeAlign(64));
    return gv;
}

static Function* EmitCipherFunction(Module& m, const WbModuleState& s)
{
    LLVMContext& c = m.getContext();
    Type* i8 = Type::getInt8Ty(c);
    Type* i64 = Type::getInt64Ty(c);

    const int bb = static_cast<int>(s.blockbytes);
    const bool bbwide = (bb > 8);

    FunctionType* ft = FunctionType::get(i64, {i64, i64}, false);
    Function* f = Function::Create(ft, GlobalValue::InternalLinkage, "__wb_cipher", &m);
    f->addFnAttr(Attribute::NoInline);
    f->addFnAttr(Attribute::NoUnwind);

    BasicBlock* entry = BasicBlock::Create(c, "Entry", f);
    IRBuilder<> b(entry);

    GlobalVariable* gvs = s.gvlive;
    Value* wbinlo = f->getArg(0);
    Value* wbinhi = f->getArg(1);

    uint8_t contrib[kMaxBlockBytes];
    for (int p = 0; p < bb; ++p)
        contrib[p] = static_cast<uint8_t>(s.shift[p] / 4);

    auto getByteFromHalves = [&](Value* vlo, Value* vhi, int p) -> Value*
    {
        Value* v = (p < 8) ? vlo : vhi;
        const int shiftamt = (p < 8) ? p : (p - 8);
        if (shiftamt != 0) v = b.CreateLShr(v, b.getInt64(shiftamt * 8));
        return b.CreateAnd(v, b.getInt64(0xFF));
    };
    auto packByteIntoHalf = [&](Value* sofar, Value* byteval, int pinhalf) -> Value*
    {
        Value* asi64 = byteval;
        if (asi64->getType()->isIntegerTy(8))
            asi64 = b.CreateZExt(asi64, i64);
        if (pinhalf != 0)
            asi64 = b.CreateShl(asi64, b.getInt64(pinhalf * 8));
        return b.CreateOr(sofar, asi64);
    };
    auto byteGep = [&](uint64_t off) -> Value*
    {
        return b.CreateInBoundsGEP(i8, gvs, b.getInt64(off));
    };

    auto applyBytePermToHalves = [&](Value* statelo, Value* statehi, uint32_t baseoff, const Twine& tag, Value*& outlo, Value*& outhi)
    {
        outlo = b.getInt64(0);
        outhi = b.getInt64(0);
        for (int i = 0; i < bb; ++i)
        {
            Value* row = byteGep(baseoff + i * 256);
            Value* bytein = getByteFromHalves(statelo, statehi, i);
            Value* cell = b.CreateInBoundsGEP(i8, row, bytein);
            Value* byteout = b.CreateAlignedLoad(i8, cell, Align(1), tag + "B" + Twine(i));
            if (i < 8)
                outlo = packByteIntoHalf(outlo, byteout, i);
            else
                outhi = packByteIntoHalf(outhi, byteout, i - 8);
        }
    };

    auto tboxLookup = [&](int r, int i, Value* statebytei64, Value*& entrylo, Value*& entryhi)
    {
        const uint32_t rowoff = kTBoxOffset + static_cast<uint32_t>(r) * kTBoxRoundStride + static_cast<uint32_t>(i) * kTBoxRowStride;
        Value* rowbase = byteGep(rowoff);
        Value* entryoff = b.CreateMul(statebytei64, b.getInt64(static_cast<uint64_t>(kTBoxEntryStride)));
        Value* entryptr = b.CreateInBoundsGEP(i8, rowbase, entryoff);
        entrylo = b.CreateAlignedLoad(i64, entryptr, Align(8), "TLoR" + Twine(r) + "I" + Twine(i));
        if (bbwide)
        {
            Value* entryhiptr = b.CreateInBoundsGEP(i8, entryptr, b.getInt64(8));
            entryhi = b.CreateAlignedLoad(i64, entryhiptr, Align(8), "THiR" + Twine(r) + "I" + Twine(i));
        }
        else
        {
            entryhi = b.getInt64(0);
        }
    };

    auto xorLookup = [&](int r, int p, Value* ai64, Value* bi64) -> Value*
    {
        const uint32_t off = kXorTblOffset + static_cast<uint32_t>(r) * kXorTblRoundStride + static_cast<uint32_t>(p) * kXorTblTableStride;
        Value* base = byteGep(off);
        Value* idx = b.CreateAdd(b.CreateShl(ai64, b.getInt64(8)), bi64);
        Value* cell = b.CreateInBoundsGEP(i8, base, idx);
        return b.CreateAlignedLoad(i8, cell, Align(1), "XR" + Twine(r) + "P" + Twine(p));
    };

    auto arkLookup = [&](int r, int p, Value* yi64) -> Value*
    {
        const uint32_t off = kARKTblOffset + static_cast<uint32_t>(r) * static_cast<uint32_t>(kMaxBlockBytes) * 256u + static_cast<uint32_t>(p) * 256u;
        Value* base = byteGep(off);
        Value* cell = b.CreateInBoundsGEP(i8, base, yi64);
        return b.CreateAlignedLoad(i8, cell, Align(1), "ARKR" + Twine(r) + "P" + Twine(p));
    };

    Value* statelo = nullptr;
    Value* statehi = nullptr;
    applyBytePermToHalves(wbinlo, wbinhi, kInEncOffset, "InEnc", statelo, statehi);

    for (int r = 0; r < s.rounds; ++r)
    {
        Value* clo[kMaxBlockBytes];
        Value* chi[kMaxBlockBytes];
        for (int i = 0; i < bb; ++i)
        {
            Value* sb = getByteFromHalves(statelo, statehi, i);
            tboxLookup(r, i, sb, clo[i], chi[i]);
        }

        Value* newlo = b.getInt64(0);
        Value* newhi = b.getInt64(0);
        const bool isarkout = (s.fusion[r] == 1);
        for (int p = 0; p < bb; ++p)
        {
            const int cc = contrib[p];
            auto extract = [&](int bi) -> Value*
            {
                Value* contribv = (p < 8) ? clo[cc * 4 + bi] : chi[cc * 4 + bi];
                const int shiftamt = (p < 8) ? (p * 8) : ((p - 8) * 8);
                Value* shifted = contribv;
                if (shiftamt != 0)
                    shifted = b.CreateLShr(contribv, b.getInt64(shiftamt));
                return b.CreateAnd(shifted, b.getInt64(0xFF));
            };
            Value* b0 = extract(0);
            Value* b1 = extract(1);
            Value* b2 = extract(2);
            Value* b3 = extract(3);
            Value* s1 = xorLookup(r, p, b0, b1);
            Value* s1i64 = b.CreateZExt(s1, i64);
            Value* s2 = xorLookup(r, p, s1i64, b2);
            Value* s2i64 = b.CreateZExt(s2, i64);
            Value* s3 = xorLookup(r, p, s2i64, b3);
            if (isarkout)
            {
                Value* s3i64 = b.CreateZExt(s3, i64);
                s3 = arkLookup(r, p, s3i64);
            }
            if (p < 8)
                newlo = packByteIntoHalf(newlo, s3, p);
            else
                newhi = packByteIntoHalf(newhi, s3, p - 8);
        }
        statelo = newlo;
        statehi = newhi;
    }

    Value* outlo = nullptr;
    Value* outhi = nullptr;
    applyBytePermToHalves(statelo, statehi, kOutEncInvOffset, "OutEncInv", outlo, outhi);

    (void)outhi;
    b.CreateRet(outlo);
    return f;
}

static void CollectCandidatesInFunction(Function& f, std::vector<WbCandidate>& out)
{
    for (BasicBlock& bb : f)
    {
        for (Instruction& i : bb)
        {
            if (auto* cb = dyn_cast<CallBase>(&i))
            {
                if (cb->isInlineAsm()) continue;
                if (cb->getIntrinsicID() != Intrinsic::not_intrinsic) continue;
                for (unsigned a = 0; a < cb->arg_size(); ++a)
                {
                    if (isa<ConstantInt>(cb->getArgOperand(a)))
                        out.push_back({&i, a, true});
                }
                continue;
            }
            if (isa<BinaryOperator>(&i) || isa<ICmpInst>(&i))
            {
                for (unsigned u = 0; u < i.getNumOperands(); ++u)
                {
                    if (isa<ConstantInt>(i.getOperand(u)))
                        out.push_back({&i, u, false});
                }
                continue;
            }
            if (auto* si = dyn_cast<StoreInst>(&i))
            {
                if (isa<ConstantInt>(si->getValueOperand()))
                    out.push_back({&i, 0, false});
                continue;
            }
            if (auto* sel = dyn_cast<SelectInst>(&i))
            {
                if (isa<ConstantInt>(sel->getTrueValue()))
                    out.push_back({&i, 1, false});
                if (isa<ConstantInt>(sel->getFalseValue()))
                    out.push_back({&i, 2, false});
                continue;
            }
            if (auto* ri = dyn_cast<ReturnInst>(&i))
            {
                Value* rv = ri->getReturnValue();
                if (rv && isa<ConstantInt>(rv))
                    out.push_back({&i, 0, false});
                continue;
            }
        }
    }
}

static bool ClassifyCandidate(const WbCandidate& cand, WbSite& out, uint32_t slotid, const WbModuleState& s, std::mt19937_64& siterng)
{
    Value* op = cand.iscallarg ? cast<CallBase>(cand.user)->getArgOperand(cand.opidx) : cand.user->getOperand(cand.opidx);
    auto* ci = dyn_cast<ConstantInt>(op);
    if (!ci) return false;

    auto* it = dyn_cast<IntegerType>(ci->getType());
    if (!it) return false;
    const unsigned bw = it->getBitWidth();
    if (bw != 8 && bw != 16 && bw != 32 && bw != 64) return false;

    const uint64_t k = ci->getZExtValue();
    if (k == 0) return false;

    out.f = cand.user->getFunction();
    out.user = cand.user;
    out.opidx = cand.opidx;
    out.iscallarg = cand.iscallarg;
    out.bitwidth = bw;
    out.k = k;
    out.slotid = slotid;
    out.extencon = true;

    uint64_t kprepared = k;

    const uint8_t* bitrotptr = (s.spec.shiftkind == ShiftKind::BitRotBaked) ? s.bitrot.data() : nullptr;
    const std::array<SBox, kMaxRounds>* invprsptr = s.hasperroundsboxes ? &s.perroundinvsboxes : nullptr;
    const uint8_t* rpptr = s.roundprogram.data();
    const std::array<SBox, kMaxRounds>* ginvsbptr = s.anyground ? &s.ginvsboxb : nullptr;
    const uint8_t* grtptr = s.anyground ? s.groute.data() : nullptr;

    const bool freshmathon = s.freshinenc && s.freshroundenc && s.freshciphermath;

    const bool bbwide = (static_cast<unsigned>(s.blockbytes) > static_cast<unsigned>(kBaseBlockBytes));
    const std::array<uint64_t, kMaxRounds>* rkhiptr = bbwide ? &s.roundkeyshi : nullptr;

    out.betasite = siterng();
    out.alphainit = siterng();
    if (bbwide)
    {
        out.betasitehi = siterng();
        out.alphainithi = siterng();
        out.wideningpad = siterng();
    }
    for (uint8_t pidx = 0; pidx < s.extwrapcount; ++pidx)
        for (uint8_t d = 0; d < s.extwrapdepth; ++d)
            out.familyindices[pidx][d] = static_cast<uint16_t>(siterng() % s.extwrapfamilysize);
    kprepared = ApplyExtEncFwd(s, out.familyindices, k);

    if (freshmathon)
    {
        // when freshmath is on the ctor runs SPNDecrypt in IR against fresh round keys ship the Decrypt INPUT (k_prepared ^ beta) here, not the output
        out.bct = kprepared ^ out.betasite;
    }
    else
    {
        uint64_t decrypthiin = bbwide ? (out.wideningpad ^ out.betasitehi) : 0u;
        uint64_t decrypthiout = 0;
        out.bct = SPNDecrypt(kprepared ^ out.betasite, s.invsbox, s.roundkeys, s.invshift, s.invmds, bitrotptr, invprsptr, s.rounds, rpptr, ginvsbptr, grtptr, static_cast<int>(s.blockbytes), decrypthiin, rkhiptr, &decrypthiout);
        if (bbwide) out.bcthi = decrypthiout;
    }
    out.maskedinit = out.bct ^ out.alphainit;
    if (bbwide) out.maskedinithi = out.bcthi ^ out.alphainithi;

    return true;
}

static void RewriteSite(Module& m, const WbModuleState& s, const WbSite& site)
{
    LLVMContext& c = m.getContext();
    Type* i8 = Type::getInt8Ty(c);
    Type* i64 = Type::getInt64Ty(c);
    Type* targetty = IntegerType::get(c, site.bitwidth);

    IRBuilder<> b(site.user);

    const bool bbwide = (static_cast<unsigned>(s.blockbytes) > static_cast<unsigned>(kBaseBlockBytes));
    const uint64_t rw0stride = static_cast<uint64_t>(Rw0SlotBytes(s));

    Value* ptencoded = nullptr;

    {
        const uint64_t mlooff = site.slotid * rw0stride + 0;
        const uint64_t alooff = site.slotid * rw0stride + (bbwide ? 16u : 8u);
        const uint64_t mhioff = site.slotid * rw0stride + 8u;
        const uint64_t ahioff = site.slotid * rw0stride + 24u;

        Value* rw0mlo = b.CreateInBoundsGEP(i8, s.gvrw0, b.getInt64(mlooff), "WbRw0MLoPtr");
        Value* rw0alo = b.CreateInBoundsGEP(i8, s.gvrw0, b.getInt64(alooff), "WbRw0ALoPtr");
        Value* rw0mhi = nullptr;
        Value* rw0ahi = nullptr;
        if (bbwide)
        {
            rw0mhi = b.CreateInBoundsGEP(i8, s.gvrw0, b.getInt64(mhioff), "WbRw0MHiPtr");
            rw0ahi = b.CreateInBoundsGEP(i8, s.gvrw0, b.getInt64(ahioff), "WbRw0AHiPtr");
        }

        LoadInst* maskedlo = b.CreateAlignedLoad(i64, rw0mlo, Align(8), "WbMaskedLo");
        maskedlo->setVolatile(true);
        LoadInst* alphalo = b.CreateAlignedLoad(i64, rw0alo, Align(8), "WbAlphaLo");
        alphalo->setVolatile(true);
        Value* maskedhi = b.getInt64(0);
        Value* alphahi = b.getInt64(0);
        if (bbwide)
        {
            LoadInst* mh = b.CreateAlignedLoad(i64, rw0mhi, Align(8), "WbMaskedHi");
            mh->setVolatile(true);
            maskedhi = mh;
            LoadInst* ah = b.CreateAlignedLoad(i64, rw0ahi, Align(8), "WbAlphaHi");
            ah->setVolatile(true);
            alphahi = ah;
        }

        Value* wbinlo = b.CreateXor(maskedlo, alphalo, "WbInLo");
        Value* wbinhi = b.CreateXor(maskedhi, alphahi, "WbInHi");

        // @__Wbcipher returns a plain i64 (low half only) under signature i64(i64, i64) no sret, no alloca, no aggregate return; at bb=8 wbinhi is a constant zero and LLVM folds the dead arg into an `xor edx, edx` at the site
        Value* ptmaskedlo = b.CreateCall(s.fncipher, {wbinlo, wbinhi}, "WbPtMaskedLo");

        ptencoded = b.CreateXor(ptmaskedlo, b.getInt64(site.betasite), "WbPtEncoded");

        Value* seedptr = b.CreateInBoundsGEP(i8, s.gvlive, b.getInt64(kSeedOffset), "WbSeedPtr");
        LoadInst* seed = b.CreateAlignedLoad(i64, seedptr, Align(8), "WbSeed");
        seed->setVolatile(true);

        auto staffordMix13 = [&](Value* x) -> Value*
        {
            Value* s0 = b.CreateLShr(x, b.getInt64(33));
            Value* x0 = b.CreateXor(x, s0);
            Value* m0 = b.CreateMul(x0, b.getInt64(kMixerMul1));
            Value* s1 = b.CreateLShr(m0, b.getInt64(33));
            Value* x1 = b.CreateXor(m0, s1);
            Value* m1 = b.CreateMul(x1, b.getInt64(kMixerMul2));
            Value* s2 = b.CreateLShr(m1, b.getInt64(33));
            return b.CreateXor(m1, s2);
        };

        Value* mixlo = b.CreateXor(alphalo, seed, "WbMixLo");
        Value* alphanextlo = staffordMix13(mixlo);
        Value* alphanexthi = b.getInt64(0);
        if (bbwide)
        {
            Value* mixhi = b.CreateXor(alphahi, seed, "WbMixHi");
            alphanexthi = staffordMix13(mixhi);
        }

        Value* tmplo = b.CreateXor(maskedlo, alphalo);
        Value* maskednewlo = b.CreateXor(tmplo, alphanextlo, "WbMaskedNewLo");
        Value* maskednewhi = b.getInt64(0);
        if (bbwide)
        {
            Value* tmphi = b.CreateXor(maskedhi, alphahi);
            maskednewhi = b.CreateXor(tmphi, alphanexthi, "WbMaskedNewHi");
        }

        StoreInst* simlo = b.CreateAlignedStore(maskednewlo, rw0mlo, Align(8));
        simlo->setVolatile(true);
        StoreInst* sialo = b.CreateAlignedStore(alphanextlo, rw0alo, Align(8));
        sialo->setVolatile(true);
        if (bbwide)
        {
            StoreInst* simhi = b.CreateAlignedStore(maskednewhi, rw0mhi, Align(8));
            simhi->setVolatile(true);
            StoreInst* siahi = b.CreateAlignedStore(alphanexthi, rw0ahi, Align(8));
            siahi->setVolatile(true);
        }
    }

    Value* ki64 = ptencoded;

    {
        const uint32_t depthstridebytes = static_cast<uint32_t>(s.extwrapdepth) * s.extwrapfamilysize * 256u;
        const uint32_t famstridebytes = s.extwrapfamilysize * 256u;
        for (uint8_t pidx = 0; pidx < s.extwrapcount; ++pidx)
        {
            const uint8_t pos = s.extwrappositions[pidx];
            // cast pos / pidx / d to unsigned before Twine. Twine(unsigned char) treats its argument as a CHARACTER byte so pos==0 encodes as NUL, which LLVM then rejects with "Invalid value name" at bitcode read time
            const unsigned posu = static_cast<unsigned>(pos);
            const unsigned pidxu = static_cast<unsigned>(pidx);

            Value* bytev = ki64;
            if (pos != 0)
                bytev = b.CreateLShr(ki64, b.getInt64(pos * 8u));
            bytev = b.CreateAnd(bytev, b.getInt64(0xFFu), Twine("WbB") + Twine(posu));

            // strip the chain in REVERSE depth order (d = depth-1 .. 0) to undo the forward chain the classifier applied at d = 0 .. depth-1
            for (int d = static_cast<int>(s.extwrapdepth) - 1; d >= 0; --d)
            {
                const unsigned du = static_cast<unsigned>(d);
                const uint32_t famidx = site.familyindices[pidx][d];
                const uint32_t invbaseoff = s.offextwrapregion + static_cast<uint32_t>(pidx) * depthstridebytes + static_cast<uint32_t>(d) * famstridebytes + famidx * 256u;
                Value* invbase = b.CreateInBoundsGEP(i8, s.gvlive, b.getInt64(invbaseoff), Twine("WbHInvBaseP") + Twine(pidxu) + Twine("D") + Twine(du));
                Value* invcell = b.CreateInBoundsGEP(i8, invbase, bytev);
                LoadInst* newbytei8 = b.CreateAlignedLoad(i8, invcell, Align(1), Twine("WbHInvP") + Twine(pidxu) + Twine("D") + Twine(du));
                bytev = b.CreateZExt(newbytei8, i64);
            }

            const uint64_t clearmask = ~(static_cast<uint64_t>(0xFFu) << (pos * 8u));
            Value* cleared = b.CreateAnd(ki64, b.getInt64(clearmask));
            Value* nbi64 = bytev;
            if (pos != 0)
                nbi64 = b.CreateShl(nbi64, b.getInt64(pos * 8u));
            ki64 = b.CreateOr(cleared, nbi64, Twine("WbPtP") + Twine(pidxu));
        }
    }

    Value* k = (site.bitwidth == 64) ? ki64 : static_cast<Value*>(b.CreateTrunc(ki64, targetty, "WbPt"));

    if (site.iscallarg)
        cast<CallBase>(site.user)->setArgOperand(site.opidx, k);
    else
        site.user->setOperand(site.opidx, k);
}

static void WriteU64LE(std::vector<uint8_t>& out, size_t off, uint64_t v)
{
    for (int i = 0; i < 8; ++i)
        out[off + i] = static_cast<uint8_t>(v >> (i * 8));
}

static void PatchBctInitializer(Module& m, WbModuleState& s, const std::vector<WbSite>& sites)
{
    LLVMContext& c = m.getContext();
    const size_t stride = BctSlotBytes(s);
    std::vector<uint8_t> bytes(kMaxSites * stride, 0);
    for (const auto& site : sites)
    {
        const size_t base = site.slotid * stride;
        WriteU64LE(bytes, base + 0, site.bct);
        if (stride > 8) WriteU64LE(bytes, base + 8, site.bcthi);
    }
    s.gvbct->setInitializer(ConstantDataArray::get(c, bytes));
}

static void PatchBetasInitializer(Module& m, WbModuleState& s, const std::vector<WbSite>& sites)
{
    LLVMContext& c = m.getContext();
    const size_t stride = BetasSlotBytes(s);
    std::vector<uint8_t> bytes(kMaxSites * stride, 0);
    for (const auto& site : sites)
    {
        const size_t base = site.slotid * stride;
        WriteU64LE(bytes, base + 0, site.betasite);
        if (stride > 8) WriteU64LE(bytes, base + 8, site.betasitehi);
    }
    s.gvbetas->setInitializer(ConstantDataArray::get(c, bytes));
}

static Function* EmitFy256(Module& m)
{
    LLVMContext& c = m.getContext();
    Type* i8 = Type::getInt8Ty(c);
    Type* i64 = Type::getInt64Ty(c);
    PointerType* i8p = PointerType::get(c, 0);

    FunctionType* ft = FunctionType::get(i64, {i8p, i64}, false);
    Function* f = Function::Create(ft, GlobalValue::InternalLinkage, "__wb_fy256", &m);
    f->addFnAttr(Attribute::NoInline);
    f->addFnAttr(Attribute::NoUnwind);

    Value* dstarg = f->getArg(0);
    Value* seedarg = f->getArg(1);

    BasicBlock* entryBB = BasicBlock::Create(c, "Entry", f);
    BasicBlock* initloop = BasicBlock::Create(c, "InitLoop", f);
    BasicBlock* initbody = BasicBlock::Create(c, "InitBody", f);
    BasicBlock* shufloop = BasicBlock::Create(c, "ShufLoop", f);
    BasicBlock* shufbody = BasicBlock::Create(c, "ShufBody", f);
    BasicBlock* shufdone = BasicBlock::Create(c, "ShufDone", f);

    {
        IRBuilder<> b(entryBB);
        b.CreateBr(initloop);
    }

    PHINode* initi = nullptr;
    {
        IRBuilder<> b(initloop);
        initi = b.CreatePHI(i64, 2, "I");
        initi->addIncoming(b.getInt64(0), entryBB);
        Value* cmp = b.CreateICmpULT(initi, b.getInt64(256));
        b.CreateCondBr(cmp, initbody, shufloop);
    }
    {
        IRBuilder<> b(initbody);
        Value* slot = b.CreateInBoundsGEP(i8, dstarg, initi);
        Value* val = b.CreateTrunc(initi, i8);
        b.CreateAlignedStore(val, slot, Align(1));
        Value* inext = b.CreateAdd(initi, b.getInt64(1));
        initi->addIncoming(inext, initbody);
        b.CreateBr(initloop);
    }

    PHINode* shufk = nullptr;
    PHINode* shufstate = nullptr;
    {
        IRBuilder<> b(shufloop);
        shufk = b.CreatePHI(i64, 2, "K");
        shufstate = b.CreatePHI(i64, 2, "State");
        shufk->addIncoming(b.getInt64(255), initloop);
        shufstate->addIncoming(seedarg, initloop);
        Value* cmp = b.CreateICmpUGT(shufk, b.getInt64(0));
        b.CreateCondBr(cmp, shufbody, shufdone);
    }
    {
        IRBuilder<> b(shufbody);
        constexpr uint64_t weylconst = 0x9E3779B97F4A7C15ULL;
        Value* weyl = b.CreateAdd(shufstate, b.getInt64(weylconst));
        Value* s0 = b.CreateLShr(weyl, b.getInt64(33));
        Value* x0 = b.CreateXor(weyl, s0);
        Value* m0 = b.CreateMul(x0, b.getInt64(kMixerMul1));
        Value* s1 = b.CreateLShr(m0, b.getInt64(33));
        Value* x1 = b.CreateXor(m0, s1);
        Value* m1 = b.CreateMul(x1, b.getInt64(kMixerMul2));
        Value* s2 = b.CreateLShr(m1, b.getInt64(33));
        Value* newstate = b.CreateXor(m1, s2, "NewState");

        Value* kp1 = b.CreateAdd(shufk, b.getInt64(1));
        Value* j = b.CreateURem(newstate, kp1);
        Value* pk = b.CreateInBoundsGEP(i8, dstarg, shufk);
        Value* pj = b.CreateInBoundsGEP(i8, dstarg, j);
        LoadInst* vk = b.CreateAlignedLoad(i8, pk, Align(1), "Vk");
        LoadInst* vj = b.CreateAlignedLoad(i8, pj, Align(1), "Vj");
        b.CreateAlignedStore(vj, pk, Align(1));
        b.CreateAlignedStore(vk, pj, Align(1));

        Value* knext = b.CreateSub(shufk, b.getInt64(1));
        shufk->addIncoming(knext, shufbody);
        shufstate->addIncoming(newstate, shufbody);
        b.CreateBr(shufloop);
    }
    {
        IRBuilder<> b(shufdone);
        b.CreateRet(shufstate);
    }

    return f;
}

static Function* EmitInvperm256(Module& m)
{
    LLVMContext& c = m.getContext();
    Type* i8 = Type::getInt8Ty(c);
    Type* i64 = Type::getInt64Ty(c);
    Type* voidty = Type::getVoidTy(c);
    PointerType* i8p = PointerType::get(c, 0);

    FunctionType* ft = FunctionType::get(voidty, {i8p, i8p}, false);
    Function* f = Function::Create(ft, GlobalValue::InternalLinkage, "__wb_invperm256", &m);
    f->addFnAttr(Attribute::NoInline);
    f->addFnAttr(Attribute::NoUnwind);

    Value* fwd = f->getArg(0);
    Value* inv = f->getArg(1);

    BasicBlock* entryBB = BasicBlock::Create(c, "Entry", f);
    BasicBlock* loopBB = BasicBlock::Create(c, "Loop", f);
    BasicBlock* bodyBB = BasicBlock::Create(c, "Body", f);
    BasicBlock* exitBB = BasicBlock::Create(c, "Exit", f);

    {
        IRBuilder<> b(entryBB);
        b.CreateBr(loopBB);
    }

    PHINode* iphi = nullptr;
    {
        IRBuilder<> b(loopBB);
        iphi = b.CreatePHI(i64, 2, "I");
        iphi->addIncoming(b.getInt64(0), entryBB);
        Value* cmp = b.CreateICmpULT(iphi, b.getInt64(256));
        b.CreateCondBr(cmp, bodyBB, exitBB);
    }
    {
        IRBuilder<> b(bodyBB);
        Value* fwdslot = b.CreateInBoundsGEP(i8, fwd, iphi);
        LoadInst* fwdval = b.CreateAlignedLoad(i8, fwdslot, Align(1));
        Value* fwdi64 = b.CreateZExt(fwdval, i64);
        Value* invslot = b.CreateInBoundsGEP(i8, inv, fwdi64);
        Value* iasi8 = b.CreateTrunc(iphi, i8);
        b.CreateAlignedStore(iasi8, invslot, Align(1));
        Value* inext = b.CreateAdd(iphi, b.getInt64(1));
        iphi->addIncoming(inext, bodyBB);
        b.CreateBr(loopBB);
    }
    {
        IRBuilder<> b(exitBB);
        b.CreateRetVoid();
    }

    return f;
}

static Function* EmitRegenInEnc(Module& m, const WbModuleState& s, Function* fnfy256, Function* fninvperm256)
{
    LLVMContext& c = m.getContext();
    Type* i8 = Type::getInt8Ty(c);
    Type* i64 = Type::getInt64Ty(c);
    Type* voidty = Type::getVoidTy(c);

    const bool freshre = s.freshroundenc;
    const bool freshmath = freshre && s.freshciphermath;
    const int rounds = s.rounds;
    const int freshpermcount = rounds * 8;
    const int freshpermbytes = freshpermcount * 256;

    uint64_t invshiftpacked = 0;
    uint32_t mdsrowlit[4] = {0, 0, 0, 0};
    uint64_t bitrotpacked = 0;
    uint64_t roundprogpacked = 0;
    uint64_t fusionpacked = 0;
    if (freshmath)
    {
        const auto invshift = InvertShift(s.shift);
        for (int k = 0; k < 8; ++k)
            invshiftpacked |= static_cast<uint64_t>(invshift[k]) << (k * 8);
        for (int bi = 0; bi < 4; ++bi)
        {
            uint32_t row = 0;
            for (int j = 0; j < 4; ++j)
                row |= static_cast<uint32_t>(s.mds[bi][j]) << (j * 8);
            mdsrowlit[bi] = row;
        }
        for (int r = 0; r < rounds; ++r)
            bitrotpacked |= static_cast<uint64_t>(s.bitrot[r] & 7u) << (r * 8);
        for (int r = 0; r < rounds; ++r)
        {
            roundprogpacked |= static_cast<uint64_t>(s.roundprogram[r]) << (r * 8);
            fusionpacked |= static_cast<uint64_t>(s.fusion[r]) << (r * 8);
        }
    }
    const uint64_t groutepacked = s.groutepacked;

    FunctionType* ft = FunctionType::get(voidty, {i64}, false);
    Function* f = Function::Create(ft, GlobalValue::InternalLinkage, "__wb_regen_in_enc", &m);
    f->addFnAttr(Attribute::NoInline);
    f->addFnAttr(Attribute::NoUnwind);

    Value* seedarg = f->getArg(0);

    BasicBlock* entryBB = BasicBlock::Create(c, "Entry", f);
    BasicBlock* fyloop = BasicBlock::Create(c, "FyLoop", f);
    BasicBlock* fybody = BasicBlock::Create(c, "FyBody", f);
    BasicBlock* invloop = BasicBlock::Create(c, "InvLoop", f);
    BasicBlock* invbody = BasicBlock::Create(c, "InvBody", f);
    BasicBlock* mcentry = BasicBlock::Create(c, "McEntry", f);

    BasicBlock* refyloop = nullptr;
    BasicBlock* refybody = nullptr;
    BasicBlock* reinvloop = nullptr;
    BasicBlock* reinvbody = nullptr;
    BasicBlock* republoop = nullptr;
    BasicBlock* repubbody = nullptr;
    if (freshre)
    {
        refyloop = BasicBlock::Create(c, "ReFyLoop", f);
        refybody = BasicBlock::Create(c, "ReFyBody", f);
        reinvloop = BasicBlock::Create(c, "ReInvLoop", f);
        reinvbody = BasicBlock::Create(c, "ReInvBody", f);
        republoop = BasicBlock::Create(c, "RePubLoop", f);
        repubbody = BasicBlock::Create(c, "RePubBody", f);
    }

    BasicBlock* rksboxdraw = nullptr;
    if (freshmath) rksboxdraw = BasicBlock::Create(c, "RkSBoxDraw", f);

    BasicBlock* rloop = BasicBlock::Create(c, "TBoxRLoop", f);
    BasicBlock* rbody = BasicBlock::Create(c, "TBoxRBody", f);
    BasicBlock* iloop = BasicBlock::Create(c, "TBoxILoop", f);
    BasicBlock* ibody = BasicBlock::Create(c, "TBoxIBody", f);
    BasicBlock* xloop = BasicBlock::Create(c, "TBoxXLoop", f);
    BasicBlock* xbody = BasicBlock::Create(c, "TBoxXBody", f);
    BasicBlock* xexit = BasicBlock::Create(c, "TBoxXExit", f);
    BasicBlock* iexit = BasicBlock::Create(c, "TBoxIExit", f);
    BasicBlock* rexit = BasicBlock::Create(c, "TBoxRExit", f);

    BasicBlock* xorrploop = nullptr;
    BasicBlock* xorrpbody = nullptr;
    BasicBlock* xorabloop = nullptr;
    BasicBlock* xorabbody = nullptr;
    BasicBlock* xorabexit = nullptr;
    BasicBlock* xorrpexit = nullptr;
    if (freshre)
    {
        xorrploop = BasicBlock::Create(c, "XorRpLoop", f);
        xorrpbody = BasicBlock::Create(c, "XorRpBody", f);
        xorabloop = BasicBlock::Create(c, "XorAbLoop", f);
        xorabbody = BasicBlock::Create(c, "XorAbBody", f);
        xorabexit = BasicBlock::Create(c, "XorAbExit", f);
        xorrpexit = BasicBlock::Create(c, "XorRpExit", f);
    }

    // ARKTable rebuild gates on fresh_math AND anyarkout. The loop body overwrites every (r,p,y) slot but none ARKOut rounds are never read, so content doesn't matter; without the AND the rebuild would do kMaxRounds*8*256 wasted stores
    const bool rebuildark = freshmath && s.anyarkout;
    BasicBlock* arkrploop = nullptr;
    BasicBlock* arkrpbody = nullptr;
    BasicBlock* arkyloop = nullptr;
    BasicBlock* arkybody = nullptr;
    BasicBlock* arkyexit = nullptr;
    BasicBlock* arkrpexit = nullptr;
    if (rebuildark)
    {
        arkrploop = BasicBlock::Create(c, "ARKRpLoop", f);
        arkrpbody = BasicBlock::Create(c, "ARKRpBody", f);
        arkyloop = BasicBlock::Create(c, "ARKYLoop", f);
        arkybody = BasicBlock::Create(c, "ARKYBody", f);
        arkyexit = BasicBlock::Create(c, "ARKYExit", f);
        arkrpexit = BasicBlock::Create(c, "ARKRpExit", f);
    }

    ArrayType* buf2k = ArrayType::get(i8, 2048);
    ArrayType* buffresh = freshre ? ArrayType::get(i8, static_cast<uint64_t>(freshpermbytes)) : nullptr;
    ArrayType* buf256 = freshmath ? ArrayType::get(i8, 256) : nullptr;
    ArrayType* bufrk = freshmath ? ArrayType::get(i64, static_cast<uint64_t>(kMaxRounds)) : nullptr;

    Value* fwdbuf = nullptr;
    Value* invbuf = nullptr;
    Value* refwdbuf = nullptr;
    Value* reinvbuf = nullptr;
    Value* sboxbuf = nullptr;
    Value* invsbxbuf = nullptr;
    Value* rkbuf = nullptr;

    {
        IRBuilder<> b(entryBB);
        fwdbuf = b.CreateAlloca(buf2k, nullptr, "WbFwdBuf");
        invbuf = b.CreateAlloca(buf2k, nullptr, "WbInvBuf");
        if (freshre)
        {
            refwdbuf = b.CreateAlloca(buffresh, nullptr, "WbReFwdBuf");
            reinvbuf = b.CreateAlloca(buffresh, nullptr, "WbReInvBuf");
        }
        if (freshmath)
        {
            sboxbuf = b.CreateAlloca(buf256, nullptr, "WbSBoxBuf");
            invsbxbuf = b.CreateAlloca(buf256, nullptr, "WbInvSBoxBuf");
            rkbuf = b.CreateAlloca(bufrk, nullptr, "WbRkBuf");
        }
        b.CreateBr(fyloop);
    }

    PHINode* fyp = nullptr;
    PHINode* fystate = nullptr;
    {
        IRBuilder<> b(fyloop);
        fyp = b.CreatePHI(i64, 2, "P");
        fystate = b.CreatePHI(i64, 2, "State");
        fyp->addIncoming(b.getInt64(0), entryBB);
        fystate->addIncoming(seedarg, entryBB);
        Value* cmp = b.CreateICmpULT(fyp, b.getInt64(8));
        b.CreateCondBr(cmp, fybody, invloop);
    }
    {
        IRBuilder<> b(fybody);
        Value* off = b.CreateMul(fyp, b.getInt64(256));
        Value* slot = b.CreateInBoundsGEP(i8, fwdbuf, off);
        Value* newstate = b.CreateCall(fnfy256, {slot, fystate}, "StateNext");
        Value* pnext = b.CreateAdd(fyp, b.getInt64(1));
        fyp->addIncoming(pnext, fybody);
        fystate->addIncoming(newstate, fybody);
        b.CreateBr(fyloop);
    }

    PHINode* invp = nullptr;
    {
        IRBuilder<> b(invloop);
        invp = b.CreatePHI(i64, 2, "P2");
        invp->addIncoming(b.getInt64(0), fyloop);
        Value* cmp = b.CreateICmpULT(invp, b.getInt64(8));
        b.CreateCondBr(cmp, invbody, mcentry);
    }
    {
        IRBuilder<> b(invbody);
        Value* off = b.CreateMul(invp, b.getInt64(256));
        Value* fwds = b.CreateInBoundsGEP(i8, fwdbuf, off);
        Value* invs = b.CreateInBoundsGEP(i8, invbuf, off);
        b.CreateCall(fninvperm256, {fwds, invs});
        Value* pnext = b.CreateAdd(invp, b.getInt64(1));
        invp->addIncoming(pnext, invbody);
        b.CreateBr(invloop);
    }
    {
        IRBuilder<> b(mcentry);
        Value* dstfwd = b.CreateInBoundsGEP(i8, s.gvdyn, b.getInt64(kInEncOffset));
        b.CreateMemCpy(dstfwd, MaybeAlign(1), fwdbuf, MaybeAlign(1), b.getInt64(2048));
        Value* dstinvr0 = b.CreateInBoundsGEP(i8, s.gvdyn, b.getInt64(s.offinvpermsperround));
        b.CreateMemCpy(dstinvr0, MaybeAlign(1), invbuf, MaybeAlign(1), b.getInt64(2048));
        b.CreateBr(freshre ? refyloop : rloop);
    }

    PHINode* refyrp = nullptr;
    PHINode* refystate = nullptr;
    if (freshre)
    {
        {
            IRBuilder<> b(refyloop);
            refyrp = b.CreatePHI(i64, 2, "ReRp");
            refystate = b.CreatePHI(i64, 2, "ReState");
            refyrp->addIncoming(b.getInt64(0), mcentry);
            refystate->addIncoming(fystate, mcentry);
            Value* cmp = b.CreateICmpULT(refyrp, b.getInt64(static_cast<uint64_t>(freshpermcount)));
            b.CreateCondBr(cmp, refybody, reinvloop);
        }
        {
            IRBuilder<> b(refybody);
            Value* off = b.CreateMul(refyrp, b.getInt64(256));
            Value* slot = b.CreateInBoundsGEP(i8, refwdbuf, off);
            Value* newstate = b.CreateCall(fnfy256, {slot, refystate}, "ReStateNext");
            Value* rpnext = b.CreateAdd(refyrp, b.getInt64(1));
            refyrp->addIncoming(rpnext, refybody);
            refystate->addIncoming(newstate, refybody);
            b.CreateBr(refyloop);
        }
    }

    PHINode* reinvrp = nullptr;
    if (freshre)
    {
        {
            IRBuilder<> b(reinvloop);
            reinvrp = b.CreatePHI(i64, 2, "ReRp2");
            reinvrp->addIncoming(b.getInt64(0), refyloop);
            Value* cmp = b.CreateICmpULT(reinvrp, b.getInt64(static_cast<uint64_t>(freshpermcount)));
            b.CreateCondBr(cmp, reinvbody, republoop);
        }
        {
            IRBuilder<> b(reinvbody);
            Value* off = b.CreateMul(reinvrp, b.getInt64(256));
            Value* fwds = b.CreateInBoundsGEP(i8, refwdbuf, off);
            Value* invs = b.CreateInBoundsGEP(i8, reinvbuf, off);
            b.CreateCall(fninvperm256, {fwds, invs});
            Value* rpnext = b.CreateAdd(reinvrp, b.getInt64(1));
            reinvrp->addIncoming(rpnext, reinvbody);
            b.CreateBr(reinvloop);
        }
    }

    PHINode* repubr = nullptr;
    if (freshre)
    {
        {
            IRBuilder<> b(republoop);
            repubr = b.CreatePHI(i64, 2, "RePubR");
            repubr->addIncoming(b.getInt64(0), reinvloop);
            Value* cmp = b.CreateICmpULT(repubr, b.getInt64(static_cast<uint64_t>(rounds)));
            BasicBlock* repubexit = freshmath ? rksboxdraw : rloop;
            b.CreateCondBr(cmp, repubbody, repubexit);
        }
        {
            IRBuilder<> b(repubbody);
            Value* srcoff = b.CreateMul(repubr, b.getInt64(2048));
            Value* srcptr = b.CreateInBoundsGEP(i8, reinvbuf, srcoff);
            Value* rplus1 = b.CreateAdd(repubr, b.getInt64(1));
            Value* islast = b.CreateICmpEQ(repubr, b.getInt64(static_cast<uint64_t>(rounds - 1)));
            Value* dstoffnormal = b.CreateAdd(b.getInt64(s.offinvpermsperround), b.CreateMul(rplus1, b.getInt64(2048)));
            Value* dstofflast = b.getInt64(kOutEncInvOffset);
            Value* dstoff = b.CreateSelect(islast, dstofflast, dstoffnormal);
            Value* dstptr = b.CreateInBoundsGEP(i8, s.gvdyn, dstoff);
            b.CreateMemCpy(dstptr, MaybeAlign(1), srcptr, MaybeAlign(1), b.getInt64(2048));
            Value* rnext = b.CreateAdd(repubr, b.getInt64(1));
            repubr->addIncoming(rnext, repubbody);
            b.CreateBr(republoop);
        }
    }

    if (freshmath)
    {
        IRBuilder<> b(rksboxdraw);
        Value* sboxslot = b.CreateInBoundsGEP(i8, sboxbuf, b.getInt64(0));
        Value* stateaftersbox = b.CreateCall(fnfy256, {sboxslot, refystate}, "StateAfterSBox");
        Value* invsboxslot = b.CreateInBoundsGEP(i8, invsbxbuf, b.getInt64(0));
        b.CreateCall(fninvperm256, {sboxslot, invsboxslot});
        Value* dstinvsbox = b.CreateInBoundsGEP(i8, s.gvdyn, b.getInt64(s.offfreshinvsbox));
        b.CreateMemCpy(dstinvsbox, MaybeAlign(1), invsboxslot, MaybeAlign(1), b.getInt64(256));

        constexpr uint64_t rksalt = 0xB07EC0DE57B105EEULL;
        Value* dynrkbase = b.CreateInBoundsGEP(i8, s.gvdyn, b.getInt64(s.offfreshrk));
        for (int r = 0; r < rounds; ++r)
        {
            Value* mixbase = b.CreateXor(stateaftersbox, b.getInt64(rksalt ^ static_cast<uint64_t>(r)));
            Value* s0 = b.CreateLShr(mixbase, b.getInt64(33));
            Value* x0 = b.CreateXor(mixbase, s0);
            Value* m0 = b.CreateMul(x0, b.getInt64(kMixerMul1));
            Value* s1 = b.CreateLShr(m0, b.getInt64(33));
            Value* x1 = b.CreateXor(m0, s1);
            Value* m1 = b.CreateMul(x1, b.getInt64(kMixerMul2));
            Value* s2 = b.CreateLShr(m1, b.getInt64(33));
            Value* rkr = b.CreateXor(m1, s2, "WbFreshRkR");

            Value* rkslotlocal = b.CreateInBoundsGEP(bufrk, rkbuf, {b.getInt64(0), b.getInt64(static_cast<uint64_t>(r))});
            b.CreateAlignedStore(rkr, rkslotlocal, Align(8));

            Value* rkslotdyn = b.CreateInBoundsGEP(i8, dynrkbase, b.getInt64(static_cast<uint64_t>(r) * 8u));
            StoreInst* si = b.CreateAlignedStore(rkr, rkslotdyn, Align(8));
            si->setVolatile(true);
        }
        b.CreateBr(rloop);
    }

    PHINode* rphi = nullptr;
    {
        IRBuilder<> b(rloop);
        rphi = b.CreatePHI(i64, 2, "R");
        BasicBlock* rlooppred = freshmath ? rksboxdraw : (freshre ? republoop : mcentry);
        rphi->addIncoming(b.getInt64(0), rlooppred);
        Value* cmp = b.CreateICmpULT(rphi, b.getInt64(static_cast<uint64_t>(rounds)));
        b.CreateCondBr(cmp, rbody, rexit);
    }
    {
        IRBuilder<> b(rbody);
        b.CreateBr(iloop);
    }
    PHINode* iphi = nullptr;
    {
        IRBuilder<> b(iloop);
        iphi = b.CreatePHI(i64, 2, "I");
        iphi->addIncoming(b.getInt64(0), rbody);
        Value* cmp = b.CreateICmpULT(iphi, b.getInt64(8));
        b.CreateCondBr(cmp, ibody, iexit);
    }
    {
        IRBuilder<> b(ibody);
        b.CreateBr(xloop);
    }
    PHINode* xphi = nullptr;
    {
        IRBuilder<> b(xloop);
        xphi = b.CreatePHI(i64, 2, "X");
        xphi->addIncoming(b.getInt64(0), ibody);
        Value* cmp = b.CreateICmpULT(xphi, b.getInt64(256));
        b.CreateCondBr(cmp, xbody, xexit);
    }
    {
        IRBuilder<> b(xbody);
        Value* ri = b.CreateAdd(b.CreateMul(rphi, b.getInt64(8)), iphi);
        Value* rioff2048 = b.CreateShl(ri, b.getInt64(11));
        Value* rioff256 = b.CreateShl(ri, b.getInt64(8));

        Value* invoff = b.CreateAdd(b.getInt64(s.offinvpermsperround), b.CreateAdd(rioff256, xphi));
        Value* invcell = b.CreateInBoundsGEP(i8, s.gvdyn, invoff);
        LoadInst* decodedi8 = b.CreateAlignedLoad(i8, invcell, Align(1), "Decoded");
        Value* decodedi64 = b.CreateZExt(decodedi8, i64);

        Value* preencoff = b.CreateAdd(b.getInt64(s.offpreencbank), b.CreateAdd(rioff2048, b.CreateShl(decodedi64, b.getInt64(3))));
        Value* preencptr = b.CreateInBoundsGEP(i8, s.gvstatic, preencoff);
        LoadInst* preencval = b.CreateAlignedLoad(i64, preencptr, Align(8), "Preenc");

        Value* entry = preencval;
        if (freshmath)
        {
            Value* rkslot = b.CreateInBoundsGEP(bufrk, rkbuf, {b.getInt64(0), rphi});
            LoadInst* rkr = b.CreateAlignedLoad(i64, rkslot, Align(8), "RkR");
            Value* rbits = b.CreateShl(rphi, b.getInt64(3));
            Value* rtr = b.CreateAnd(b.CreateLShr(b.getInt64(roundprogpacked), rbits), b.getInt64(0xff));
            Value* fgr = b.CreateAnd(b.CreateLShr(b.getInt64(fusionpacked), rbits), b.getInt64(0xff));
            Value* isB = b.CreateICmpEQ(rtr, b.getInt64(1));
            Value* isF = b.CreateICmpEQ(rtr, b.getInt64(4));
            Value* isG = b.CreateICmpEQ(rtr, b.getInt64(5));
            Value* rtisA = b.CreateICmpEQ(rtr, b.getInt64(0));
            Value* fgisFull = b.CreateICmpNE(fgr, b.getInt64(1));
            Value* rtisAorForG = b.CreateOr(b.CreateOr(rtisA, isF), isG);
            Value* isAFull = b.CreateAnd(rtisAorForG, fgisFull);

            Value* ibits = b.CreateShl(iphi, b.getInt64(3));
            Value* rkbyteinputi64 = b.CreateAnd(b.CreateLShr(rkr, ibits), b.getInt64(0xff));
            Value* presboxi64 = b.CreateSelect(isB, b.CreateXor(decodedi64, rkbyteinputi64), decodedi64);

            Value* subbedptr = b.CreateInBoundsGEP(i8, sboxbuf, presboxi64);
            LoadInst* sboxlookupi8 = b.CreateAlignedLoad(i8, subbedptr, Align(1), "SBoxALookup");

            Value* sboxboff = b.CreateAdd(b.getInt64(static_cast<uint64_t>(s.offsboxgb)), b.CreateAdd(b.CreateShl(rphi, b.getInt64(8)), presboxi64));
            Value* sboxbptr = b.CreateInBoundsGEP(i8, s.gvdyn, sboxboff);
            LoadInst* sboxblookupi8 = b.CreateAlignedLoad(i8, sboxbptr, Align(1), "SBoxBLookup");

            Value* grouteshift = b.CreateAdd(b.CreateShl(rphi, b.getInt64(3)), iphi);
            Value* groutebit = b.CreateAnd(b.CreateLShr(b.getInt64(groutepacked), grouteshift), b.getInt64(1));
            Value* groutenonzero = b.CreateICmpNE(groutebit, b.getInt64(0));
            Value* usesboxb = b.CreateAnd(isG, groutenonzero);

            Value* presboxi8 = b.CreateTrunc(presboxi64, i8);
            Value* postg = b.CreateSelect(usesboxb, static_cast<Value*>(sboxblookupi8), static_cast<Value*>(sboxlookupi8));
            Value* subbedi8 = b.CreateSelect(isF, presboxi8, postg, "Subbed");
            Value* subbedi32 = b.CreateZExt(subbedi8, Type::getInt32Ty(c));

            Value* brshift = b.CreateShl(rphi, b.getInt64(3));
            Value* brshifti32 = b.CreateTrunc(brshift, Type::getInt32Ty(c));
            Value* brshifted = b.CreateLShr(b.getInt64(bitrotpacked), brshift);
            Value* ki64 = b.CreateAnd(brshifted, b.getInt64(7));
            Value* ki32 = b.CreateTrunc(ki64, Type::getInt32Ty(c));
            Value* left = b.CreateShl(subbedi32, ki32);
            Value* invki32 = b.CreateSub(ConstantInt::get(Type::getInt32Ty(c), 8), ki32);
            Value* right = b.CreateLShr(subbedi32, invki32);
            Value* rotated = b.CreateOr(left, right);
            Value* rotatedmasked = b.CreateAnd(rotated, ConstantInt::get(Type::getInt32Ty(c), 0xff));
            Value* subbedeffi32 = rotatedmasked;
            Value* subbedeffi64 = b.CreateZExt(subbedeffi32, i64);
            (void)subbedi32;
            (void)brshifti32;

            Value* ci64 = b.CreateLShr(iphi, b.getInt64(2));
            Value* c4 = b.CreateShl(ci64, b.getInt64(2));
            Value* ji64 = b.CreateAnd(iphi, b.getInt64(3));
            Value* jbits = b.CreateShl(ji64, b.getInt64(3));
            Value* jiszero = b.CreateICmpEQ(ji64, b.getInt64(0));
            Value* outputarkon = b.CreateAnd(isAFull, jiszero);

            Value* roff2048 = b.CreateShl(rphi, b.getInt64(11));

            Value* newentry = b.getInt64(0);
            for (int bi = 0; bi < 4; ++bi)
            {
                Value* posidx = b.CreateAdd(c4, b.getInt64(static_cast<uint64_t>(bi)));
                Value* posbits = b.CreateShl(posidx, b.getInt64(3));
                Value* isshifted = b.CreateLShr(b.getInt64(invshiftpacked), posbits);
                Value* post = b.CreateAnd(isshifted, b.getInt64(0xff));
                Value* postbits = b.CreateShl(post, b.getInt64(3));

                Value* mdsshifted = b.CreateLShr(b.getInt64(static_cast<uint64_t>(mdsrowlit[bi])), jbits);
                Value* coeff = b.CreateAnd(mdsshifted, b.getInt64(0xff));

                Value* gmuloff = b.CreateAdd(b.getInt64(s.offgfmultbl), b.CreateAdd(b.CreateShl(coeff, b.getInt64(8)), subbedeffi64));
                Value* gmulptr = b.CreateInBoundsGEP(i8, s.gvstatic, gmuloff);
                LoadInst* contribi8 = b.CreateAlignedLoad(i8, gmulptr, Align(1), "Contrib");

                Value* rkbyteatposi64 = b.CreateAnd(b.CreateLShr(rkr, postbits), b.getInt64(0xff));
                Value* rkbyteatposi8 = b.CreateTrunc(rkbyteatposi64, i8);
                Value* arkdelta = b.CreateSelect(outputarkon, rkbyteatposi8, ConstantInt::get(i8, 0));
                Value* contribarked = b.CreateXor(contribi8, arkdelta);

                Value* contribi64 = b.CreateZExt(contribarked, i64);
                Value* fwdoff = b.CreateAdd(roff2048, b.CreateAdd(b.CreateShl(post, b.getInt64(8)), contribi64));
                Value* fwdcell = b.CreateInBoundsGEP(i8, refwdbuf, fwdoff);
                LoadInst* enci8 = b.CreateAlignedLoad(i8, fwdcell, Align(1), "ReEnc");
                Value* enci64 = b.CreateZExt(enci8, i64);
                Value* encshifted = b.CreateShl(enci64, postbits);
                newentry = b.CreateOr(newentry, encshifted);
            }
            entry = newentry;
        }
        else if (freshre)
        {
            Value* roff2048 = b.CreateShl(rphi, b.getInt64(11));
            Value* newentry = b.getInt64(0);
            for (int p = 0; p < 8; ++p)
            {
                Value* shifted = b.CreateLShr(preencval, b.getInt64(static_cast<uint64_t>(p * 8)));
                Value* rawi8 = b.CreateTrunc(shifted, i8);
                Value* rawi64 = b.CreateZExt(rawi8, i64);
                Value* fwdoff = b.CreateAdd(roff2048, b.CreateAdd(b.getInt64(static_cast<uint64_t>(p * 256)), rawi64));
                Value* fwdcell = b.CreateInBoundsGEP(i8, refwdbuf, fwdoff);
                LoadInst* enci8 = b.CreateAlignedLoad(i8, fwdcell, Align(1), "ReEnc");
                Value* enci64 = b.CreateZExt(enci8, i64);
                Value* encshifted = b.CreateShl(enci64, b.getInt64(static_cast<uint64_t>(p * 8)));
                newentry = b.CreateOr(newentry, encshifted);
            }
            entry = newentry;
        }

        Value* rroundoff = b.CreateMul(rphi, b.getInt64(kTBoxRoundStride));
        Value* irowoff = b.CreateMul(iphi, b.getInt64(kTBoxRowStride));
        Value* xentryoff = b.CreateShl(xphi, b.getInt64(4));
        Value* tboxoff = b.CreateAdd(b.getInt64(kTBoxOffset), b.CreateAdd(b.CreateAdd(rroundoff, irowoff), xentryoff));
        Value* tboxptr = b.CreateInBoundsGEP(i8, s.gvdyn, tboxoff);
        StoreInst* si = b.CreateAlignedStore(entry, tboxptr, Align(8));
        si->setVolatile(true);

        Value* xnext = b.CreateAdd(xphi, b.getInt64(1));
        xphi->addIncoming(xnext, xbody);
        b.CreateBr(xloop);
    }
    {
        IRBuilder<> b(xexit);
        Value* inext = b.CreateAdd(iphi, b.getInt64(1));
        iphi->addIncoming(inext, xexit);
        b.CreateBr(iloop);
    }
    {
        IRBuilder<> b(iexit);
        Value* rnext = b.CreateAdd(rphi, b.getInt64(1));
        rphi->addIncoming(rnext, iexit);
        b.CreateBr(rloop);
    }
    {
        IRBuilder<> b(rexit);
        if (freshre) b.CreateBr(xorrploop);
        else b.CreateRetVoid();
    }

    PHINode* xorrp = nullptr;
    Value* permbaseoff = nullptr;
    Value* dynrpbase = nullptr;
    if (freshre)
    {
        {
            IRBuilder<> b(xorrploop);
            xorrp = b.CreatePHI(i64, 2, "XorRp");
            xorrp->addIncoming(b.getInt64(0), rexit);
            Value* cmp = b.CreateICmpULT(xorrp, b.getInt64(static_cast<uint64_t>(freshpermcount)));
            b.CreateCondBr(cmp, xorrpbody, xorrpexit);
        }
        {
            IRBuilder<> b(xorrpbody);
            permbaseoff = b.CreateMul(xorrp, b.getInt64(256), "PermBase");
            Value* rpr = b.CreateLShr(xorrp, b.getInt64(3));
            Value* rpp = b.CreateAnd(xorrp, b.getInt64(7));
            Value* rroundoff = b.CreateMul(rpr, b.getInt64(kXorTblRoundStride));
            Value* ptbloff = b.CreateMul(rpp, b.getInt64(kXorTblTableStride));
            dynrpbase = b.CreateAdd(b.getInt64(kXorTblOffset), b.CreateAdd(rroundoff, ptbloff), "DynRpBase");
            b.CreateBr(xorabloop);
        }

        PHINode* xorab = nullptr;
        {
            IRBuilder<> b(xorabloop);
            xorab = b.CreatePHI(i64, 2, "XorAb");
            xorab->addIncoming(b.getInt64(0), xorrpbody);
            Value* cmp = b.CreateICmpULT(xorab, b.getInt64(65536));
            b.CreateCondBr(cmp, xorabbody, xorabexit);
        }
        {
            IRBuilder<> b(xorabbody);
            Value* ai64 = b.CreateLShr(xorab, b.getInt64(8));
            Value* bi64 = b.CreateAnd(xorab, b.getInt64(0xff));
            Value* invaoff = b.CreateAdd(permbaseoff, ai64);
            Value* invaptr = b.CreateInBoundsGEP(i8, reinvbuf, invaoff);
            LoadInst* dai8 = b.CreateAlignedLoad(i8, invaptr, Align(1), "Da");
            Value* invboff = b.CreateAdd(permbaseoff, bi64);
            Value* invbptr = b.CreateInBoundsGEP(i8, reinvbuf, invboff);
            LoadInst* dbi8 = b.CreateAlignedLoad(i8, invbptr, Align(1), "Db");
            Value* xri8 = b.CreateXor(dai8, dbi8);
            Value* xri64 = b.CreateZExt(xri8, i64);
            Value* fwdoff = b.CreateAdd(permbaseoff, xri64);
            Value* fwdcell = b.CreateInBoundsGEP(i8, refwdbuf, fwdoff);
            LoadInst* enci8 = b.CreateAlignedLoad(i8, fwdcell, Align(1), "XEnc");
            Value* dstoff = b.CreateAdd(dynrpbase, xorab);
            Value* dstptr = b.CreateInBoundsGEP(i8, s.gvdyn, dstoff);
            StoreInst* si = b.CreateAlignedStore(enci8, dstptr, Align(1));
            si->setVolatile(true);
            Value* abnext = b.CreateAdd(xorab, b.getInt64(1));
            xorab->addIncoming(abnext, xorabbody);
            b.CreateBr(xorabloop);
        }
        {
            IRBuilder<> b(xorabexit);
            Value* rpnext = b.CreateAdd(xorrp, b.getInt64(1));
            xorrp->addIncoming(rpnext, xorabexit);
            b.CreateBr(xorrploop);
        }
        {
            IRBuilder<> b(xorrpexit);
            if (rebuildark) b.CreateBr(arkrploop);
            else b.CreateRetVoid();
        }
    }

    if (rebuildark)
    {
        PHINode* arkrp = nullptr;
        Value* arkri64 = nullptr;
        Value* arkpi64 = nullptr;
        Value* arkrkbytepi64 = nullptr;
        Value* arkrpbasei64 = nullptr;
        Value* arkdynbasei64 = nullptr;
        {
            IRBuilder<> b(arkrploop);
            arkrp = b.CreatePHI(i64, 2, "ARKRp");
            arkrp->addIncoming(b.getInt64(0), xorrpexit);
            Value* cmp = b.CreateICmpULT(arkrp, b.getInt64(static_cast<uint64_t>(freshpermcount)));
            b.CreateCondBr(cmp, arkrpbody, arkrpexit);
        }
        {
            IRBuilder<> b(arkrpbody);
            arkri64 = b.CreateLShr(arkrp, b.getInt64(3), "ARKR");
            arkpi64 = b.CreateAnd(arkrp, b.getInt64(7), "ARKP");

            Value* rkslot = b.CreateInBoundsGEP(bufrk, rkbuf, {b.getInt64(0), arkri64});
            LoadInst* rkrval = b.CreateAlignedLoad(i64, rkslot, Align(8), "ARKRkR");
            Value* pbits = b.CreateShl(arkpi64, b.getInt64(3));
            arkrkbytepi64 = b.CreateAnd(b.CreateLShr(rkrval, pbits), b.getInt64(0xff), "ARKRkByteP");
            arkrpbasei64 = b.CreateShl(arkrp, b.getInt64(8), "ARKRpBase");
            Value* rdynoff = b.CreateShl(arkri64, b.getInt64(12));
            Value* pdynoff = b.CreateShl(arkpi64, b.getInt64(8));
            Value* rpdynoff = b.CreateAdd(rdynoff, pdynoff);
            arkdynbasei64 = b.CreateAdd(b.getInt64(kARKTblOffset), rpdynoff, "ARKDynBase");
            b.CreateBr(arkyloop);
        }
        PHINode* arky = nullptr;
        {
            IRBuilder<> b(arkyloop);
            arky = b.CreatePHI(i64, 2, "ARKY");
            arky->addIncoming(b.getInt64(0), arkrpbody);
            Value* cmp = b.CreateICmpULT(arky, b.getInt64(256));
            b.CreateCondBr(cmp, arkybody, arkyexit);
        }
        {
            IRBuilder<> b(arkybody);
            Value* invoff = b.CreateAdd(arkrpbasei64, arky);
            Value* invcell = b.CreateInBoundsGEP(i8, reinvbuf, invoff);
            LoadInst* invyi8 = b.CreateAlignedLoad(i8, invcell, Align(1), "ARKInvY");
            Value* invyi64 = b.CreateZExt(invyi8, i64);
            Value* xoredi64 = b.CreateXor(invyi64, arkrkbytepi64);
            Value* fwdoff = b.CreateAdd(arkrpbasei64, xoredi64);
            Value* fwdcell = b.CreateInBoundsGEP(i8, refwdbuf, fwdoff);
            LoadInst* enci8 = b.CreateAlignedLoad(i8, fwdcell, Align(1), "ARKEnc");
            Value* dstoff = b.CreateAdd(arkdynbasei64, arky);
            Value* dstptr = b.CreateInBoundsGEP(i8, s.gvdyn, dstoff);
            StoreInst* si = b.CreateAlignedStore(enci8, dstptr, Align(1));
            si->setVolatile(true);
            Value* ynext = b.CreateAdd(arky, b.getInt64(1));
            arky->addIncoming(ynext, arkybody);
            b.CreateBr(arkyloop);
        }
        {
            IRBuilder<> b(arkyexit);
            Value* rpnext = b.CreateAdd(arkrp, b.getInt64(1));
            arkrp->addIncoming(rpnext, arkyexit);
            b.CreateBr(arkrploop);
        }
        {
            IRBuilder<> b(arkrpexit);
            b.CreateRetVoid();
        }
    }

    return f;
}

static Function* EmitDecryptFn(Module& m, const WbModuleState& s)
{
    LLVMContext& c = m.getContext();
    Type* i8 = Type::getInt8Ty(c);
    Type* i32 = Type::getInt32Ty(c);
    Type* i64 = Type::getInt64Ty(c);

    const int rounds = s.rounds;

    const auto invshift = InvertShift(s.shift);
    uint64_t invshiftpacked = 0;
    for (int k = 0; k < 8; ++k)
        invshiftpacked |= static_cast<uint64_t>(invshift[k]) << (k * 8);
    uint8_t bitrotarr[kMaxRounds] = {0};
    for (int r = 0; r < rounds; ++r)
        bitrotarr[r] = s.bitrot[r] & 7u;

    FunctionType* ft = FunctionType::get(i64, {i64}, false);
    Function* f = Function::Create(ft, GlobalValue::InternalLinkage, "__wb_decrypt", &m);
    f->addFnAttr(Attribute::NoInline);
    f->addFnAttr(Attribute::NoUnwind);

    Value* inputarg = f->getArg(0);
    BasicBlock* entryBB = BasicBlock::Create(c, "Entry", f);
    IRBuilder<> b(entryBB);

    Value* dynrkbase = b.CreateInBoundsGEP(i8, s.gvdyn, b.getInt64(s.offfreshrk));
    Value* dyninvsboxbase = b.CreateInBoundsGEP(i8, s.gvdyn, b.getInt64(s.offfreshinvsbox));

    Value* state = inputarg;

    for (int r = rounds - 1; r >= 0; --r)
    {
        const uint8_t rt = s.roundprogram[r];

        Value* rkslot = b.CreateInBoundsGEP(i8, dynrkbase, b.getInt64(static_cast<uint64_t>(r) * 8u));
        LoadInst* rkr = b.CreateAlignedLoad(i64, rkslot, Align(8), "DecRk");
        rkr->setVolatile(true);

        if (rt == 0 || rt == 4 || rt == 5)
            state = b.CreateXor(state, rkr, "DecARKed");

        Value* newstate = b.getInt64(0);
        for (int i = 0; i < 8; ++i)
        {
            const unsigned srcpos = static_cast<unsigned>(invshift[i]);
            Value* shifted = b.CreateLShr(state, b.getInt64(srcpos * 8u));
            Value* bytei64 = b.CreateAnd(shifted, b.getInt64(0xff));
            Value* placed = b.CreateShl(bytei64, b.getInt64(static_cast<uint64_t>(i) * 8u));
            newstate = b.CreateOr(newstate, placed);
        }
        state = newstate;

        newstate = b.getInt64(0);
        for (int cc = 0; cc < 2; ++cc)
        {
            Value* colbytes[4];
            for (int j = 0; j < 4; ++j)
            {
                Value* shifted = b.CreateLShr(state, b.getInt64(static_cast<uint64_t>((cc * 4 + j) * 8)));
                colbytes[j] = b.CreateAnd(shifted, b.getInt64(0xff));
            }
            for (int bi = 0; bi < 4; ++bi)
            {
                Value* outbi64 = b.getInt64(0);
                for (int j = 0; j < 4; ++j)
                {
                    const uint64_t coeffshifted = static_cast<uint64_t>(s.invmds[bi][j]) << 8;
                    Value* gmuloff = b.CreateAdd(b.getInt64(static_cast<uint64_t>(s.offgfmultbl) + coeffshifted), colbytes[j]);
                    Value* gmulptr = b.CreateInBoundsGEP(i8, s.gvstatic, gmuloff);
                    LoadInst* gmi8 = b.CreateAlignedLoad(i8, gmulptr, Align(1), "DecGm");
                    Value* gmi64 = b.CreateZExt(gmi8, i64);
                    outbi64 = b.CreateXor(outbi64, gmi64);
                }
                Value* placed = b.CreateShl(outbi64, b.getInt64(static_cast<uint64_t>((cc * 4 + bi) * 8)));
                newstate = b.CreateOr(newstate, placed);
            }
        }
        state = newstate;

        const uint8_t kbyte = bitrotarr[r];
        if (kbyte != 0)
        {
            Value* newrot = b.getInt64(0);
            for (int i = 0; i < 8; ++i)
            {
                Value* shifted = b.CreateLShr(state, b.getInt64(static_cast<uint64_t>(i) * 8u));
                Value* bi32 = b.CreateTrunc(b.CreateAnd(shifted, b.getInt64(0xff)), i32);
                Value* rr = b.CreateLShr(bi32, ConstantInt::get(i32, kbyte));
                Value* rl = b.CreateShl(bi32, ConstantInt::get(i32, 8 - kbyte));
                Value* rot = b.CreateAnd(b.CreateOr(rr, rl), ConstantInt::get(i32, 0xff));
                Value* roti64 = b.CreateZExt(rot, i64);
                Value* placed = b.CreateShl(roti64, b.getInt64(static_cast<uint64_t>(i) * 8u));
                newrot = b.CreateOr(newrot, placed);
            }
            state = newrot;
        }

        if (rt != 4)
        {
            const bool isG = (rt == 5);
            const uint8_t grouter = s.groute[r];
            newstate = b.getInt64(0);
            for (int i = 0; i < 8; ++i)
            {
                Value* shifted = b.CreateLShr(state, b.getInt64(static_cast<uint64_t>(i) * 8u));
                Value* bi64 = b.CreateAnd(shifted, b.getInt64(0xff));
                Value* invsboxrowbase;
                if (isG && ((grouter >> i) & 1u) != 0u)
                {
                    invsboxrowbase = b.CreateInBoundsGEP(i8, s.gvdyn, b.getInt64(static_cast<uint64_t>(s.offinvsboxgb) + static_cast<uint64_t>(r) * 256u));
                }
                else
                {
                    invsboxrowbase = dyninvsboxbase;
                }
                Value* invsboxptr = b.CreateInBoundsGEP(i8, invsboxrowbase, bi64);
                LoadInst* invsbxi8 = b.CreateAlignedLoad(i8, invsboxptr, Align(1), "DecInvSBx");
                invsbxi8->setVolatile(true);
                Value* invsbxi64 = b.CreateZExt(invsbxi8, i64);
                Value* placed = b.CreateShl(invsbxi64, b.getInt64(static_cast<uint64_t>(i) * 8u));
                newstate = b.CreateOr(newstate, placed);
            }
            state = newstate;
        }

        if (rt == 1)
            state = b.CreateXor(state, rkr, "DecARKedB");
    }

    b.CreateRet(state);
    (void)invshiftpacked;
    return f;
}

static Function* EmitRuntimeInitCtor(Module& m, const WbModuleState& s, Function* fnregeninenc, Function* fndecrypt, size_t slotcount)
{
    if (slotcount == 0) return nullptr;

    LLVMContext& c = m.getContext();
    Type* i8 = Type::getInt8Ty(c);
    Type* i64 = Type::getInt64Ty(c);
    Type* voidty = Type::getVoidTy(c);

    FunctionType* ft = FunctionType::get(voidty, {}, false);
    Function* f = Function::Create(ft, GlobalValue::InternalLinkage, "__wb_runtime_init", &m);
    f->addFnAttr(Attribute::NoInline);
    f->addFnAttr(Attribute::NoUnwind);

    BasicBlock* entryBB = BasicBlock::Create(c, "Entry", f);
    BasicBlock* loopBB = BasicBlock::Create(c, "Loop", f);
    BasicBlock* bodyBB = BasicBlock::Create(c, "Body", f);
    BasicBlock* exitBB = BasicBlock::Create(c, "Exit", f);

    Value* seedval = nullptr;
    {
        IRBuilder<> b(entryBB);

        // memcpy static→dyn MUST be the ctor's first action — @__wb_cipher's body and every per-site load target @__wb_dyn, which is zero in .bss until this memcpy fills it; anything that reads @__wb_dyn before this runs sees zeros and crashes
        b.CreateMemCpy(s.gvdyn, MaybeAlign(64), s.gvstatic, MaybeAlign(64), b.getInt64(static_cast<uint64_t>(s.packedcipher.size())));

        Function* rdtscfn = Intrinsic::getDeclaration(&m, Intrinsic::readcyclecounter);
        seedval = b.CreateCall(rdtscfn, {}, "WbInitSeed");

        {
            constexpr uint64_t salt = 0x5E6C1A17C0DEE57BULL;
            Value* sgmix = b.CreateXor(seedval, b.getInt64(salt), "WbSgMix");
            Value* s0 = b.CreateLShr(sgmix, b.getInt64(33));
            Value* x0 = b.CreateXor(sgmix, s0);
            Value* m0 = b.CreateMul(x0, b.getInt64(kMixerMul1));
            Value* s1 = b.CreateLShr(m0, b.getInt64(33));
            Value* x1 = b.CreateXor(m0, s1);
            Value* m1 = b.CreateMul(x1, b.getInt64(kMixerMul2));
            Value* s2 = b.CreateLShr(m1, b.getInt64(33));
            Value* freshsg = b.CreateXor(m1, s2, "WbFreshSg");

            Value* sgptr = b.CreateInBoundsGEP(i8, s.gvdyn, b.getInt64(kSeedOffset), "WbSgPtr");
            StoreInst* sgstore = b.CreateAlignedStore(freshsg, sgptr, Align(8));
            sgstore->setVolatile(true);
        }

        if (fnregeninenc != nullptr)
            b.CreateCall(fnregeninenc, {seedval});

        b.CreateBr(loopBB);
    }

    PHINode* slotphi = nullptr;
    {
        IRBuilder<> b(loopBB);
        slotphi = b.CreatePHI(i64, 2, "Slot");
        slotphi->addIncoming(b.getInt64(0), entryBB);
        Value* cmp = b.CreateICmpULT(slotphi, b.getInt64(slotcount), "SlotLtCap");
        b.CreateCondBr(cmp, bodyBB, exitBB);
    }

    {
        IRBuilder<> b(bodyBB);
        const uint64_t bctstride = static_cast<uint64_t>(BctSlotBytes(s));
        const uint64_t rw0stride = static_cast<uint64_t>(Rw0SlotBytes(s));
        const bool bbwide = (static_cast<unsigned>(s.blockbytes) > static_cast<unsigned>(kBaseBlockBytes));

        auto staffordMix13 = [&](Value* x) -> Value*
        {
            Value* s0 = b.CreateLShr(x, b.getInt64(33));
            Value* x0 = b.CreateXor(x, s0);
            Value* m0 = b.CreateMul(x0, b.getInt64(kMixerMul1));
            Value* s1 = b.CreateLShr(m0, b.getInt64(33));
            Value* x1 = b.CreateXor(m0, s1);
            Value* m1 = b.CreateMul(x1, b.getInt64(kMixerMul2));
            Value* s2 = b.CreateLShr(m1, b.getInt64(33));
            return b.CreateXor(m1, s2);
        };

        Value* mix = b.CreateXor(seedval, slotphi, "WbInitMix");
        Value* freshalphalo = staffordMix13(mix);
        Value* freshalphahi = b.getInt64(0);
        if (bbwide)
        {
            constexpr uint64_t alphahisalt = 0xA177A1F70047C4C3ULL;
            Value* mixhi = b.CreateXor(b.CreateXor(seedval, slotphi), b.getInt64(alphahisalt), "WbInitMixHi");
            freshalphahi = staffordMix13(mixhi);
        }

        Value* bctlooff = b.CreateMul(slotphi, b.getInt64(bctstride));
        Value* bctloptr = b.CreateInBoundsGEP(i8, s.gvbct, bctlooff);
        LoadInst* bctloloaded = b.CreateAlignedLoad(i64, bctloptr, Align(8), "WbInitBctLoRaw");
        bctloloaded->setVolatile(true);
        Value* bctlo = (fndecrypt != nullptr) ? static_cast<Value*>(b.CreateCall(fndecrypt, {bctloloaded}, "WbInitBct")) : static_cast<Value*>(bctloloaded);

        Value* bcthi = b.getInt64(0);
        if (bbwide)
        {
            Value* bcthioff = b.CreateAdd(bctlooff, b.getInt64(8));
            Value* bcthiptr = b.CreateInBoundsGEP(i8, s.gvbct, bcthioff);
            LoadInst* bcthiloaded = b.CreateAlignedLoad(i64, bcthiptr, Align(8), "WbInitBctHi");
            bcthiloaded->setVolatile(true);
            bcthi = bcthiloaded;
        }

        Value* newmaskedlo = b.CreateXor(bctlo, freshalphalo, "WbInitMaskedLo");
        Value* newmaskedhi = b.getInt64(0);
        if (bbwide)
            newmaskedhi = b.CreateXor(bcthi, freshalphahi, "WbInitMaskedHi");

        Value* rw0baseoff = b.CreateMul(slotphi, b.getInt64(rw0stride));
        if (!bbwide)
        {
            Value* rw0mptr = b.CreateInBoundsGEP(i8, s.gvrw0, rw0baseoff);
            Value* rw0aptr = b.CreateInBoundsGEP(i8, s.gvrw0, b.CreateAdd(rw0baseoff, b.getInt64(8)));
            StoreInst* sim = b.CreateAlignedStore(newmaskedlo, rw0mptr, Align(8));
            sim->setVolatile(true);
            StoreInst* sia = b.CreateAlignedStore(freshalphalo, rw0aptr, Align(8));
            sia->setVolatile(true);
        }
        else
        {
            Value* rw0mloptr = b.CreateInBoundsGEP(i8, s.gvrw0, rw0baseoff);
            Value* rw0mhiptr = b.CreateInBoundsGEP(i8, s.gvrw0, b.CreateAdd(rw0baseoff, b.getInt64(8)));
            Value* rw0aloptr = b.CreateInBoundsGEP(i8, s.gvrw0, b.CreateAdd(rw0baseoff, b.getInt64(16)));
            Value* rw0ahiptr = b.CreateInBoundsGEP(i8, s.gvrw0, b.CreateAdd(rw0baseoff, b.getInt64(24)));
            StoreInst* simlo = b.CreateAlignedStore(newmaskedlo, rw0mloptr, Align(8));
            simlo->setVolatile(true);
            StoreInst* simhi = b.CreateAlignedStore(newmaskedhi, rw0mhiptr, Align(8));
            simhi->setVolatile(true);
            StoreInst* sialo = b.CreateAlignedStore(freshalphalo, rw0aloptr, Align(8));
            sialo->setVolatile(true);
            StoreInst* siahi = b.CreateAlignedStore(freshalphahi, rw0ahiptr, Align(8));
            siahi->setVolatile(true);
        }

        Value* slotnext = b.CreateAdd(slotphi, b.getInt64(1), "SlotNext");
        slotphi->addIncoming(slotnext, bodyBB);
        b.CreateBr(loopBB);
    }

    {
        IRBuilder<> b(exitBB);
        b.CreateRetVoid();
    }

    appendToGlobalCtors(m, f, 1000);
    return f;
}

static void AnchorInUsed(Module& m, std::initializer_list<GlobalValue*> values)
{
    SmallVector<GlobalValue*, 8> v(values.begin(), values.end());
    appendToUsed(m, v);
}

bool RunWhiteboxEnc(Module& m)
{
    if (!WhiteboxEnable) return false;
    if (!AnyFunctionAnnotated(m)) return false;
    if (m.getGlobalVariable("__wb_static")) return false;

    WbModuleState s;
    BuildCipherState(s);

    s.gvstatic = InstallByteArray(m, "__wb_static", s.packedcipher.size(), s.packedcipher.data(), true);
    s.gvdyn = InstallByteArray(m, "__wb_dyn", s.packedcipher.size(), nullptr, false);
    // setExternallyInitialized(true) stops GlobalOpt propagating @__wb_static's constant bytes into @__wb_dyn loads and eliding the ctor's memcpy. once the ctor writes fresh content into @__wb_dyn, folded static reads would be stale and the metamorphism plumbing silently breaks
    s.gvdyn->setExternallyInitialized(true);
    s.gvrw0 = InstallByteArray(m, "__wb_rw0", kMaxSites * Rw0SlotBytes(s), nullptr, false);
    s.gvbct = InstallByteArray(m, "__wb_bct", kMaxSites * BctSlotBytes(s), nullptr, false);
    s.gvbetas = InstallByteArray(m, "__wb_betas", kMaxSites * BetasSlotBytes(s), nullptr, false);

    s.gvlive = s.gvdyn;

    s.fncipher = EmitCipherFunction(m, s);

    AnchorInUsed(m, {s.gvstatic, s.gvdyn, s.gvrw0, s.gvbct, s.gvbetas, s.fncipher});

    std::vector<WbSite> allsites;
    std::mt19937_64 sitereng(s.seed ^ 0x517E517E57B105EEULL);

    for (Function& f : m)
    {
        if (f.isDeclaration()) continue;
        if (!HasObfuscateAnnotation(f)) continue;
        if (f.getName() == "__wb_cipher") continue;

        if (allsites.size() >= kMaxSites) break;

        std::vector<WbCandidate> candidates;
        CollectCandidatesInFunction(f, candidates);
        if (candidates.empty()) continue;

        const unsigned fncap = 64;
        unsigned rewritteninfn = 0;

        for (const WbCandidate& cand : candidates)
        {
            if (rewritteninfn >= fncap) break;
            if (allsites.size() >= kMaxSites) break;

            WbSite site{};
            if (!ClassifyCandidate(cand, site, static_cast<uint32_t>(allsites.size()), s, sitereng))
                continue;

            RewriteSite(m, s, site);
            allsites.push_back(site);
            ++rewritteninfn;
        }
    }

    if (!allsites.empty())
    {
        PatchBctInitializer(m, s, allsites);
        PatchBetasInitializer(m, s, allsites);
    }

    if (!allsites.empty())
    {
        Function* fnfy256 = nullptr;
        Function* fninvperm256 = nullptr;
        Function* fnregeninenc = nullptr;
        Function* fndecrypt = nullptr;
        if (s.freshinenc)
        {
            fnfy256 = EmitFy256(m);
            fninvperm256 = EmitInvperm256(m);
            fnregeninenc = EmitRegenInEnc(m, s, fnfy256, fninvperm256);
        }
        if (s.freshinenc && s.freshroundenc && s.freshciphermath)
            fndecrypt = EmitDecryptFn(m, s);

        Function* fnctor = EmitRuntimeInitCtor(m, s, fnregeninenc, fndecrypt, allsites.size());

        SmallVector<GlobalValue*, 5> extra;
        if (fnfy256) extra.push_back(fnfy256);
        if (fninvperm256) extra.push_back(fninvperm256);
        if (fnregeninenc) extra.push_back(fnregeninenc);
        if (fndecrypt) extra.push_back(fndecrypt);
        if (!extra.empty()) appendToUsed(m, extra);
        (void)fnctor;
    }

    return true;
}

PreservedAnalyses WhiteboxEncPass::run(Module& m, ModuleAnalysisManager&)
{
    const bool mutated = RunWhiteboxEnc(m);
    return mutated ? PreservedAnalyses::none() : PreservedAnalyses::all();
}