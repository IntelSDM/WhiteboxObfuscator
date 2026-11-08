#pragma once

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"

struct WhiteboxEncPass : llvm::PassInfoMixin<WhiteboxEncPass>
{
    llvm::PreservedAnalyses run(llvm::Module& m, llvm::ModuleAnalysisManager&);
};

bool RunWhiteboxEnc(llvm::Module& m);

extern llvm::cl::opt<bool> WhiteboxEnable;