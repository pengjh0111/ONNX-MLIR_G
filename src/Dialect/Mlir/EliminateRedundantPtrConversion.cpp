#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define DEBUG_TYPE "redundant-ptr-conversion"

namespace {

class RedundantPtrConversionPattern : public OpRewritePattern<LLVM::IntToPtrOp> {
public:
  using OpRewritePattern<LLVM::IntToPtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::IntToPtrOp intToPtrOp, 
                               PatternRewriter &rewriter) const override {
    Value intValue = intToPtrOp.getOperand();
    
    auto ptrToIntOp = intValue.getDefiningOp<LLVM::PtrToIntOp>();
    if (!ptrToIntOp)
      return failure();
      
    Value originalPtr = ptrToIntOp.getOperand();
    
    Type finalPtrType = intToPtrOp.getType();
    
    if (originalPtr.getType() != finalPtrType) {
      LLVM_DEBUG(llvm::dbgs() << "Original pointer type doesn't match final pointer type\n");
      return failure();
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Found redundant ptr->int->ptr conversion\n");
    
    rewriter.replaceOp(intToPtrOp, originalPtr);
    
    if (ptrToIntOp->use_empty()) {
      rewriter.eraseOp(ptrToIntOp);
    }
    
    return success();
  }
};

struct RedundantPointerConversionPass
    : public PassWrapper<RedundantPointerConversionPass, OperationPass<ModuleOp>> {
  
  StringRef getArgument() const final { return "eliminate-redundant-ptr-conversions"; }
  StringRef getDescription() const final {
    return "Eliminate redundant llvm.ptr <-> i64 conversions";
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();
    
    LLVM_DEBUG(llvm::dbgs() << "Running RedundantPointerConversionPass\n");
    
    RewritePatternSet patterns(context);
    patterns.add<RedundantPtrConversionPattern>(context);
    
    if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Completed RedundantPointerConversionPass\n");
  }
};

} // end anonymous namespace


namespace onnx_mlir {
    std::unique_ptr<Pass> createRedundantPointerConversionPass() {
      return std::make_unique<RedundantPointerConversionPass>();
    }
} // namespace onnx_mlir

static mlir::PassRegistration<RedundantPointerConversionPass> pass;