#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

namespace {
  
struct InsertGPUAllocPass 
    : public PassWrapper<InsertGPUAllocPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertGPUAllocPass)
  void runOnOperation() override;
  StringRef getArgument() const final { return "insert-gpu-alloc"; }
  StringRef getDescription() const final { 
    return "Convert host-side memref.alloc, memref.reinterpret_cast, and krnl.global derived memory values to gpu.alloc, "
           "and insert necessary copy operations (static case only)."; 
  }
};

static Value createGpuAllocAndCopy(OpBuilder &builder, Location loc, Value origVal,
                                   ArrayRef<Value> dims, bool doCopy) {
  // This pass does not support dynamic cases; if dims is not empty, report an error
  if (!dims.empty()) {
    llvm::report_fatal_error("ConvertHostToGpuAllocPass: Dynamic dimensions not supported");
  }

  auto origType = origVal.getType().cast<MemRefType>();
  
  if (!origType.getLayout().isIdentity()) {
    llvm::report_fatal_error("InsertGPUAllocsPass: Non-identity (non-contiguous) layouts are not supported");
  }

  auto allocType = MemRefType::get(
      origType.getShape(), 
      origType.getElementType(), 
      /*use the original layout*/ origType.getLayout(),
      origType.getMemorySpace());
  
  // Insert the gpu.alloc operation.
  auto gpuAlloc = builder.create<gpu::AllocOp>(loc, allocType, /*asyncToken=*/nullptr,
                                               /*asyncDependencies=*/std::nullopt,
                                               dims, /*symbolOperands=*/std::nullopt,
                                               /*hostShared=*/true);
  Value allocResult = gpuAlloc.getResult(0);

  // If a copy is needed, insert a memref.copy operation:
  // Copy the original data to the GPU allocated memory.
  if (doCopy) {
    builder.create<memref::CopyOp>(loc, origVal, allocResult);
  }
  
  // If the allocated type does not match the original type, insert a cast operation.
  if (allocType != origType) {
    allocResult = builder.create<memref::CastOp>(loc, origType, allocResult);
  }
  return allocResult;
}

/// Process an operand: For operands produced by memref.alloc, memref.reinterpret_cast, or "krnl.global",
/// call createGpuAllocAndCopy as appropriate to obtain a GPU memory variable;
/// if the operand comes from memref.alloc and does not require a copy, return a flag so that all users of memref.alloc
/// can be replaced immediately.
static Value processOperand(OpBuilder &builder, Operation *launchOp, Location loc, Value operand, bool &needUpdateUsers) {
  Operation *defOp = operand.getDefiningOp();
  llvm::SmallVector<Value, 4> dims;
  if (auto memType = operand.getType().dyn_cast<MemRefType>()) {
    for (unsigned i = 0, e = memType.getRank(); i < e; ++i) {
      if (memType.isDynamicDim(i))
        dims.push_back(builder.create<memref::DimOp>(loc, operand, i));
    }
    if (!dims.empty())
      llvm::report_fatal_error("InsertGPUAllocsPass: 不支持动态维度");
  }

  // if (defOp && (isa<memref::AllocOp>(defOp) ||
  //               isa<memref::ReinterpretCastOp>(defOp) ||
  //               defOp->getName().getStringRef() == "krnl.global")) {
  //   // For reinterpret_cast or krnl.global, a copy is required.
  //   bool doCopy = isa<memref::ReinterpretCastOp>(defOp) ||
  //                 (defOp->getName().getStringRef() == "krnl.global");
  //   // If coming from memref.alloc and no copy is needed, mark that all users need to be updated.
  //   if (!doCopy && isa<memref::AllocOp>(defOp)) {
  //     builder.setInsertionPoint(defOp);
  //     needUpdateUsers = true;
  //   } 
  //   else if(defOp->getName().getStringRef() == "krnl.global"){
  //     builder.setInsertionPoint(defOp);
  //     needUpdateUsers = false;
  //   }
  //   else {
  //     needUpdateUsers = false;
  //     builder.setInsertionPoint(launchOp);
  //   }
  //   return createGpuAllocAndCopy(builder, loc, operand, dims, doCopy);
  // }

  if (defOp && (isa<memref::AllocOp>(defOp) ||
                isa<memref::ReinterpretCastOp>(defOp) ||
                defOp->getName().getStringRef() == "krnl.global")) {
    bool doCopy = false;
    // For krnl.global, insert the copy at the same location as the original krnl.global op.
    if (defOp->getName().getStringRef() == "krnl.global") {
      doCopy = true;
      builder.setInsertionPointAfter(defOp);
      needUpdateUsers = false;
    }
    // For reinterpret_cast, insert copy within the launch op.
    else if (isa<memref::ReinterpretCastOp>(defOp)) {
      doCopy = true;
      builder.setInsertionPoint(launchOp);
      needUpdateUsers = false;
    }
    // For memref.alloc with no copy required, update users.
    else if (isa<memref::AllocOp>(defOp)) {
      doCopy = false;
      builder.setInsertionPoint(defOp);
      needUpdateUsers = true;
    }
    return createGpuAllocAndCopy(builder, loc, operand, dims, doCopy);
  }


  // In all other cases, return the original value directly.
  needUpdateUsers = false;
  return operand;
}

void InsertGPUAllocPass::runOnOperation() {
  ModuleOp module = getOperation();
  OpBuilder builder(module.getContext());

  // Traverse each gpu.launch_func
  module.walk([&](gpu::LaunchFuncOp launchOp) {
    auto operands = launchOp.getOperands();
    llvm::SmallVector<Value, 4> newOperands;
    bool launchChanged = false;

    for (Value operand : operands) {
      bool updateUsers = false;
      Operation *defOp = operand.getDefiningOp();
      if (defOp && (isa<memref::AllocOp>(defOp) ||
                    isa<memref::ReinterpretCastOp>(defOp) ||
                    defOp->getName().getStringRef() == "krnl.global")) {
        Value newVal = processOperand(builder, launchOp.getOperation(), launchOp.getLoc(), operand, updateUsers);
        newOperands.push_back(newVal);
        launchChanged = true;
        // If the operand comes from memref.alloc and does not require a data copy,
        // immediately replace all users of the original memref.alloc (excluding those already updated in launch_op)
        // and then delete the original memref.alloc.
        if (updateUsers) {
          Operation *allocOp = defOp;
          SmallVector<OpOperand*, 8> operandUses;
          for (OpOperand &use : allocOp->getResult(0).getUses())
            operandUses.push_back(&use);
          for (OpOperand *use : operandUses) {
            use->set(newVal);
          }
          allocOp->erase();
        }
      } else {
        newOperands.push_back(operand);
      }
    }
    if (launchChanged)
      launchOp.getOperation()->setOperands(newOperands);
  });
}

} // end anonymous namespace

namespace onnx_mlir {
  namespace krnl {
  
  std::unique_ptr<mlir::Pass> createInsertGPUAllocPass() {
      return std::make_unique<InsertGPUAllocPass>();
  }
  
  } // namespace krnl
  } // namespace onnx_mlir
  
static mlir::PassRegistration<InsertGPUAllocPass> pass;
