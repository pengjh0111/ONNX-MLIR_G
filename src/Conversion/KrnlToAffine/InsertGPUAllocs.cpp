// #include "mlir/IR/Builders.h"
// #include "mlir/IR/Diagnostics.h"
// #include "mlir/IR/Operation.h"
// #include "mlir/IR/PatternMatch.h"
// #include "mlir/IR/OpDefinition.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/GPU/IR/GPUDialect.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Pass/Pass.h"

// #include "llvm/Support/ErrorHandling.h"

// using namespace mlir;

// namespace {
  
// struct InsertGPUAllocPass 
//     : public PassWrapper<InsertGPUAllocPass, OperationPass<ModuleOp>> {
// public:
//     MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertGPUAllocPass)
//   void runOnOperation() override;
//   StringRef getArgument() const final { return "insert-gpu-alloc"; }
//   StringRef getDescription() const final { 
//     return "Convert host-side memref.alloc, memref.reinterpret_cast, and krnl.global derived memory values to gpu.alloc, "
//            "and insert necessary copy operations (static case only). "
//            "Note: This pass must be used after --lower-krnl-region, otherwise it will be ineffective as "
//            "it depends on krnl regions being properly lowered to standard operations first."; 
//   }
//   private:
//   // Member function declaration
//   Value processOperand(OpBuilder &builder, Operation *contextOp, Location loc, Value operand, bool &needUpdateUsers);

//   // Global replacement map to avoid duplicate GPU allocations
//   DenseMap<Value, Value> replacementMap;
// };

// static Value createGpuAllocAndCopy(OpBuilder &builder, Location loc, Value origVal,
//                                    ArrayRef<Value> dims, bool doCopy) {
//   // This pass does not support dynamic cases; if dims is not empty, report an error
//   if (!dims.empty()) {
//     llvm::report_fatal_error("ConvertHostToGpuAllocPass: Dynamic dimensions not supported");
//   }

//   auto origType = origVal.getType().cast<MemRefType>();
  
//   if (!origType.getLayout().isIdentity()) {
//     llvm::report_fatal_error("InsertGPUAllocsPass: Non-identity (non-contiguous) layouts are not supported");
//   }

//   auto allocType = MemRefType::get(
//       origType.getShape(), 
//       origType.getElementType(), 
//       /*use the original layout*/ origType.getLayout(),
//       origType.getMemorySpace());
  
//   // Insert the gpu.alloc operation.
//   auto gpuAlloc = builder.create<gpu::AllocOp>(loc, allocType, /*asyncToken=*/nullptr,
//                                                /*asyncDependencies=*/std::nullopt,
//                                                dims, /*symbolOperands=*/std::nullopt,
//                                                /*hostShared=*/true);
//   Value allocResult = gpuAlloc.getResult(0);

//   // If a copy is needed, insert a memref.copy operation:
//   // Copy the original data to the GPU allocated memory.
//   if (doCopy) {
//     builder.create<memref::CopyOp>(loc, origVal, allocResult);
//   }
  
//   // If the allocated type does not match the original type, insert a cast operation.
//   if (allocType != origType) {
//     allocResult = builder.create<memref::CastOp>(loc, origType, allocResult);
//   }
//   return allocResult;
// }

// /// Process an operand: For operands produced by memref.alloc, memref.reinterpret_cast, or "krnl.global",
// /// call createGpuAllocAndCopy as appropriate to obtain a GPU memory variable;
// /// if the operand comes from memref.alloc and does not require a copy, return a flag so that all users of memref.alloc
// /// can be replaced immediately.
// Value InsertGPUAllocPass::processOperand(OpBuilder &builder, Operation *launchOp, Location loc, Value operand, bool &needUpdateUsers) {
//   Operation *defOp = operand.getDefiningOp();

//   // Check if we already have a replacement for this value
//   if (replacementMap.count(operand)) {
//     needUpdateUsers = false;
//     return replacementMap[operand];
//   }

//   llvm::SmallVector<Value, 4> dims;
//   if (auto memType = operand.getType().dyn_cast<MemRefType>()) {
//     for (unsigned i = 0, e = memType.getRank(); i < e; ++i) {
//       if (memType.isDynamicDim(i))
//         dims.push_back(builder.create<memref::DimOp>(loc, operand, i));
//     }
//     if (!dims.empty())
//       llvm::report_fatal_error("InsertGPUAllocsPass: Dynamic dimensions not supported");
//   }

//   // if (defOp && (isa<memref::AllocOp>(defOp) ||
//   //               isa<memref::ReinterpretCastOp>(defOp) ||
//   //               defOp->getName().getStringRef() == "krnl.global")) {
//   //   // For reinterpret_cast or krnl.global, a copy is required.
//   //   bool doCopy = isa<memref::ReinterpretCastOp>(defOp) ||
//   //                 (defOp->getName().getStringRef() == "krnl.global");
//   //   // If coming from memref.alloc and no copy is needed, mark that all users need to be updated.
//   //   if (!doCopy && isa<memref::AllocOp>(defOp)) {
//   //     builder.setInsertionPoint(defOp);
//   //     needUpdateUsers = true;
//   //   } 
//   //   else if(defOp->getName().getStringRef() == "krnl.global"){
//   //     builder.setInsertionPoint(defOp);
//   //     needUpdateUsers = false;
//   //   }
//   //   else {
//   //     needUpdateUsers = false;
//   //     builder.setInsertionPoint(launchOp);
//   //   }
//   //   return createGpuAllocAndCopy(builder, loc, operand, dims, doCopy);
//   // }

//   if (defOp && (isa<memref::AllocOp>(defOp) ||
//                 isa<memref::ReinterpretCastOp>(defOp) ||
//                 defOp->getName().getStringRef() == "krnl.global")) {
//     bool doCopy = false;
//     // For krnl.global, insert the copy at the same location as the original krnl.global op.
//     if (defOp->getName().getStringRef() == "krnl.global") {
//       doCopy = true;
//       builder.setInsertionPointAfter(defOp);
//       needUpdateUsers = false;
//     }
//     // For reinterpret_cast, insert copy within the launch op.
//     else if (isa<memref::ReinterpretCastOp>(defOp)) {
//       doCopy = true;
//       builder.setInsertionPoint(launchOp);
//       needUpdateUsers = false;
//     }
//     // For memref.alloc with no copy required, update users.
//     else if (isa<memref::AllocOp>(defOp)) {
//       // doCopy = false;
//       // builder.setInsertionPoint(defOp);
//       // needUpdateUsers = true;
//       doCopy = false;
//       auto funcOp = defOp->getParentOfType<func::FuncOp>();
//       if (!funcOp) {
//         llvm::report_fatal_error("memref.alloc op is not within a function");
//       }
//       builder.setInsertionPointToStart(&funcOp.getBody().front());
//       needUpdateUsers = true;
//     }
//     // return createGpuAllocAndCopy(builder, loc, operand, dims, doCopy);
//     Value newVal = createGpuAllocAndCopy(builder, loc, operand, dims, doCopy);
//     replacementMap[operand] = newVal;
//     return newVal;
//   }


//   // In all other cases, return the original value directly.
//   needUpdateUsers = false;
//   return operand;
// }

// void InsertGPUAllocPass::runOnOperation() {
//   ModuleOp module = getOperation();
//   OpBuilder builder(module.getContext());

//   // Traverse each gpu.launch_func
//   module.walk([&](gpu::LaunchFuncOp launchOp) {
//     auto operands = launchOp.getOperands();
//     llvm::SmallVector<Value, 4> newOperands;
//     bool launchChanged = false;

//     for (Value operand : operands) {
//       bool updateUsers = false;
//       Operation *defOp = operand.getDefiningOp();
//       if (defOp && (isa<memref::AllocOp>(defOp) ||
//                     isa<memref::ReinterpretCastOp>(defOp) ||
//                     defOp->getName().getStringRef() == "krnl.global")) {
//         Value newVal = processOperand(builder, launchOp.getOperation(), launchOp.getLoc(), operand, updateUsers);
//         newOperands.push_back(newVal);
//         launchChanged = true;
//         // If the operand comes from memref.alloc and does not require a data copy,
//         // immediately replace all users of the original memref.alloc (excluding those already updated in launch_op)
//         // and then delete the original memref.alloc.
//         if (updateUsers) {
//           Operation *allocOp = defOp;
//           SmallVector<OpOperand*, 8> operandUses;
//           for (OpOperand &use : allocOp->getResult(0).getUses())
//             operandUses.push_back(&use);
//           for (OpOperand *use : operandUses) {
//             use->set(newVal);
//           }
//           allocOp->erase();
//         }
//       } else {
//         newOperands.push_back(operand);
//       }
//     }
//     if (launchChanged)
//       launchOp.getOperation()->setOperands(newOperands);
//   });

//   // Second pass: Handle memref.extract_aligned_pointer_as_index operations for mgpuCudnn calls
//   module.walk([&](memref::ExtractAlignedPointerAsIndexOp extractOp) {
//     Value memref = extractOp.getSource();
//     Operation *defOp = memref.getDefiningOp();
    
//     // Check if this memref comes from krnl.global or memref.alloc
//     if (defOp && (defOp->getName().getStringRef() == "krnl.global" ||
//                   isa<memref::AllocOp>(defOp))) {
      
//       // Check if we've already created a replacement for this value
//       if (replacementMap.count(memref)) {
//         extractOp.setOperand(replacementMap[memref]);
//         return;
//       }
      
//       // Check if the result flows into an mgpuCudnn call
//       bool flowsToMgpuCudnn = false;
//       for (Operation *user : extractOp->getResult(0).getUsers()) {
//         // Trace through inttoptr and index_cast operations
//         while (user && (isa<arith::IndexCastOp>(user) || user->getName().getStringRef() == "llvm.inttoptr")) {
//           if (user->getNumResults() > 0 && !user->getResult(0).getUses().empty()) {
//             user = *user->getResult(0).getUsers().begin();
//           } else {
//             break;
//           }
//         }
        
//         // Check if we reached a call operation
//         if (user && isa<func::CallOp>(user)) {
//           auto callOp = cast<func::CallOp>(user);
//           if (callOp.getCallee().starts_with("mgpuCudnn") || callOp.getCallee().starts_with("mgpuCulibs")) {
//             flowsToMgpuCudnn = true;
//             break;
//           }
//         }
//       }
      
//       if (flowsToMgpuCudnn) {
//         // Create gpu.alloc and copy
//         bool doCopy = (defOp->getName().getStringRef() == "krnl.global");
//         bool updateUsers = false;
        
//         // Use processOperand to ensure we use the global replacement map
//         Value gpuMem = this->processOperand(builder, extractOp.getOperation(), extractOp.getLoc(), memref, updateUsers);
        
//         // Replace the operand of the extract operation
//         extractOp.setOperand(gpuMem);
        
//         // Handle memref.alloc replacement if needed
//         if (updateUsers && isa<memref::AllocOp>(defOp)) {
//           SmallVector<OpOperand*, 8> operandUses;
//           for (OpOperand &use : memref.getUses())
//             operandUses.push_back(&use);
//           for (OpOperand *use : operandUses) {
//             use->set(gpuMem);
//           }
//           defOp->erase();
//         }
//       }
//     }
//   });
// }

// } // end anonymous namespace

// namespace onnx_mlir {
//   namespace krnl {
  
//   std::unique_ptr<mlir::Pass> createInsertGPUAllocPass() {
//       return std::make_unique<InsertGPUAllocPass>();
//   }
  
//   } // namespace krnl
// } // namespace onnx_mlir
  
// static mlir::PassRegistration<InsertGPUAllocPass> pass;

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
    return "Convert host-side memref.alloc, memref.reinterpret_cast, krnl.global, and gpu.alloc host_shared "
           "to async gpu.alloc, and insert necessary async gpu.memcpy operations. "
           "Adds explicit synchronization to decouple memory operations from compute operations."; 
  }
  private:
  // Member function declaration
  Value processOperand(OpBuilder &builder, Location loc, Value operand, bool &needUpdateUsers);

  // Global replacement map to avoid duplicate GPU allocations
  DenseMap<Value, Value> replacementMap;
};

static Value createGpuAllocAndCopyAsync(OpBuilder &builder, Location loc, Value origVal,
                                      ArrayRef<Value> dims, bool doCopy) {
  // This pass does not support dynamic cases; if dims is not empty, report an error
  if (!dims.empty()) {
    llvm::report_fatal_error("InsertGPUAllocPass: Dynamic dimensions not supported");
  }

  auto origType = origVal.getType().cast<MemRefType>();
  
  if (!origType.getLayout().isIdentity()) {
    llvm::report_fatal_error("InsertGPUAllocPass: Non-identity (non-contiguous) layouts are not supported");
  }

  auto allocType = MemRefType::get(
      origType.getShape(), 
      origType.getElementType(), 
      /*use the original layout*/ origType.getLayout(),
      origType.getMemorySpace());
  
  // Create an initial token
  Value initialToken = builder.create<gpu::WaitOp>(
      loc, 
      builder.getType<gpu::AsyncTokenType>(),
      ValueRange{}).getAsyncToken();
  
  // Insert the async gpu.alloc operation without host_shared flag
  auto gpuAlloc = builder.create<gpu::AllocOp>(
      loc, allocType, 
      builder.getType<gpu::AsyncTokenType>(), // Return an async token
      ValueRange{initialToken}, // Use the initial token
      dims, /*symbolOperands=*/std::nullopt,
      /*hostShared=*/false); // No host_shared flag
  
  Value allocResult = gpuAlloc.getResult(0);
  Value allocToken = gpuAlloc.getAsyncToken();

  // If a copy is needed, insert async gpu.memcpy operation
  if (doCopy) {
    auto memcpyOp = builder.create<gpu::MemcpyOp>(
        loc, 
        builder.getType<gpu::AsyncTokenType>(),
        ValueRange{allocToken}, // Chain with allocation token
        allocResult, origVal);
    
    // Update token to the memcpy token
    allocToken = memcpyOp.getAsyncToken();
  }
  
  // Add explicit synchronization point
  builder.create<gpu::WaitOp>(loc, TypeRange{}, ValueRange{allocToken});
  
  // If the allocated type does not match the original type, insert a cast operation.
  if (allocType != origType) {
    allocResult = builder.create<memref::CastOp>(loc, origType, allocResult);
  }
  
  return allocResult;
}


// 添加辅助函数：检查是否为标量操作函数
static bool isScalarOperation(StringRef funcName) {
  return funcName.contains("AddScalar") || 
         funcName.contains("SubScalar") || 
         funcName.contains("MulScalar") || 
         funcName.contains("RSubScalar");
}

// 添加辅助函数：检查参数是否为标量参数
static bool isScalarParameter(func::CallOp callOp, Value paramValue) {
  if (!isScalarOperation(callOp.getCallee())) {
    return false;
  }
  
  // 对于标量操作函数，第二个参数（index 1）是标量参数
  // 函数签名：mgpuCudnnXXXScalar(input_ptr, scalar_ptr, output_ptr, n, c, h, w, stream)
  auto operands = callOp.getOperands();
  
  // 需要追踪 extract_aligned_pointer -> index_cast -> inttoptr -> call operand 的链
  // 找到该参数在call操作中的位置
  for (unsigned i = 0; i < operands.size(); ++i) {
    Value operand = operands[i];
    
    // 追踪指针操作链，看是否最终来源于我们关心的paramValue
    Value currentVal = operand;
    
    // 反向追踪：inttoptr -> index_cast -> extract_aligned_pointer
    if (auto intToPtrOp = currentVal.getDefiningOp<mlir::LLVM::IntToPtrOp>()) {
      currentVal = intToPtrOp.getArg();
      
      if (auto indexCastOp = currentVal.getDefiningOp<mlir::arith::IndexCastOp>()) {
        currentVal = indexCastOp->getOperand(0);
        
        if (auto extractOp = currentVal.getDefiningOp<mlir::memref::ExtractAlignedPointerAsIndexOp>()) {
          Value sourceMemref = extractOp.getSource();
          
          // 检查源memref是否与我们的paramValue匹配
          if (sourceMemref == paramValue) {
            // 找到了参数位置，检查是否为标量参数位置
            // 标量参数通常在第二个位置（index 1）
            return (i == 1);
          }
        }
      }
    }
  }
  
  return false;
}

/// Process an operand: For operands produced by memref.alloc, memref.reinterpret_cast, "krnl.global",
/// or gpu.alloc host_shared, call createGpuAllocAndCopyAsync as appropriate to obtain a GPU memory variable.
Value InsertGPUAllocPass::processOperand(OpBuilder &builder, Location loc, Value operand, bool &needUpdateUsers) {
  Operation *defOp = operand.getDefiningOp();

  // Check if we already have a replacement for this value
  if (replacementMap.count(operand)) {
    needUpdateUsers = false;
    return replacementMap[operand];
  }

  llvm::SmallVector<Value, 4> dims;
  if (auto memType = operand.getType().dyn_cast<MemRefType>()) {
    for (unsigned i = 0, e = memType.getRank(); i < e; ++i) {
      if (memType.isDynamicDim(i))
        dims.push_back(builder.create<memref::DimOp>(loc, operand, i));
    }
    if (!dims.empty())
      llvm::report_fatal_error("InsertGPUAllocsPass: Dynamic dimensions not supported");
  }

  if (defOp && (isa<memref::AllocOp>(defOp) ||
                isa<memref::ReinterpretCastOp>(defOp) ||
                defOp->getName().getStringRef() == "krnl.global" ||
                (isa<gpu::AllocOp>(defOp) && cast<gpu::AllocOp>(defOp).getHostShared()))) {
    bool doCopy = false;
    
    // For krnl.global or reinterpret_cast, need to copy data
    if (defOp->getName().getStringRef() == "krnl.global" || isa<memref::ReinterpretCastOp>(defOp)) {
      doCopy = true;
    }
    
    // For memref.alloc and gpu.alloc host_shared, update insertion point to function start
    if (isa<memref::AllocOp>(defOp) || (isa<gpu::AllocOp>(defOp) && cast<gpu::AllocOp>(defOp).getHostShared())) {
      auto funcOp = defOp->getParentOfType<func::FuncOp>();
      if (!funcOp || funcOp.isExternal() || funcOp.getBody().empty()) {
        // Skip external functions or functions without a body
        needUpdateUsers = false;
        return operand;
      }
      builder.setInsertionPointToStart(&funcOp.getBody().front());
      needUpdateUsers = true;
    } else {
      // For krnl.global, set insertion point after the op
      if (defOp->getName().getStringRef() == "krnl.global") {
        builder.setInsertionPointAfter(defOp);
      }
      needUpdateUsers = false;
    }
    
    // Create async GPU allocation and copy if needed with explicit synchronization
    Value newVal = createGpuAllocAndCopyAsync(builder, loc, operand, dims, doCopy);
    
    // Store the replacement in the map
    replacementMap[operand] = newVal;
    
    return newVal;
  }

  // In all other cases, return the original value directly.
  needUpdateUsers = false;
  return operand;
}

// void InsertGPUAllocPass::runOnOperation() {
//   ModuleOp module = getOperation();
//   OpBuilder builder(module.getContext());

//   // Clear the replacement map
//   replacementMap.clear();

//   // First pass: Process all functions to handle memory allocations at function start
//   module.walk([&](func::FuncOp funcOp) {
//     // Skip external functions (functions without a body)
//     if (funcOp.isExternal() || funcOp.getBody().empty())
//       return;
    
//     builder.setInsertionPointToStart(&funcOp.getBody().front());
    
//     // Process gpu.launch_func operations
//     llvm::SmallVector<gpu::LaunchFuncOp, 8> launchOps;
//     funcOp.walk([&](gpu::LaunchFuncOp op) {
//       launchOps.push_back(op);
//     });
    
//     for (auto launchOp : launchOps) {
//       auto operands = launchOp.getOperands();
//       llvm::SmallVector<Value, 4> newOperands;
//       bool launchChanged = false;

//       // Process each operand
//       for (Value operand : operands) {
//         bool updateUsers = false;
//         Operation *defOp = operand.getDefiningOp();
//         if (defOp && (isa<memref::AllocOp>(defOp) ||
//                     isa<memref::ReinterpretCastOp>(defOp) ||
//                     defOp->getName().getStringRef() == "krnl.global" ||
//                     (isa<gpu::AllocOp>(defOp) && cast<gpu::AllocOp>(defOp).getHostShared()))) {
          
//           // Process the operand, getting new GPU memory value
//           Value newVal = processOperand(builder, launchOp.getLoc(), operand, updateUsers);
          
//           newOperands.push_back(newVal);
//           launchChanged = true;
          
//           // If updating users is needed (for memref.alloc or gpu.alloc host_shared)
//           if (updateUsers) {
//             // Replace all users of the original allocation
//             SmallVector<OpOperand*, 8> operandUses;
//             for (OpOperand &use : operand.getUses())
//               operandUses.push_back(&use);
//             for (OpOperand *use : operandUses) {
//               use->set(newVal);
//             }
//             // Erase the original allocation op
//             defOp->erase();
//           }
//         } else {
//           newOperands.push_back(operand);
//         }
//       }
      
//       // Only update the operands if they changed
//       if (launchChanged) {
//         launchOp.getOperation()->setOperands(newOperands);
//       }
//     }
//   });

//   // Second pass: Handle memref.extract_aligned_pointer_as_index operations for CUDA library calls
//   module.walk([&](memref::ExtractAlignedPointerAsIndexOp extractOp) {
//     Value memref = extractOp.getSource();
//     Operation *defOp = memref.getDefiningOp();
    
//     // Check if this memref comes from krnl.global, memref.alloc or gpu.alloc with host_shared
//     bool isTargetOp = false;
    
//     if (defOp) {
//       if (defOp->getName().getStringRef() == "krnl.global" || 
//           isa<memref::AllocOp>(defOp) ||
//           (isa<gpu::AllocOp>(defOp) && cast<gpu::AllocOp>(defOp).getHostShared())) {
//         isTargetOp = true;
//       }
//     }
    
//     if (isTargetOp) {
//       // Check if we've already created a replacement for this value
//       if (replacementMap.count(memref)) {
//         extractOp.setOperand(replacementMap[memref]);  // Use GPU memory
//         return;
//       }
      
//       // Check if the result flows into a CUDA library call
//       bool flowsToGpuLibCall = false;
//       for (Operation *user : extractOp->getResult(0).getUsers()) {
//         // Trace through inttoptr and index_cast operations
//         while (user && (isa<arith::IndexCastOp>(user) || user->getName().getStringRef() == "llvm.inttoptr")) {
//           if (user->getNumResults() > 0 && !user->getResult(0).getUses().empty()) {
//             user = *user->getResult(0).getUsers().begin();
//           } else {
//             break;
//           }
//         }
        
//         // Check if we reached a call operation to a CUDA library function
//         if (user && isa<func::CallOp>(user)) {
//           auto callOp = cast<func::CallOp>(user);
//           if (callOp.getCallee().starts_with("mgpuCudnn") || 
//               callOp.getCallee().starts_with("mgpuCulibs") ||
//               callOp.getCallee().starts_with("mgpu")) {
//             flowsToGpuLibCall = true;
//             break;
//           }
//         }
//       }
      
//       if (flowsToGpuLibCall) {
//         // For gpu.alloc host_shared, replace directly
//         if (auto gpuAllocOp = dyn_cast_or_null<gpu::AllocOp>(defOp)) {
//           if (gpuAllocOp.getHostShared()) {
//             auto funcOp = gpuAllocOp->getParentOfType<func::FuncOp>();
//             if (!funcOp) {
//               llvm::report_fatal_error("gpu.alloc op is not within a function");
//             }
            
//             // Move replacement to the start of the function
//             builder.setInsertionPointToStart(&funcOp.getBody().front());
            
//             // Create new async GPU allocation without host_shared
//             Value initialToken = builder.create<gpu::WaitOp>(
//                 gpuAllocOp.getLoc(),
//                 builder.getType<gpu::AsyncTokenType>(),
//                 ValueRange{}).getAsyncToken();
                
//             auto newGpuAlloc = builder.create<gpu::AllocOp>(
//                 gpuAllocOp.getLoc(),
//                 gpuAllocOp.getMemref().getType(),
//                 builder.getType<gpu::AsyncTokenType>(),
//                 ValueRange{initialToken},
//                 gpuAllocOp.getDynamicSizes(),
//                 gpuAllocOp.getSymbolOperands(),
//                 /*hostShared=*/false);
            
//             // Add explicit synchronization 
//             builder.create<gpu::WaitOp>(
//                 gpuAllocOp.getLoc(), 
//                 TypeRange{}, 
//                 ValueRange{newGpuAlloc.getAsyncToken()});
            
//             // Replace and store in map
//             Value newMem = newGpuAlloc.getMemref();
//             replacementMap[memref] = newMem;
            
//             // Replace all uses
//             gpuAllocOp.getMemref().replaceAllUsesWith(newMem);
//             gpuAllocOp.erase();
            
//             // Update the extract op
//             extractOp.setOperand(newMem);
//           }
//         } else {
//           // For memref.alloc or krnl.global, process at function start
//           auto funcOp = extractOp->getParentOfType<func::FuncOp>();
//           if (!funcOp) {
//             llvm::report_fatal_error("extract operation is not within a function");
//           }
          
//           builder.setInsertionPointToStart(&funcOp.getBody().front());
          
//           // Process the operand to create GPU memory
//           bool updateUsers = false;
//           Value gpuMem = processOperand(builder, extractOp.getLoc(), memref, updateUsers);
          
//           // Replace the operand of the extract operation
//           extractOp.setOperand(gpuMem);
          
//           // Handle memref.alloc replacement if needed
//           if (updateUsers && isa<memref::AllocOp>(defOp)) {
//             defOp->getResult(0).replaceAllUsesWith(gpuMem);
//             defOp->erase();
//           }
//         }
//       }
//     }
//   });
// }
// 修改第二个pass的逻辑
void InsertGPUAllocPass::runOnOperation() {
  ModuleOp module = getOperation();
  OpBuilder builder(module.getContext());

  // Clear the replacement map
  replacementMap.clear();

  // First pass: Process all functions to handle memory allocations at function start
  module.walk([&](func::FuncOp funcOp) {
    // Skip external functions (functions without a body)
    if (funcOp.isExternal() || funcOp.getBody().empty())
      return;
    
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    
    // Process gpu.launch_func operations
    llvm::SmallVector<gpu::LaunchFuncOp, 8> launchOps;
    funcOp.walk([&](gpu::LaunchFuncOp op) {
      launchOps.push_back(op);
    });
    
    for (auto launchOp : launchOps) {
      auto operands = launchOp.getOperands();
      llvm::SmallVector<Value, 4> newOperands;
      bool launchChanged = false;

      // Process each operand
      for (Value operand : operands) {
        bool updateUsers = false;
        Operation *defOp = operand.getDefiningOp();
        if (defOp && (isa<memref::AllocOp>(defOp) ||
                    isa<memref::ReinterpretCastOp>(defOp) ||
                    defOp->getName().getStringRef() == "krnl.global" ||
                    (isa<gpu::AllocOp>(defOp) && cast<gpu::AllocOp>(defOp).getHostShared()))) {
          
          // Process the operand, getting new GPU memory value
          Value newVal = processOperand(builder, launchOp.getLoc(), operand, updateUsers);
          
          newOperands.push_back(newVal);
          launchChanged = true;
          
          // If updating users is needed (for memref.alloc or gpu.alloc host_shared)
          if (updateUsers) {
            // Replace all users of the original allocation
            SmallVector<OpOperand*, 8> operandUses;
            for (OpOperand &use : operand.getUses())
              operandUses.push_back(&use);
            for (OpOperand *use : operandUses) {
              use->set(newVal);
            }
            // Erase the original allocation op
            defOp->erase();
          }
        } else {
          newOperands.push_back(operand);
        }
      }
      
      // Only update the operands if they changed
      if (launchChanged) {
        launchOp.getOperation()->setOperands(newOperands);
      }
    }
  });

  // Second pass: Handle memref.extract_aligned_pointer_as_index operations for CUDA library calls
  module.walk([&](memref::ExtractAlignedPointerAsIndexOp extractOp) {
    Value memref = extractOp.getSource();
    Operation *defOp = memref.getDefiningOp();
    
    // Check if this memref comes from krnl.global, memref.alloc or gpu.alloc with host_shared
    bool isTargetOp = false;
    
    if (defOp) {
      if (defOp->getName().getStringRef() == "krnl.global" || 
          isa<memref::AllocOp>(defOp) ||
          (isa<gpu::AllocOp>(defOp) && cast<gpu::AllocOp>(defOp).getHostShared())) {
        isTargetOp = true;
      }
    }
    
    if (isTargetOp) {
      // Check if we've already created a replacement for this value
      if (replacementMap.count(memref)) {
        extractOp.setOperand(replacementMap[memref]);  // Use GPU memory
        return;
      }
      
      // Check if the result flows into a CUDA library call
      bool flowsToGpuLibCall = false;
      bool isScalarParam = false;  //新增：标记是否为标量参数
      
      for (Operation *user : extractOp->getResult(0).getUsers()) {
        // Trace through inttoptr and index_cast operations
        Operation* currentUser = user;
        while (currentUser && (isa<arith::IndexCastOp>(currentUser) || currentUser->getName().getStringRef() == "llvm.inttoptr")) {
          if (currentUser->getNumResults() > 0 && !currentUser->getResult(0).getUses().empty()) {
            currentUser = *currentUser->getResult(0).getUsers().begin();
          } else {
            break;
          }
        }
        
        // Check if we reached a call operation to a CUDA library function
        if (currentUser && isa<func::CallOp>(currentUser)) {
          auto callOp = cast<func::CallOp>(currentUser);
          if (callOp.getCallee().starts_with("mgpuCudnn") || 
              callOp.getCallee().starts_with("mgpuCulibs") ||
              callOp.getCallee().starts_with("mgpu")) {
            flowsToGpuLibCall = true;
            
            //新增：检查是否为标量参数
            isScalarParam = isScalarParameter(callOp, memref);
            
            break;
          }
        }
      }
      
      //修改：如果是标量参数，跳过GPU内存转换
      if (flowsToGpuLibCall && !isScalarParam) {
        // For gpu.alloc host_shared, replace directly
        if (auto gpuAllocOp = dyn_cast_or_null<gpu::AllocOp>(defOp)) {
          if (gpuAllocOp.getHostShared()) {
            auto funcOp = gpuAllocOp->getParentOfType<func::FuncOp>();
            if (!funcOp) {
              llvm::report_fatal_error("gpu.alloc op is not within a function");
            }
            
            // Move replacement to the start of the function
            builder.setInsertionPointToStart(&funcOp.getBody().front());
            
            // Create new async GPU allocation without host_shared
            Value initialToken = builder.create<gpu::WaitOp>(
                gpuAllocOp.getLoc(),
                builder.getType<gpu::AsyncTokenType>(),
                ValueRange{}).getAsyncToken();
                
            auto newGpuAlloc = builder.create<gpu::AllocOp>(
                gpuAllocOp.getLoc(),
                gpuAllocOp.getMemref().getType(),
                builder.getType<gpu::AsyncTokenType>(),
                ValueRange{initialToken},
                gpuAllocOp.getDynamicSizes(),
                gpuAllocOp.getSymbolOperands(),
                /*hostShared=*/false);
            
            // Add explicit synchronization 
            builder.create<gpu::WaitOp>(
                gpuAllocOp.getLoc(), 
                TypeRange{}, 
                ValueRange{newGpuAlloc.getAsyncToken()});
            
            // Replace and store in map
            Value newMem = newGpuAlloc.getMemref();
            replacementMap[memref] = newMem;
            
            // Replace all uses
            gpuAllocOp.getMemref().replaceAllUsesWith(newMem);
            gpuAllocOp.erase();
            
            // Update the extract op
            extractOp.setOperand(newMem);
          }
        } else {
          // For memref.alloc or krnl.global, process at function start
          auto funcOp = extractOp->getParentOfType<func::FuncOp>();
          if (!funcOp) {
            llvm::report_fatal_error("extract operation is not within a function");
          }
          
          builder.setInsertionPointToStart(&funcOp.getBody().front());
          
          // Process the operand to create GPU memory
          bool updateUsers = false;
          Value gpuMem = processOperand(builder, extractOp.getLoc(), memref, updateUsers);
          
          // Replace the operand of the extract operation
          extractOp.setOperand(gpuMem);
          
          // Handle memref.alloc replacement if needed
          if (updateUsers && isa<memref::AllocOp>(defOp)) {
            defOp->getResult(0).replaceAllUsesWith(gpuMem);
            defOp->erase();
          }
        }
      }
      //新增：如果是标量参数流向标量操作函数，保持CPU内存分配
      else if (flowsToGpuLibCall && isScalarParam) {
        // 对于标量参数，我们什么都不做，保持原有的CPU内存分配
        // 这样标量就会保留在CPU端，避免在CPU代码中解引用GPU指针的问题
        // llvm::outs() << "Keeping scalar parameter in CPU memory for function call\n";
      }
    }
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