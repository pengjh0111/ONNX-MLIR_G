#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Debug.h"
#include <cmath>

#include "KernelCostEstimator.h"

using namespace mlir;

namespace onnx_mlir {

// Implementation of static method
double KernelCostEstimator::estimateKernelCost(gpu::GPUFuncOp funcOp) {
  KernelCostEstimator estimator;
  estimator.analyzeFunction(funcOp);
  return estimator.calculateCost();
}

// Implementation of analysis function
void KernelCostEstimator::analyzeFunction(gpu::GPUFuncOp funcOp) {
  // Try to extract block/grid dimensions from attributes
  // In practice, these might be in the launch function
  extractDimensions(funcOp);
  
  // Analyze memref parameters to estimate data size
  for (auto arg : funcOp.getArguments()) {
    if (auto memrefType = arg.getType().dyn_cast<MemRefType>()) {
      analyzeMemRefType(memrefType);
    }
  }
  
  // Traverse all operations in the function
  funcOp.walk([&](Operation *op) {
    analyzeOperation(op);
  });
}

// Implementation of dimension extraction
void KernelCostEstimator::extractDimensions(gpu::GPUFuncOp funcOp) {
  // Look for attributes that might contain dimensions
  // This is a simplified version - in actual code you would analyze the launch site
  
  // If not found, use reasonable default values
  blockDimX = 32;  // Common CUDA block size
  blockDimY = 4;
  gridDimX = 128;
  gridDimY = 128;
  
  // Check function attributes (if they exist)
  if (auto attr = funcOp->getAttrOfType<IntegerAttr>("blockDimX"))
    blockDimX = attr.getInt();
  if (auto attr = funcOp->getAttrOfType<IntegerAttr>("gridDimX"))
    gridDimX = attr.getInt();
  
  // Traverse the function body to find any BlockDim/GridDim usage
  funcOp.walk([&](gpu::BlockDimOp dimOp) {
    // If we find explicit block dim operations, adjust our estimate
    if (dimOp.getDimension() == gpu::Dimension::x)
      blockDimX = 32; // Common size
  });
}

// Implementation of memory reference type analysis
void KernelCostEstimator::analyzeMemRefType(MemRefType memrefType) {
  if (memrefType.hasStaticShape()) {
    unsigned size = 1;
    for (auto dim : memrefType.getShape()) {
      size *= dim;
    }
    totalMemRefSize += size * getElementSize(memrefType.getElementType());
  } else {
    // For dynamic shapes, use reasonable default values
    totalMemRefSize += 4096;
  }
  
  // Check for shared memory
  if (auto memSpace = memrefType.getMemorySpace()) {
    if (auto intAttr = memSpace.dyn_cast<mlir::IntegerAttr>()) {
      if (intAttr.getInt() == 3) { // Shared memory space
        hasSharedMemory = true;
      }
    }
  }
}

// Implementation of getting element size
unsigned KernelCostEstimator::getElementSize(Type elementType) {
  if (elementType.isF32())
    return 4;
  if (elementType.isF64())
    return 8;
  if (elementType.isInteger(32))
    return 4;
  if (elementType.isInteger(64))
    return 8;
  
  return 4; // Default size
}

// Implementation of analyzing a single operation
void KernelCostEstimator::analyzeOperation(Operation* op) {
  if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
    numLoads++;
    analyzeMemAccess(loadOp.getIndices());
  }
  else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
    numStores++;
    analyzeMemAccess(storeOp.getIndices());
  }
  else if (isa<arith::AddFOp>(op) || isa<arith::SubFOp>(op) || isa<arith::AddIOp>(op) || 
      isa<arith::SubIOp>(op) || isa<arith::MulFOp>(op) || isa<arith::MulIOp>(op) || 
      isa<arith::NegFOp>(op)) {
    numArithOps++;
  }
  else if (isa<arith::DivFOp>(op) || isa<arith::DivSIOp>(op) || isa<arith::DivUIOp>(op) || 
          isa<arith::RemFOp>(op) || isa<arith::RemSIOp>(op) || isa<arith::RemUIOp>(op)) {
    numComplexOps++; // More expensive operations
  }
  else if (isa<scf::ForOp>(op) || isa<scf::IfOp>(op) || isa<scf::WhileOp>(op)) {
    numControlOps++;
  }
  else if (isa<gpu::BarrierOp>(op) || isa<gpu::ShuffleOp>(op)) {
    numControlOps += 5; // Synchronization operations are expensive
  }
}

// Implementation of memory access pattern analysis
void KernelCostEstimator::analyzeMemAccess(ValueRange indices) {
  // Simplification: in an actual implementation, we would analyze
  // whether the innermost dimension is related to thread.x
  // in order to correctly detect coalesced access
  
  // Check if any indices use thread ID
  for (auto idx : indices) {
    // Look for thread_id_x as part of index calculation
    if (auto defOp = idx.getDefiningOp()) {
      // Simplified check - the actual case would be more complex
      if (isa<gpu::ThreadIdOp>(defOp)) {
        return; // Might be coalesced access
      }
      
      // Look for thread ID in affine expressions
      if (auto applyOp = dyn_cast<affine::AffineApplyOp>(defOp)) {
        // Very simplified - actual analysis would be more complex
        hasCoalescedAccess = true;
      }
    }
  }
  
  // If no thread ID correlation is found, it might be non-coalesced access
  hasCoalescedAccess = false;
}

// Implementation of calculating the final cost
double KernelCostEstimator::calculateCost() {
  unsigned totalThreads = blockDimX * blockDimY * blockDimZ * 
                         gridDimX * gridDimY * gridDimZ;
  
  // Memory operation cost (main contributor to execution time)
  double memCost = (numLoads + numStores) * (hasCoalescedAccess ? 1.0 : 4.0);
  if (hasSharedMemory)
    memCost *= 0.7; // Shared memory reduces cost
    
  // Computation cost
  double computeCost = numArithOps * 0.2 + numComplexOps * 1.0;
  
  // Control flow cost
  double controlCost = numControlOps * 2.0; // Control divergence is expensive
  
  // Data size impact
  double dataSizeFactor = std::log10(totalMemRefSize + 1) * 0.2;
  
  // Total cost formula
  double cost = (memCost + computeCost + controlCost) * (1.0 + dataSizeFactor);
  
  // Adjust based on thread count
  if (totalThreads > 0) {
    // We use logarithm to model diminishing returns of parallelism
    cost *= (1.0 + std::log10(totalThreads) * 0.1);
  }
  
  return cost;
}

} // namespace onnx_mlir