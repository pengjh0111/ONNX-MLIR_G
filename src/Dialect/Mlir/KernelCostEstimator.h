#ifndef KERNEL_COST_ESTIMATOR_H
#define KERNEL_COST_ESTIMATOR_H

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include <cmath>

namespace onnx_mlir {

class KernelCostEstimator {
public:
  // Main function to estimate kernel execution cost
  static double estimateKernelCost(mlir::gpu::GPUFuncOp funcOp);

private:
  // Counters for different operation types
  unsigned numLoads = 0;
  unsigned numStores = 0;
  unsigned numArithOps = 0;  // Basic arithmetic (add, sub, mul, etc)
  unsigned numComplexOps = 0; // More expensive ops (div, exp, etc)
  unsigned numControlOps = 0;  // Control flow operations
  
  // Memory access patterns
  bool hasCoalescedAccess = true;  // Optimistic assumption
  bool hasSharedMemory = false;
  
  // Data size approximation
  unsigned totalMemRefSize = 0;
  
  // Thread/block configuration
  unsigned blockDimX = 1, blockDimY = 1, blockDimZ = 1;
  unsigned gridDimX = 1, gridDimY = 1, gridDimZ = 1;
  
  // Analysis methods
  void analyzeFunction(mlir::gpu::GPUFuncOp funcOp);
  void extractDimensions(mlir::gpu::GPUFuncOp funcOp);
  void analyzeMemRefType(mlir::MemRefType memrefType);
  unsigned getElementSize(mlir::Type elementType);
  void analyzeOperation(mlir::Operation* op);
  void analyzeMemAccess(mlir::ValueRange indices);
  double calculateCost();
};

} // namespace onnx_mlir

#endif // KERNEL_COST_ESTIMATOR_H