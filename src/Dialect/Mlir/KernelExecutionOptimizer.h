#ifndef KERNEL_EXECUTION_OPTIMIZER_H
#define KERNEL_EXECUTION_OPTIMIZER_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include <vector>

#include "DependencyGraph.h"

namespace onnx_mlir {

// Class for estimating GPU kernel execution costs
// class KernelCostEstimator {
// public:
//   // Estimate cost for a given GPU function
//   static double estimateKernelCost(mlir::gpu::GPUFuncOp funcOp);
  
// private:
//   // Operation counts
//   unsigned numLoads = 0;
//   unsigned numStores = 0;
//   unsigned numArithOps = 0;
//   unsigned numComplexOps = 0;
//   unsigned numControlOps = 0;
  
//   // Memory access patterns
//   bool hasCoalescedAccess = true;
//   bool hasSharedMemory = false;
//   unsigned totalMemRefSize = 0;
  
//   // Thread and block dimensions
//   unsigned blockDimX = 1, blockDimY = 1, blockDimZ = 1;
//   unsigned gridDimX = 1, gridDimY = 1, gridDimZ = 1;
  
//   // Analysis methods
//   void analyzeFunction(mlir::gpu::GPUFuncOp funcOp);
//   void extractDimensions(mlir::gpu::GPUFuncOp funcOp);
//   void analyzeMemRefType(mlir::MemRefType memrefType);
//   unsigned getElementSize(mlir::Type elementType);
//   void analyzeOperation(mlir::Operation* op);
//   void analyzeMemAccess(mlir::ValueRange indices);
//   double calculateCost();
// };

class KernelCostEstimator; 

// Utility class to represent a kernel or group of kernels for scheduling
class KernelScheduleUnit {
public:
  KernelScheduleUnit(DependencyNode* node, double cost)
      : nodes({node}), totalCost(cost), isGroup(false) {}
  
  // Create a group by combining two units
  static KernelScheduleUnit createGroup(const KernelScheduleUnit& unit1, 
                                       const KernelScheduleUnit& unit2);
  
  std::vector<DependencyNode*> nodes;
  double totalCost;
  bool isGroup;
  
private:
  // Private constructor for group creation
  KernelScheduleUnit() : totalCost(0), isGroup(true) {}
};

// Cost caches
using KernelCostCache = llvm::DenseMap<mlir::gpu::GPUFuncOp, double>;
using KernelFuncCache = llvm::DenseMap<std::pair<mlir::StringRef, mlir::StringRef>, mlir::gpu::GPUFuncOp>;

// Find kernel function definition
mlir::gpu::GPUFuncOp findGPUFunc(mlir::StringRef moduleName, mlir::StringRef funcName, 
                               mlir::ModuleOp topModule, KernelFuncCache& funcCache);

// Calculate costs for kernels at a level
std::vector<KernelScheduleUnit> calculateKernelCosts(
    const std::vector<DependencyNode*>& kernelsAtLevel,
    mlir::ModuleOp moduleOp,
    KernelCostCache& costCache,
    KernelFuncCache& funcCache);

// Optimize scheduling within a topological level
std::vector<std::vector<DependencyNode*>> optimizeKernelScheduling(
    const std::vector<DependencyNode*>& nodesAtLevel,
    mlir::ModuleOp moduleOp,
    KernelCostCache& costCache,
    KernelFuncCache& funcCache,
    float toleranceFactor = 1.1);

// Apply optimized scheduling to dependency graph
void applyOptimizedScheduling(
    mlir::func::FuncOp funcOp,
    DependencyGraph& graph,
    mlir::ModuleOp moduleOp);

// Main entry point for optimization
void optimizeKernelExecution(mlir::ModuleOp moduleOp);

// Create the optimization pass
// std::unique_ptr<mlir::Pass> createKernelExecutionOptimizerPass();

} // namespace onnx_mlir

#endif // KERNEL_EXECUTION_OPTIMIZER_H