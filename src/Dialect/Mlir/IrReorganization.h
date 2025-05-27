// #ifndef IR_REORGANIZATION_H
// #define IR_REORGANIZATION_H

// #include "mlir/IR/Operation.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/Dialect/GPU/IR/GPUDialect.h"
// #include "mlir/Dialect/SCF/IR/SCF.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "llvm/ADT/DenseMap.h"
// #include "llvm/ADT/SmallVector.h"
// #include <map>

// #include "DependencyGraph.h"

// using namespace mlir;

// namespace onnx_mlir {

// // Reorganize IR based on the topological levels
// void reorganizeIR(func::FuncOp funcOp, DependencyGraph &graph);

// // Reorganize GPU modules to group kernels by topological level
// void reorganizeGPUModules(ModuleOp moduleOp, DependencyGraph &graph);

// } // end anonymous namespace

// #endif // IR_REORGANIZATION_H


// #ifndef IR_REORGANIZATION_H
// #define IR_REORGANIZATION_H

// #include "mlir/IR/Operation.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/IR/IRMapping.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/GPU/IR/GPUDialect.h"
// #include "llvm/ADT/DenseSet.h"
// #include "llvm/ADT/SmallVector.h"
// #include "llvm/ADT/SetVector.h"
// #include <map>
// #include "llvm/ADT/DenseMap.h"

// #include "DependencyGraph.h"

// using namespace mlir;

// namespace onnx_mlir {

// // Main reorganization functions
// void reorganizeIR(func::FuncOp funcOp, DependencyGraph &graph);
// void reorganizeGPUModules(ModuleOp moduleOp, DependencyGraph &graph);

// // Helper functions for processing different node types
// Value processKernelNode(DependencyNode* node, OpBuilder& builder, IRMapping& mapper, 
//                        Value waitToken, llvm::DenseSet<Operation*>& processedOps);

// void processLoopNode(DependencyNode* node, OpBuilder& builder, IRMapping& mapper,
//                     llvm::SmallVector<Operation*, 16>& allocaOps,
//                     llvm::DenseSet<Operation*>& processedOps);

// void processCuLibsNode(DependencyNode* node, OpBuilder& builder, IRMapping& mapper,
//                       llvm::DenseSet<Operation*>& processedOps);

// // Helper functions for CuLibs processing
// void collectDependentOps(Operation* op, llvm::SetVector<Operation*>& requiredOps, 
//                         const llvm::DenseSet<Operation*>& processedOps);

// bool shouldMoveWithCuLibs(Operation* op);

// } // namespace onnx_mlir

// #endif // IR_REORGANIZATION_H



#ifndef IR_REORGANIZATION_H
#define IR_REORGANIZATION_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/DenseMap.h"
#include <map>

#include "DependencyGraph.h"

using namespace mlir;

namespace onnx_mlir {

// Main reorganization functions
void reorganizeIR(func::FuncOp funcOp, DependencyGraph &graph);
void reorganizeGPUModules(ModuleOp moduleOp, DependencyGraph &graph);

// Helper functions for processing different node types
Value processKernelNode(DependencyNode* node, OpBuilder& builder, IRMapping& mapper, 
                       Value waitToken, llvm::DenseSet<Operation*>& processedOps);

void processLoopNode(DependencyNode* node, OpBuilder& builder, IRMapping& mapper,
                    llvm::SmallVector<Operation*, 16>& allocaOps,
                    llvm::DenseSet<Operation*>& processedOps);

void processCuLibsNode(DependencyNode* node, OpBuilder& builder, IRMapping& mapper,
                      llvm::DenseSet<Operation*>& processedOps);

void processCuLibsNodeWithStream(DependencyNode* node, OpBuilder& builder, IRMapping& mapper,
                                llvm::DenseSet<Operation*>& processedOps, Value stream);

// Helper functions for CuLibs processing
void collectDependentOps(Operation* op, llvm::SetVector<Operation*>& requiredOps, 
                        const llvm::DenseSet<Operation*>& processedOps);

bool shouldMoveWithCuLibs(Operation* op);

} // namespace onnx_mlir

#endif // IR_REORGANIZATION_H