#ifndef IR_REORGANIZATION_H
#define IR_REORGANIZATION_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <map>

#include "DependencyGraph.h"

using namespace mlir;

namespace onnx_mlir {

// Reorganize IR based on the topological levels
void reorganizeIR(func::FuncOp funcOp, DependencyGraph &graph);

// Reorganize GPU modules to group kernels by topological level
void reorganizeGPUModules(ModuleOp moduleOp, DependencyGraph &graph);

} // end anonymous namespace

#endif // IR_REORGANIZATION_H