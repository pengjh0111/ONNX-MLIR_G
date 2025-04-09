#ifndef TOPOLOGICAL_SORT_H
#define TOPOLOGICAL_SORT_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetVector.h"
#include <queue>

#include "DependencyGraph.h"

using namespace mlir;

namespace onnx_mlir {

// Performs topological sort on the graph to identify parallel groups
void performTopologicalSort(DependencyGraph &graph);

// Utility function to print the topological levels (for debugging)
void dumpTopologicalLevels(DependencyGraph &graph);

} // end anonymous namespace

#endif // TOPOLOGICAL_SORT_H