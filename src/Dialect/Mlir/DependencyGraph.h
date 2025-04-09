#ifndef DEPENDENCY_GRAPH_H
#define DEPENDENCY_GRAPH_H

#include "mlir/IR/Operation.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h" 
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetVector.h"
#include <vector>
#include <memory>

using namespace mlir;

namespace onnx_mlir {

// Node types in our dependency graph
enum class NodeType {
  Kernel,  // A GPU kernel launch
  Loop     // A nested loop structure
};

// Representation of a node in our dependency graph
struct DependencyNode {
  NodeType type;
  Operation* op;  // The main operation (kernel launch or outer loop)
  
  // For kernels
  StringRef kernelModuleName;
  StringRef kernelName;
  
  // For tracking dependencies
  llvm::SetVector<Value> inputs;
  llvm::SetVector<Value> outputs;
  
  // For topological sorting
  unsigned topologicalLevel = 0;
};

// The dependency graph
struct DependencyGraph {
  std::vector<std::unique_ptr<DependencyNode>> nodes;
  llvm::DenseMap<Operation*, DependencyNode*> opToNodeMap;
  
  // Edges represented as adjacency lists
  llvm::DenseMap<DependencyNode*, llvm::SmallVector<DependencyNode*, 4>> outEdges;
  llvm::DenseMap<DependencyNode*, llvm::SmallVector<DependencyNode*, 4>> inEdges;
  
  // Add a node to the graph
  DependencyNode* addNode(std::unique_ptr<DependencyNode> node);
  
  // Add an edge between nodes
  void addEdge(DependencyNode* from, DependencyNode* to);
};

// Function declarations
bool isKernelLaunch(Operation* op);
bool isLoopNest(Operation* op);

void extractKernelDependencies(gpu::LaunchFuncOp kernelOp, 
                              llvm::SetVector<Value> &inputs,
                              llvm::SetVector<Value> &outputs);

void extractLoopDependencies(scf::ForOp loopOp,
                           llvm::SetVector<Value> &inputs,
                           llvm::SetVector<Value> &outputs);

void dumpDependencyGraph(DependencyGraph &graph);

// Build the dependency graph from a function
std::unique_ptr<DependencyGraph> buildDependencyGraph(func::FuncOp funcOp);

} // end anonymous namespace

#endif // DEPENDENCY_GRAPH_H