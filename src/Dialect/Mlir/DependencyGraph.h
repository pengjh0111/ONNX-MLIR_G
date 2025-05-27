// #ifndef DEPENDENCY_GRAPH_H
// #define DEPENDENCY_GRAPH_H

// #include "mlir/IR/Operation.h"
// #include "mlir/Dialect/GPU/IR/GPUDialect.h"
// #include "mlir/Dialect/SCF/IR/SCF.h"
// #include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h" 
// #include "llvm/ADT/DenseMap.h"
// #include "llvm/ADT/SmallVector.h"
// #include "llvm/ADT/SetVector.h"
// #include <vector>
// #include <memory>

// using namespace mlir;

// namespace onnx_mlir {

// // Node types in our dependency graph
// enum class NodeType {
//   Kernel,  // A GPU kernel launch
//   Loop     // A nested loop structure
// };

// // Representation of a node in our dependency graph
// struct DependencyNode {
//   NodeType type;
//   Operation* op;  // The main operation (kernel launch or outer loop)
  
//   // For kernels
//   StringRef kernelModuleName;
//   StringRef kernelName;
  
//   // For tracking dependencies
//   llvm::SetVector<Value> inputs;
//   llvm::SetVector<Value> outputs;
  
//   // For topological sorting
//   unsigned topologicalLevel = 0;
// };

// // The dependency graph
// struct DependencyGraph {
//   std::vector<std::unique_ptr<DependencyNode>> nodes;
//   llvm::DenseMap<Operation*, DependencyNode*> opToNodeMap;
  
//   // Edges represented as adjacency lists
//   llvm::DenseMap<DependencyNode*, llvm::SmallVector<DependencyNode*, 4>> outEdges;
//   llvm::DenseMap<DependencyNode*, llvm::SmallVector<DependencyNode*, 4>> inEdges;
  
//   // Add a node to the graph
//   DependencyNode* addNode(std::unique_ptr<DependencyNode> node);
  
//   // Add an edge between nodes
//   void addEdge(DependencyNode* from, DependencyNode* to);
// };

// // Function declarations
// bool isKernelLaunch(Operation* op);
// bool isLoopNest(Operation* op);

// void extractKernelDependencies(gpu::LaunchFuncOp kernelOp, 
//                               llvm::SetVector<Value> &inputs,
//                               llvm::SetVector<Value> &outputs);

// void extractLoopDependencies(scf::ForOp loopOp,
//                            llvm::SetVector<Value> &inputs,
//                            llvm::SetVector<Value> &outputs);

// void dumpDependencyGraph(DependencyGraph &graph);

// // Build the dependency graph from a function
// std::unique_ptr<DependencyGraph> buildDependencyGraph(func::FuncOp funcOp);

// } // end anonymous namespace

// #endif // DEPENDENCY_GRAPH_H

#ifndef DEPENDENCY_GRAPH_H
#define DEPENDENCY_GRAPH_H

#include "mlir/IR/Operation.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetVector.h"
#include <vector>
#include <memory>

using namespace mlir;

namespace onnx_mlir {

// Node types in the dependency graph
enum class NodeType {
  Kernel,   // GPU kernel launch
  Loop,     // SCF loop nest  
  CuLibs    // CuLibs wrapper function call
};

// Represents a node in the dependency graph
struct DependencyNode {
  NodeType type;
  Operation* op;  // The main operation (gpu.launch_func, scf.for, or culibs call)
  
  // For kernel nodes
  StringRef kernelModuleName;
  StringRef kernelName;
  
  // For culibs nodes
  StringRef culibsFunctionName;
  llvm::SmallVector<Operation*, 4> culibsOps;  // All related ops (create, call, sync, destroy)
  
  // Dependencies
  llvm::SetVector<Value> inputs;   // Input memrefs
  llvm::SetVector<Value> outputs;  // Output memrefs
  
  // Topological sort level (0 = unassigned)
  unsigned topologicalLevel = 0;
};

// The dependency graph structure
struct DependencyGraph {
  // All nodes in the graph
  llvm::SmallVector<std::unique_ptr<DependencyNode>, 16> nodes;
  
  // Edge lists
  llvm::DenseMap<DependencyNode*, llvm::SmallVector<DependencyNode*, 4>> outEdges;
  llvm::DenseMap<DependencyNode*, llvm::SmallVector<DependencyNode*, 4>> inEdges;
  
  // Mapping from operations to nodes
  llvm::DenseMap<Operation*, DependencyNode*> opToNodeMap;
  
  // Helper methods
  DependencyNode* addNode(std::unique_ptr<DependencyNode> node);
  void addEdge(DependencyNode* from, DependencyNode* to);
};

// Function declarations
std::unique_ptr<DependencyGraph> buildDependencyGraph(func::FuncOp funcOp);
void dumpDependencyGraph(DependencyGraph &graph);

// Helper functions
bool isKernelLaunch(Operation* op);
bool isLoopNest(Operation* op);
bool isCuLibsCall(Operation* op);
bool isCuLibsStreamCreate(Operation* op);
bool isCuLibsStreamSync(Operation* op);
bool isCuLibsStreamDestroy(Operation* op);

} // namespace onnx_mlir

#endif // DEPENDENCY_GRAPH_H