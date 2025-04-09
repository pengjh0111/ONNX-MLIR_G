#include "TopoLogicalSort.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <queue>

using namespace mlir;

namespace onnx_mlir {

// Performs topological sort on the graph to identify parallel groups
void performTopologicalSort(DependencyGraph &graph) {
  // Count in-degrees for each node
  llvm::DenseMap<DependencyNode*, unsigned> inDegree;
  
  // Initialize in-degrees
  for (const auto &nodePair : graph.nodes) {
    DependencyNode* node = nodePair.get();
    inDegree[node] = graph.inEdges.count(node) ? graph.inEdges[node].size() : 0;
  }
  
  // Queue of nodes with no incoming edges (in-degree = 0)
  std::queue<DependencyNode*> zeroInDegree;
  
  // Add all nodes with no incoming edges to the queue
  for (const auto &nodePair : graph.nodes) {
    DependencyNode* node = nodePair.get();
    if (inDegree[node] == 0) {
      zeroInDegree.push(node);
      node->topologicalLevel = 1; // First level
    }
  }
  
  // Current topological level
  unsigned currentLevel = 1;
  
  // Process nodes level by level
  while (!zeroInDegree.empty()) {
    // Get the number of nodes at this level
    unsigned levelSize = zeroInDegree.size();
    
    // Process all nodes at the current level
    for (unsigned i = 0; i < levelSize; i++) {
      DependencyNode* node = zeroInDegree.front();
      zeroInDegree.pop();
      
      // Process all outgoing edges
      if (graph.outEdges.count(node)) {
        for (DependencyNode* successor : graph.outEdges[node]) {
          // Decrease in-degree of successor
          if (--inDegree[successor] == 0) {
            // All dependencies are processed, add to next level
            successor->topologicalLevel = currentLevel + 1;
            zeroInDegree.push(successor);
          }
        }
      }
    }
    
    // Move to next level
    currentLevel++;
  }
  
  // Check for cycles (nodes not assigned a level)
  for (const auto &nodePair : graph.nodes) {
    DependencyNode* node = nodePair.get();
    if (node->topologicalLevel == 0) {
      llvm::errs() << "Warning: Cycle detected in dependency graph\n";
      
      // Assign a high level to break cycles
      node->topologicalLevel = 999; 
    }
  }
}

// Utility function to print the topological levels (for debugging)
void dumpTopologicalLevels(DependencyGraph &graph) {
  llvm::errs() << "Topological Levels:\n";
  
  // Group nodes by level
  llvm::DenseMap<unsigned, llvm::SmallVector<DependencyNode*, 8>> levelToNodes;
  
  for (const auto &nodePair : graph.nodes) {
    DependencyNode* node = nodePair.get();
    levelToNodes[node->topologicalLevel].push_back(node);
  }
  
  // Print each level
  for (const auto &level : levelToNodes) {
    llvm::errs() << "Level " << level.first << ":\n";
    for (DependencyNode* node : level.second) {
      if (node->type == NodeType::Kernel) {
        llvm::errs() << "  Kernel: " << node->kernelModuleName 
                     << "::" << node->kernelName << "\n";
      } else {
        llvm::errs() << "  Loop: " << *node->op << "\n";
      }
    }
  }
}

} // end anonymous namespace