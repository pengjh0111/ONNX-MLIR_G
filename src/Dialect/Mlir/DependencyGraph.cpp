#include "mlir/IR/Operation.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetVector.h"
#include <vector>
#include <set>

#include "DependencyGraph.h"

using namespace mlir;

namespace onnx_mlir {

// Implementation of DependencyGraph::addNode
DependencyNode* DependencyGraph::addNode(std::unique_ptr<DependencyNode> node) {
  DependencyNode* nodePtr = node.get();
  opToNodeMap[node->op] = nodePtr;
  nodes.push_back(std::move(node));
  return nodePtr;
}

// Implementation of DependencyGraph::addEdge
void DependencyGraph::addEdge(DependencyNode* from, DependencyNode* to) {
  outEdges[from].push_back(to);
  inEdges[to].push_back(from);
}

// Check if an operation is a kernel launch
bool isKernelLaunch(Operation* op) {
  return isa<gpu::LaunchFuncOp>(op);
}

// Check if an operation is a loop nest (outermost loop)
bool isLoopNest(Operation* op) {
  return isa<scf::ForOp>(op) && !op->getParentOfType<scf::ForOp>();
}

// Helper function: Find GPU function definition
gpu::GPUFuncOp findKernelFunc(gpu::LaunchFuncOp kernelOp) {
  // Get top-level module
  ModuleOp topModule = kernelOp->getParentOfType<ModuleOp>();
  if (!topModule) {
    return nullptr;
  }

  // Get kernel module and function name
  StringRef kernelModuleName = kernelOp.getKernelModuleName();
  StringRef kernelName = kernelOp.getKernelName();
  
  // First try to find the gpu.module
  gpu::GPUModuleOp gpuModule = nullptr;
  topModule.walk([&](gpu::GPUModuleOp module) {
    if (module.getName() == kernelModuleName) {
      gpuModule = module;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  // If GPU module is found, search for kernel function within it
  if (gpuModule) {
    gpu::GPUFuncOp kernelFunc = nullptr;
    gpuModule.walk([&](gpu::GPUFuncOp func) {
      if (func.getName() == kernelName) {
        kernelFunc = func;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    
    if (kernelFunc)
      return kernelFunc;
  }
  
  // Fallback: Search throughout the entire top-level module
  gpu::GPUFuncOp result = nullptr;
  topModule.walk([&](gpu::GPUFuncOp func) {
    if (func.getName() == kernelName) {
      result = func;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  return result;
}

// Extract memref inputs and outputs from a kernel launch
// void extractKernelDependencies(gpu::LaunchFuncOp kernelOp, 
//                               llvm::SetVector<Value> &inputs,
//                               llvm::SetVector<Value> &outputs) {
//   // Process kernel operands
//   for (auto arg : kernelOp.getKernelOperands()) {
//     if (arg.getType().isa<MemRefType>()) {
//       // For simplicity, treat all memrefs as both inputs and outputs
//       // A more sophisticated approach would analyze the kernel function
//       inputs.insert(arg);
//       outputs.insert(arg);
//     }
//   }
// }
// Extract memref inputs and outputs from a kernel launch, analyze kernel function definition
void extractKernelDependencies(gpu::LaunchFuncOp kernelOp, 
                              llvm::SetVector<Value> &inputs,
                              llvm::SetVector<Value> &outputs) {
  // Find kernel function definition
  gpu::GPUFuncOp kernelFunc = findKernelFunc(kernelOp);
  
  if (!kernelFunc) {
    // If function not found, fall back to conservative analysis
    llvm::errs() << "Warning: Could not find kernel function definition for \"" 
                << kernelOp.getKernelName() << "\", using conservative dependency analysis\n";
    for (auto arg : kernelOp.getKernelOperands()) {
      if (arg.getType().isa<MemRefType>()) {
        inputs.insert(arg);
        outputs.insert(arg);
      }
    }
    return;
  }

  // llvm::errs() << "Found kernel function definition for \"" << kernelOp.getKernelName() << "\"\n";
  
  llvm::SmallVector<std::pair<BlockArgument, Value>, 8> argOperandPairs;
  
  // Count the number of MemRef type parameters and operands
  unsigned memrefArgCount = 0;
  for (unsigned i = 0; i < kernelFunc.getNumArguments(); ++i) {
    if (kernelFunc.getArgument(i).getType().isa<MemRefType>()) {
      memrefArgCount++;
    }
  }
  
  unsigned memrefOpCount = 0;
  llvm::SmallVector<Value, 8> memrefOperands;
  for (auto operand : kernelOp.getKernelOperands()) {
    if (operand.getType().isa<MemRefType>()) {
      memrefOperands.push_back(operand);
      memrefOpCount++;
    }
  }
  
  // Debug information
  // llvm::errs() << "  MemRef arguments: " << memrefArgCount << ", MemRef operands: " << memrefOpCount << "\n";
  
  // Traverse all MemRef type function parameters
  unsigned opIdx = 0;
  for (unsigned i = 0; i < kernelFunc.getNumArguments(); ++i) {
    BlockArgument arg = kernelFunc.getArgument(i);
    if (arg.getType().isa<MemRefType>()) {
      // Ensure operand index is within valid range
      if (opIdx < memrefOperands.size()) {
        Value operand = memrefOperands[opIdx++];
        argOperandPairs.push_back({arg, operand});
        
        // Debug output
        // llvm::errs() << "  Mapped arg " << i << " to operand:\n    ";
        // arg.print(llvm::errs());
        // llvm::errs() << " -> ";
        // operand.print(llvm::errs());
        // llvm::errs() << "\n";
      }
    }
  }
  
  // Track which parameters are used for load and store
  llvm::DenseSet<BlockArgument> loadArgs;
  llvm::DenseSet<BlockArgument> storeArgs;
  
  // Analyze memory operations in kernel function body
  unsigned loadCount = 0, storeCount = 0;
  kernelFunc.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      // Check if the loaded memref is a function parameter
      Value memref = loadOp.getMemref();
      if (auto blockArg = dyn_cast<BlockArgument>(memref)) {
        if (blockArg.getOwner() == &kernelFunc.getBody().front()) {
          loadArgs.insert(blockArg);
          loadCount++;
          // Debug information
          // llvm::errs() << "  Found load from arg " << blockArg.getArgNumber() << "\n";
        }
      }
    } 
    else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      // Check if the stored memref is a function parameter
      Value memref = storeOp.getMemref();
      if (auto blockArg = dyn_cast<BlockArgument>(memref)) {
        if (blockArg.getOwner() == &kernelFunc.getBody().front()) {
          storeArgs.insert(blockArg);
          storeCount++;
          // Debug information
          // llvm::errs() << "  Found store to arg " << blockArg.getArgNumber() << "\n";
        }
      }
    }
  });
  
  // llvm::errs() << "  Found " << loadCount << " loads and " << storeCount << " stores in kernel\n";
  
  // Map function parameter analysis to kernel operands
  for (auto &pair : argOperandPairs) {
    BlockArgument arg = pair.first;
    Value operand = pair.second;
    
    bool isInput = loadArgs.count(arg) > 0;
    bool isOutput = storeArgs.count(arg) > 0;
    
    // llvm::errs() << "  Arg " << arg.getArgNumber() << ": input=" << (isInput ? "yes" : "no") 
    //             << ", output=" << (isOutput ? "yes" : "no") << "\n";
    
    if (isInput) {
      inputs.insert(operand);
    }
    
    if (isOutput) {
      outputs.insert(operand);
    }
    
    // If the parameter is neither loaded nor stored, treat it as input to be conservative
    if (!isInput && !isOutput) {
      inputs.insert(operand);
      llvm::errs() << "  Conservative: treating unused arg " << arg.getArgNumber() << " as input\n";
    }
  }
  
  // llvm::errs() << "  Final analysis - Inputs: " << inputs.size() 
  //             << ", Outputs: " << outputs.size() << "\n";
}

// Extract memref inputs and outputs from a loop nest
void extractLoopDependencies(scf::ForOp loopOp,
                           llvm::SetVector<Value> &inputs,
                           llvm::SetVector<Value> &outputs) {
  // Walk through the loop body to find all memref accesses
  loopOp.walk([&](Operation* op) {
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      inputs.insert(loadOp.getMemref());
    } 
    else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      outputs.insert(storeOp.getMemref());
      // The stored value might also be a load from another memref
      if (auto loadOp = dyn_cast_or_null<memref::LoadOp>(
          storeOp.getValue().getDefiningOp())) {
        inputs.insert(loadOp.getMemref());
      }
    }
  });
}

// Print text representation of the dependency graph
void dumpDependencyGraph(DependencyGraph &graph) {
  llvm::errs() << "===== Dependency Graph =====\n";
  
  // Print all nodes
  llvm::errs() << "Nodes (" << graph.nodes.size() << " total):\n";
  for (unsigned i = 0; i < graph.nodes.size(); ++i) {
    DependencyNode* node = graph.nodes[i].get();
    
    llvm::errs() << "  [" << i << "] ";
    if (node->type == NodeType::Kernel) {
      llvm::errs() << "Kernel: " << node->kernelModuleName << "::" << node->kernelName;
    } else {
      llvm::errs() << "Loop at: ";
      if (node->op) node->op->getLoc().print(llvm::errs());
      else llvm::errs() << "<unknown location>";
    }
    llvm::errs() << "\n";
    
    // Print input dependencies
    llvm::errs() << "    Inputs (" << node->inputs.size() << "):\n";
    for (Value input : node->inputs) {
      llvm::errs() << "      ";
      input.print(llvm::errs());
      llvm::errs() << "\n";
    }
    
    // Print output dependencies
    llvm::errs() << "    Outputs (" << node->outputs.size() << "):\n";
    for (Value output : node->outputs) {
      llvm::errs() << "      ";
      output.print(llvm::errs());
      llvm::errs() << "\n";
    }
    
    // Print topological sort level (if calculated)
    if (node->topologicalLevel > 0) {
      llvm::errs() << "    Topological Level: " << node->topologicalLevel << "\n";
    }
    
    llvm::errs() << "\n";
  }
  
  // Print all edges
  llvm::errs() << "Edges:\n";
  for (unsigned i = 0; i < graph.nodes.size(); ++i) {
    DependencyNode* node = graph.nodes[i].get();
    
    // Get outgoing edges for this node
    if (graph.outEdges.count(node)) {
      const auto &edges = graph.outEdges[node];
      if (!edges.empty()) {
        llvm::errs() << "  From [" << i << "] to:\n";
        
        for (DependencyNode* target : edges) {
          // Find target node index
          for (unsigned j = 0; j < graph.nodes.size(); ++j) {
            if (graph.nodes[j].get() == target) {
              llvm::errs() << "    [" << j << "]";
              
              // Output shared memory references causing this dependency
              bool foundSharedMem = false;
              for (Value out : node->outputs) {
                for (Value in : target->inputs) {
                  if (out == in) {
                    if (!foundSharedMem) {
                      llvm::errs() << " via: ";
                      foundSharedMem = true;
                    } else {
                      llvm::errs() << ", ";
                    }
                    out.print(llvm::errs());
                  }
                }
              }
              
              llvm::errs() << "\n";
              break;
            }
          }
        }
      }
    }
  }
  
  llvm::errs() << "===========================\n";
}

// Build the dependency graph from a function
std::unique_ptr<DependencyGraph> buildDependencyGraph(func::FuncOp funcOp) {
  auto graph = std::make_unique<DependencyGraph>();
  
  // 创建程序顺序映射，记录每个操作的原始顺序
  llvm::DenseMap<Operation*, unsigned> programOrder;
  unsigned orderCounter = 0;
  
  // 遍历函数体收集操作顺序
  funcOp.walk([&](Operation* op) {
    programOrder[op] = orderCounter++;
  });

  // First pass: create nodes for all kernels and loop nests
  funcOp.walk([&](Operation* op) {
    if (isKernelLaunch(op)) {
      auto kernelOp = cast<gpu::LaunchFuncOp>(op);
      auto node = std::make_unique<DependencyNode>();
      node->type = NodeType::Kernel;
      node->op = op;
      node->kernelModuleName = kernelOp.getKernelModuleName();
      node->kernelName = kernelOp.getKernelName();
      
      extractKernelDependencies(kernelOp, node->inputs, node->outputs);
      graph->addNode(std::move(node));
    } 
    else if (isLoopNest(op)) {
      auto loopOp = cast<scf::ForOp>(op);
      auto node = std::make_unique<DependencyNode>();
      node->type = NodeType::Loop;
      node->op = op;
      
      extractLoopDependencies(loopOp, node->inputs, node->outputs);
      graph->addNode(std::move(node));
    }
  });
  
  // Second pass: create edges based on dependencies
  for (const auto &nodePair : graph->nodes) {
    DependencyNode* node = nodePair.get();
    
    // For each output of this node
    for (auto output : node->outputs) {
      // Check if it's an input to any other node
      for (const auto &otherNodePair : graph->nodes) {
        DependencyNode* otherNode = otherNodePair.get();
        if (otherNode == node) continue;
        
        // If this node's output is another node's input, add an edge
        if (otherNode->inputs.count(output) && 
            programOrder[node->op] < programOrder[otherNode->op]) { //确保依赖边满足原IR执行次序，避免出现虚假的依赖边
          graph->addEdge(node, otherNode);
        }
      }
    }
  }
  
  return graph;
}

} // end anonymous namespace