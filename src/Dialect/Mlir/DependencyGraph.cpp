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

// 辅助函数：查找GPU函数定义
gpu::GPUFuncOp findKernelFunc(gpu::LaunchFuncOp kernelOp) {
  // 获取顶层模块
  ModuleOp topModule = kernelOp->getParentOfType<ModuleOp>();
  if (!topModule) {
    return nullptr;
  }

  // 获取kernel模块和函数名
  StringRef kernelModuleName = kernelOp.getKernelModuleName();
  StringRef kernelName = kernelOp.getKernelName();
  
  // 先尝试找到gpu.module
  gpu::GPUModuleOp gpuModule = nullptr;
  topModule.walk([&](gpu::GPUModuleOp module) {
    if (module.getName() == kernelModuleName) {
      gpuModule = module;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  
  // 如果找到GPU模块，在其中搜索kernel函数
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
  
  // 回退：在整个顶层模块中搜索
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
// 从内核启动中提取memref输入和输出，分析内核函数定义
void extractKernelDependencies(gpu::LaunchFuncOp kernelOp, 
                              llvm::SetVector<Value> &inputs,
                              llvm::SetVector<Value> &outputs) {
  // 查找kernel函数定义
  gpu::GPUFuncOp kernelFunc = findKernelFunc(kernelOp);
  
  if (!kernelFunc) {
    // 如果找不到函数，回退到保守分析
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
  
  // 映射函数参数到对应的launch操作数（改进版）
  llvm::SmallVector<std::pair<BlockArgument, Value>, 8> argOperandPairs;
  
  // 统计MemRef类型的参数和操作数数量
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
  
  // 调试信息
  // llvm::errs() << "  MemRef arguments: " << memrefArgCount << ", MemRef operands: " << memrefOpCount << "\n";
  
  // 遍历所有MemRef类型的函数参数
  unsigned opIdx = 0;
  for (unsigned i = 0; i < kernelFunc.getNumArguments(); ++i) {
    BlockArgument arg = kernelFunc.getArgument(i);
    if (arg.getType().isa<MemRefType>()) {
      // 确保操作数索引在有效范围内
      if (opIdx < memrefOperands.size()) {
        Value operand = memrefOperands[opIdx++];
        argOperandPairs.push_back({arg, operand});
        
        // 调试输出
        // llvm::errs() << "  Mapped arg " << i << " to operand:\n    ";
        // arg.print(llvm::errs());
        // llvm::errs() << " -> ";
        // operand.print(llvm::errs());
        // llvm::errs() << "\n";
      }
    }
  }
  
  // 跟踪哪些参数用于load和store
  llvm::DenseSet<BlockArgument> loadArgs;
  llvm::DenseSet<BlockArgument> storeArgs;
  
  // 分析kernel函数体中的内存操作
  unsigned loadCount = 0, storeCount = 0;
  kernelFunc.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      // 检查被加载的memref是否为函数参数
      Value memref = loadOp.getMemref();
      if (auto blockArg = dyn_cast<BlockArgument>(memref)) {
        if (blockArg.getOwner() == &kernelFunc.getBody().front()) {
          loadArgs.insert(blockArg);
          loadCount++;
          // 调试信息
          // llvm::errs() << "  Found load from arg " << blockArg.getArgNumber() << "\n";
        }
      }
    } 
    else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      // 检查被存储的memref是否为函数参数
      Value memref = storeOp.getMemref();
      if (auto blockArg = dyn_cast<BlockArgument>(memref)) {
        if (blockArg.getOwner() == &kernelFunc.getBody().front()) {
          storeArgs.insert(blockArg);
          storeCount++;
          // 调试信息
          // llvm::errs() << "  Found store to arg " << blockArg.getArgNumber() << "\n";
        }
      }
    }
  });
  
  // llvm::errs() << "  Found " << loadCount << " loads and " << storeCount << " stores in kernel\n";
  
  // 将函数参数分析映射到kernel操作数
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
    
    // 如果参数既没有被加载也没有被存储，保守起见当作输入
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

// 打印依赖图的文本表示
void dumpDependencyGraph(DependencyGraph &graph) {
  llvm::errs() << "===== Dependency Graph =====\n";
  
  // 打印所有节点
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
    
    // 打印输入依赖
    llvm::errs() << "    Inputs (" << node->inputs.size() << "):\n";
    for (Value input : node->inputs) {
      llvm::errs() << "      ";
      input.print(llvm::errs());
      llvm::errs() << "\n";
    }
    
    // 打印输出依赖
    llvm::errs() << "    Outputs (" << node->outputs.size() << "):\n";
    for (Value output : node->outputs) {
      llvm::errs() << "      ";
      output.print(llvm::errs());
      llvm::errs() << "\n";
    }
    
    // 打印拓扑排序级别（如果已计算）
    if (node->topologicalLevel > 0) {
      llvm::errs() << "    Topological Level: " << node->topologicalLevel << "\n";
    }
    
    llvm::errs() << "\n";
  }
  
  // 打印所有边
  llvm::errs() << "Edges:\n";
  for (unsigned i = 0; i < graph.nodes.size(); ++i) {
    DependencyNode* node = graph.nodes[i].get();
    
    // 获取此节点的出边
    if (graph.outEdges.count(node)) {
      const auto &edges = graph.outEdges[node];
      if (!edges.empty()) {
        llvm::errs() << "  From [" << i << "] to:\n";
        
        for (DependencyNode* target : edges) {
          // 查找目标节点的索引
          for (unsigned j = 0; j < graph.nodes.size(); ++j) {
            if (graph.nodes[j].get() == target) {
              llvm::errs() << "    [" << j << "]";
              
              // 输出导致此依赖的共享内存引用
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
        if (otherNode->inputs.count(output)) {
          graph->addEdge(node, otherNode);
        }
      }
    }
  }
  
  return graph;
}

} // end anonymous namespace