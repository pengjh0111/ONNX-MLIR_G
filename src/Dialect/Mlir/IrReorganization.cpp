#include "mlir/IR/Operation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <map>

#include "IrReorganization.h"
#include "DependencyGraph.h"

using namespace mlir;

namespace onnx_mlir {

// Fixed version of reorganizeIR function
void reorganizeIR(func::FuncOp funcOp, DependencyGraph &graph) {
  OpBuilder builder(funcOp.getContext());
  
  // Create mapping to track operation mapping relationships
  IRMapping mapper;
  
  // Group nodes by topological level
  std::map<unsigned, llvm::SmallVector<DependencyNode*, 8>> nodesByLevel;
  for (const auto &nodePair : graph.nodes) {
    DependencyNode* node = nodePair.get();
    nodesByLevel[node->topologicalLevel].push_back(node);
  }
  
  // Create new block
  Block* oldBlock = &funcOp.getBody().front();
  Block* newBlock = new Block();
  
  // Map arguments
  for (auto &blockArg : oldBlock->getArguments()) {
    auto newArg = newBlock->addArgument(blockArg.getType(), blockArg.getLoc());
    mapper.map(blockArg, newArg);
  }
  
  // Track processed operations
  llvm::DenseSet<Operation*> processedOps;
  
  // Collect all alloca operations, which need to be placed before use
  llvm::SmallVector<Operation*, 16> allocaOps;
  funcOp.walk([&](memref::AllocaOp allocaOp) {
    allocaOps.push_back(allocaOp);
  });
  
  // Phase 1: First copy non-graph node prefix operations, while handling all allocas
  for (auto &op : oldBlock->getOperations()) {
    if (graph.opToNodeMap.count(&op)) {
      // Stop copying prefix operations when encountering a node in the graph
      break;
    }
    
    // Skip all GPU wait operations, as we will add necessary wait points according to the dependency graph
    if (isa<gpu::WaitOp>(op)) {
      processedOps.insert(&op);
      continue;
    }
    
    // Record all alloca operations, to be processed together later
    if (isa<memref::AllocaOp>(op)) {
      processedOps.insert(&op);
      continue; // Skip for now, process later
    }
    
    Operation *newOp = op.clone(mapper);
    newBlock->push_back(newOp);
    
    // Update mapping and mark as processed
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      mapper.map(op.getResult(i), newOp->getResult(i));
    }
    processedOps.insert(&op);
  }
  
  // Find maximum topological level
  unsigned maxLevel = 0;
  for (const auto &nodePair : graph.nodes) {
    maxLevel = std::max(maxLevel, nodePair.get()->topologicalLevel);
  }
  
  // For tracking tokens from the final level
  llvm::SmallVector<Value, 8> finalLevelTokens;
  
  // Phase 2: Process nodes by topological level
  for (unsigned level = 1; level <= maxLevel; level++) {
    auto levelIt = nodesByLevel.find(level);
    if (levelIt == nodesByLevel.end() || levelIt->second.empty())
      continue;
      
    auto &nodesAtLevel = levelIt->second;
    
    // Collect async tokens for this level
    llvm::SmallVector<Value, 8> levelTokens;
    
    // Count kernels at current level
    unsigned kernelCount = 0;
    for (auto node : nodesAtLevel) {
      if (node->type == NodeType::Kernel) {
        kernelCount++;
      }
    }
    
    // Step 1: Create async wait tokens for all kernels
    llvm::SmallVector<Value, 8> waitTokens;
    if (kernelCount > 0) {
      builder.setInsertionPointToEnd(newBlock);
      
      for (unsigned i = 0; i < kernelCount; i++) {
        // Create async wait operation
        auto waitOp = builder.create<gpu::WaitOp>(
            funcOp.getLoc(),
            builder.getType<gpu::AsyncTokenType>(),
            ValueRange{});
        waitTokens.push_back(waitOp.getAsyncToken());
      }
    }
    
    // Step 2: Process all nodes at the current level
    unsigned kernelIndex = 0;  // Used to track which kernel is currently being processed
    
    for (auto node : nodesAtLevel) {
      builder.setInsertionPointToEnd(newBlock);
      
      if (node->type == NodeType::Kernel) {
        auto kernelOp = cast<gpu::LaunchFuncOp>(node->op);
        
        // Create kernel symbol reference
        auto kernelSymbol = SymbolRefAttr::get(
            builder.getContext(),
            kernelOp.getKernelModuleName(),
            {SymbolRefAttr::get(builder.getContext(), kernelOp.getKernelName())});
        
        // Map operands
        SmallVector<Value, 8> remappedOperands;
        for (Value operand : kernelOp.getKernelOperands()) {
          remappedOperands.push_back(mapper.lookupOrDefault(operand));
        }
        
        // Map grid and block sizes
        auto gridSize = kernelOp.getGridSizeOperandValues();
        auto blockSize = kernelOp.getBlockSizeOperandValues();
        
        mlir::gpu::KernelDim3 mappedGridSize = {
          mapper.lookupOrDefault(gridSize.x),
          mapper.lookupOrDefault(gridSize.y),
          mapper.lookupOrDefault(gridSize.z)
        };
        
        mlir::gpu::KernelDim3 mappedBlockSize = {
          mapper.lookupOrDefault(blockSize.x),
          mapper.lookupOrDefault(blockSize.y),
          mapper.lookupOrDefault(blockSize.z)
        };
        
        // Use previously created async wait token
        Value waitToken = waitTokens[kernelIndex++];
            
        // Directly create async kernel launch
        auto newLaunchOp = builder.create<gpu::LaunchFuncOp>(
            kernelOp.getLoc(),
            kernelSymbol,
            mappedGridSize,
            mappedBlockSize,
            Value(),  // No dynamic shared memory
            remappedOperands,
            builder.getType<gpu::AsyncTokenType>(),  // Async token type
            ValueRange{waitToken},  // Use previously created async wait token
            std::nullopt);  // No cluster size
            
        // Collect async tokens for this level
        levelTokens.push_back(newLaunchOp.getAsyncToken());
            
        // Map results
        if (kernelOp->getNumResults() > 0) {
          mapper.map(kernelOp->getResult(0), newLaunchOp->getResult(0));
        }
        
        // Mark as processed
        processedOps.insert(node->op);
      } 
      else if (node->type == NodeType::Loop) {
        // Find all memref.alloca operations associated with this loop
        llvm::SmallVector<Operation*, 8> loopLocalAllocas;
        for (auto allocaOp : allocaOps) {
          // Check if this alloca is used by this loop
          bool used = false;
          Value allocaResult = allocaOp->getResult(0);
          node->op->walk([&](Operation *user) {
            for (Value operand : user->getOperands()) {
              if (operand == allocaResult) {
                used = true;
                return WalkResult::interrupt();
              }
            }
            return WalkResult::advance();
          });
          
          if (used) {
            loopLocalAllocas.push_back(allocaOp);
            processedOps.insert(allocaOp); // Mark as processed
          }
        }
        
        // Recreate all local allocas before this loop
        for (auto allocaOp : loopLocalAllocas) {
          auto newAllocaOp = builder.clone(*allocaOp, mapper);
          
          // Update mapping
          for (unsigned i = 0; i < allocaOp->getNumResults(); ++i) {
            mapper.map(allocaOp->getResult(i), newAllocaOp->getResult(i));
          }
        }
        
        // Clone loop operation
        Operation *newOp = builder.clone(*node->op, mapper);
        
        // Update mapping
        for (unsigned i = 0; i < node->op->getNumResults(); ++i) {
          mapper.map(node->op->getResult(i), newOp->getResult(i));
        }
        
        // Mark as processed
        processedOps.insert(node->op);
      }
    }
    
    // If current level has operations
    if (!levelTokens.empty()) {
      // If not the last level, add a synchronization point
      if (level < maxLevel) {
        builder.setInsertionPointToEnd(newBlock);
        
        // Non-async wait - ensure all operations at this level complete before moving to the next level
        builder.create<gpu::WaitOp>(funcOp.getLoc(), TypeRange{}, levelTokens);
      } 
      // If it's the last level, save tokens to add a final sync point before function return
      else {
        finalLevelTokens = levelTokens;
      }
    }
  }
  
  // Process remaining unused alloca operations
  for (auto allocaOp : allocaOps) {
    if (!processedOps.count(allocaOp)) {
      builder.setInsertionPointToEnd(newBlock);
      auto newAllocaOp = builder.clone(*allocaOp, mapper);
      
      // Update mapping
      for (unsigned i = 0; i < allocaOp->getNumResults(); ++i) {
        mapper.map(allocaOp->getResult(i), newAllocaOp->getResult(i));
      }
      
      processedOps.insert(allocaOp);
    }
  }
  
  // Phase 3: Copy remaining unprocessed operations, but skip gpu.wait operations
  bool hasReturnOp = false;
  Operation* returnOp = nullptr;
  
  for (auto &op : oldBlock->getOperations()) {
    if (processedOps.count(&op))
      continue;  // Skip already processed operations
      
    // Skip all GPU wait operations
    if (isa<gpu::WaitOp>(op)) {
      processedOps.insert(&op);
      continue;
    }
    
    // If it's a return operation, don't clone it yet, process it later
    if (isa<func::ReturnOp>(op)) {
      hasReturnOp = true;
      returnOp = &op;
      continue;
    }
    
    Operation *newOp = op.clone(mapper);
    newBlock->push_back(newOp);
    
    // Update mapping
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      mapper.map(op.getResult(i), newOp->getResult(i));
    }
  }
  
  // If there are tokens from the final level, add a final sync point
  if (!finalLevelTokens.empty()) {
    builder.setInsertionPointToEnd(newBlock);
    
    // Add final synchronization wait
    builder.create<gpu::WaitOp>(funcOp.getLoc(), TypeRange{}, finalLevelTokens);
  }
  
  // If there's a return operation, clone it now
  if (hasReturnOp) {
    builder.setInsertionPointToEnd(newBlock);
    Operation *newReturnOp = returnOp->clone(mapper);
    newBlock->push_back(newReturnOp);
  }
  
  // Replace old block
  // 1. Add new block to function body
  funcOp.getBody().push_back(newBlock);
  
  // 2. Update usage relationships
  for (auto &op : oldBlock->getOperations()) {
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      Value oldResult = op.getResult(i);
      if (mapper.contains(oldResult)) {
        oldResult.replaceAllUsesWith(mapper.lookup(oldResult));
      }
    }
  }
  
  // 3. Remove old block
  oldBlock->dropAllUses();
  oldBlock->erase();
}

// Fixed version of reorganizeGPUModules function
// void reorganizeGPUModules(ModuleOp moduleOp, DependencyGraph &graph) {
//   OpBuilder builder(moduleOp.getContext());
  
//   // Group modules by topological level
//   std::map<unsigned, llvm::SmallVector<StringRef, 8>> modulesByLevel;
//   std::set<std::pair<unsigned, StringRef>> processedModules;
  
//   for (const auto &nodePair : graph.nodes) {
//     DependencyNode* node = nodePair.get();
//     if (node->type == NodeType::Kernel) {
//       auto level = node->topologicalLevel;
//       auto moduleName = node->kernelModuleName;
      
//       // Add each module only once per level
//       auto key = std::make_pair(level, moduleName);
//       if (processedModules.insert(key).second) {
//         modulesByLevel[level].push_back(moduleName);
//       }
//     }
//   }
  
//   // Record original module and function names for later updates
//   struct ModuleInfo {
//     llvm::SmallVector<Operation*, 4> toRemove;
//     llvm::SmallVector<std::pair<gpu::GPUFuncOp, std::string>, 4> funcRenameMap;
//   };
  
//   // Create combined modules for each level
//   for (const auto &levelPair : modulesByLevel) {
//     unsigned level = levelPair.first;
//     const auto &modules = levelPair.second;
    
//     // Skip if this level has only one module
//     if (modules.size() <= 1)
//       continue;
      
//     // Create new combined module
//     std::string combinedName = "main_graph_kernel_level_" + std::to_string(level);
//     builder.setInsertionPointToStart(moduleOp.getBody());
    
//     auto combinedModule = builder.create<gpu::GPUModuleOp>(
//         moduleOp.getLoc(), 
//         builder.getStringAttr(combinedName));
    
//     // Add kernel functions from all modules at this level to the combined module
//     builder.setInsertionPointToStart(combinedModule.getBody());
    
//     ModuleInfo info;
//     int funcCounter = 0;  // Simple counter
    
//     // Phase 1: Clone all functions and collect renaming information
//     for (auto moduleName : modules) {
//       bool found = false;
      
//       // Find the module
//       moduleOp.walk([&](gpu::GPUModuleOp op) {
//         if (op.getName() == moduleName) {
//           found = true;
          
//           // Walk through all functions in the module
//           for (Operation &op : op.getBody()->getOperations()) {
//             if (auto funcOp = dyn_cast<gpu::GPUFuncOp>(op)) {
//               // Create new name for the function
//               std::string newFuncName = "func_" + std::to_string(funcCounter++);
              
//               // Clone and rename function
//               auto clonedFunc = cast<gpu::GPUFuncOp>(builder.clone(op));
//               clonedFunc.setName(newFuncName);
              
//               // Record renaming information
//               info.funcRenameMap.push_back({cast<gpu::GPUFuncOp>(op), newFuncName});
//             }
//           }
          
//           // Mark module for deletion
//           info.toRemove.push_back(op);
//         }
//       });
//     }
    
//     // Phase 2: Update all kernel launches
//     moduleOp.walk([&](gpu::LaunchFuncOp op) {
//       StringRef opModuleName = op.getKernelModuleName();
//       StringRef opKernelName = op.getKernelName();
      
//       // Check if this launch uses any module at the current level
//       for (auto moduleName : modules) {
//         if (opModuleName == moduleName) {
//           // Find the corresponding renaming information
//           for (auto &pair : info.funcRenameMap) {
//             gpu::GPUFuncOp origFunc = pair.first;
//             std::string newFuncName = pair.second;
            
//             if (origFunc.getName() == opKernelName) {
//               // Create new symbol reference
//               auto newKernel = SymbolRefAttr::get(
//                   builder.getContext(),
//                   StringAttr::get(builder.getContext(), combinedName),
//                   {SymbolRefAttr::get(builder.getContext(), newFuncName)});
              
//               // Update kernel attribute
//               op->setAttr("kernel", newKernel);
//               break;
//             }
//           }
//           break;
//         }
//       }
//     });
    
//     // Phase 3: Delete original modules
//     for (Operation *op : info.toRemove) {
//       op->erase();
//     }
//   }
// }

void reorganizeGPUModules(ModuleOp moduleOp, DependencyGraph &graph) {
  OpBuilder builder(moduleOp.getContext());
  
  // Direct counter - ensure unique names are generated
  int moduleCounter = 0;
  int funcCounter = 0;
  
  // Scan once and get all modules
  llvm::SmallVector<gpu::GPUModuleOp, 4> allModules;
  moduleOp.walk([&](gpu::GPUModuleOp op) {
    allModules.push_back(op);
  });
  
  // Exit if no modules
  if (allModules.empty())
    return;
  
  // Create a new merged module
  std::string combinedName = "merged_module_" + std::to_string(moduleCounter++);
  builder.setInsertionPointToStart(moduleOp.getBody());
  
  auto combinedModule = builder.create<gpu::GPUModuleOp>(
      moduleOp.getLoc(),
      builder.getStringAttr(combinedName));
  
  builder.setInsertionPointToStart(combinedModule.getBody());
  
  // Create mapping: <old module name, old function name> -> new function name
  std::map<std::pair<std::string, std::string>, std::string> renameMap;
  
  // Step 1: Copy all functions and rename them
  for (auto moduleOp : allModules) {
    std::string oldModuleName = moduleOp.getName().str();
    
    for (Operation &op : moduleOp.getBody()->getOperations()) {
      if (auto funcOp = dyn_cast<gpu::GPUFuncOp>(op)) {
        std::string oldFuncName = funcOp.getName().str();
        
        // Create new function name
        std::string newFuncName = "kernel_" + std::to_string(funcCounter++);
        
        // Clone and rename function
        auto clonedFunc = cast<gpu::GPUFuncOp>(builder.clone(op));
        clonedFunc.setName(newFuncName);
        
        // Save renaming mapping
        renameMap[{oldModuleName, oldFuncName}] = newFuncName;
      }
    }
  }
  
  // Step 2: Update all kernel launch references
  moduleOp.walk([&](gpu::LaunchFuncOp op) {
    std::string oldModuleName = op.getKernelModuleName().str();
    std::string oldFuncName = op.getKernelName().str();
    
    auto it = renameMap.find({oldModuleName, oldFuncName});
    if (it != renameMap.end()) {
      std::string newFuncName = it->second;
      
      // Create new symbol reference
      auto newKernel = SymbolRefAttr::get(
          builder.getContext(),
          StringAttr::get(builder.getContext(), combinedName),
          {SymbolRefAttr::get(builder.getContext(), newFuncName)});
      
      // Update attribute
      op->setAttr("kernel", newKernel);
    }
  });
  
  // Step 3: Delete old modules
  for (auto moduleOp : allModules) {
    moduleOp.erase();
  }
}

} // namespace onnx_mlir