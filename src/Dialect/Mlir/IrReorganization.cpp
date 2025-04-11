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

// Single_gpu_module version
// void reorganizeGPUModules(ModuleOp moduleOp, DependencyGraph &graph) {
//   OpBuilder builder(moduleOp.getContext());
  
//   // Direct counter - ensure unique names are generated
//   int moduleCounter = 0;
//   int funcCounter = 0;
  
//   // Scan once and get all modules
//   llvm::SmallVector<gpu::GPUModuleOp, 4> allModules;
//   moduleOp.walk([&](gpu::GPUModuleOp op) {
//     allModules.push_back(op);
//   });
  
//   // Exit if no modules
//   if (allModules.empty())
//     return;
  
//   // Create a new merged module
//   std::string combinedName = "merged_module_" + std::to_string(moduleCounter++);
//   builder.setInsertionPointToStart(moduleOp.getBody());
  
//   auto combinedModule = builder.create<gpu::GPUModuleOp>(
//       moduleOp.getLoc(),
//       builder.getStringAttr(combinedName));
  
//   builder.setInsertionPointToStart(combinedModule.getBody());
  
//   // Create mapping: <old module name, old function name> -> new function name
//   std::map<std::pair<std::string, std::string>, std::string> renameMap;
  
//   // Step 1: Copy all functions and rename them
//   for (auto moduleOp : allModules) {
//     std::string oldModuleName = moduleOp.getName().str();
    
//     for (Operation &op : moduleOp.getBody()->getOperations()) {
//       if (auto funcOp = dyn_cast<gpu::GPUFuncOp>(op)) {
//         std::string oldFuncName = funcOp.getName().str();
        
//         // Create new function name
//         std::string newFuncName = "kernel_" + std::to_string(funcCounter++);
        
//         // Clone and rename function
//         auto clonedFunc = cast<gpu::GPUFuncOp>(builder.clone(op));
//         clonedFunc.setName(newFuncName);
        
//         // Save renaming mapping
//         renameMap[{oldModuleName, oldFuncName}] = newFuncName;
//       }
//     }
//   }
  
//   // Step 2: Update all kernel launch references
//   moduleOp.walk([&](gpu::LaunchFuncOp op) {
//     std::string oldModuleName = op.getKernelModuleName().str();
//     std::string oldFuncName = op.getKernelName().str();
    
//     auto it = renameMap.find({oldModuleName, oldFuncName});
//     if (it != renameMap.end()) {
//       std::string newFuncName = it->second;
      
//       // Create new symbol reference
//       auto newKernel = SymbolRefAttr::get(
//           builder.getContext(),
//           StringAttr::get(builder.getContext(), combinedName),
//           {SymbolRefAttr::get(builder.getContext(), newFuncName)});
      
//       // Update attribute
//       op->setAttr("kernel", newKernel);
//     }
//   });
  
//   // Step 3: Delete old modules
//   for (auto moduleOp : allModules) {
//     moduleOp.erase();
//   }
// }

// multi_gpu_module version
void reorganizeGPUModules(ModuleOp moduleOp, DependencyGraph &graph) {
  OpBuilder builder(moduleOp.getContext());
  
  // Scan all existing modules and functions
  llvm::SmallVector<gpu::GPUModuleOp, 4> allModules;
  std::map<std::pair<std::string, std::string>, gpu::GPUFuncOp> funcMap;
  
  moduleOp.walk([&](gpu::GPUModuleOp op) {
    allModules.push_back(op);
    
    // Collect all functions in this module
    std::string moduleName = op.getName().str();
    op.walk([&](gpu::GPUFuncOp funcOp) {
      std::string funcName = funcOp.getName().str();
      funcMap[{moduleName, funcName}] = funcOp;
    });
  });
  
  // Exit if no modules
  if (allModules.empty())
    return;
  
  // Group kernel nodes by topological level
  std::map<unsigned, llvm::SmallVector<DependencyNode*, 8>> kernelsByLevel;
  for (const auto &nodePair : graph.nodes) {
    DependencyNode* node = nodePair.get();
    if (node->type == NodeType::Kernel) {
      kernelsByLevel[node->topologicalLevel].push_back(node);
    }
  }
  
  // Create renaming map: <old module name, old function name> -> <new module name, new function name>
  using ModuleFuncKey = std::pair<std::string, std::string>;
  std::map<ModuleFuncKey, ModuleFuncKey> renameMap;
  
  // Create a counter for each topological level to ensure function name uniqueness
  std::map<unsigned, int> levelFuncCounter;
  
  // Step 1: Create a module for each topological level and copy corresponding functions
  for (const auto &levelPair : kernelsByLevel) {
    unsigned level = levelPair.first;
    const auto &kernels = levelPair.second;
    
    // Skip empty levels
    if (kernels.empty())
      continue;
    
    // Initialize function counter for this level
    levelFuncCounter[level] = 0;
    
    // Create a new module for this level
    std::string newModuleName = "level_" + std::to_string(level) + "_module";
    builder.setInsertionPointToStart(moduleOp.getBody());
    
    auto levelModule = builder.create<gpu::GPUModuleOp>(
        moduleOp.getLoc(),
        builder.getStringAttr(newModuleName));
    
    builder.setInsertionPointToStart(levelModule.getBody());
    
    // Copy all kernel functions for this level
    for (DependencyNode* kernel : kernels) {
      std::string oldModuleName = kernel->kernelModuleName.str();
      std::string oldFuncName = kernel->kernelName.str();
      
      // Find the original function
      auto funcKey = std::make_pair(oldModuleName, oldFuncName);
      auto funcIt = funcMap.find(funcKey);
      
      if (funcIt != funcMap.end()) {
        // Create a new unique function name
        std::string newFuncName = "kernel_" + std::to_string(level) + "_" + 
                                  std::to_string(levelFuncCounter[level]++);
        
        // Clone the function to the new module
        auto clonedFunc = cast<gpu::GPUFuncOp>(builder.clone(*funcIt->second));
        
        // Set the new function name
        clonedFunc.setName(newFuncName);
        
        // Save mapping relationship
        renameMap[funcKey] = {newModuleName, newFuncName};
      }
    }
  }
  
  // Step 2: Update all kernel launch references
  moduleOp.walk([&](gpu::LaunchFuncOp op) {
    std::string oldModuleName = op.getKernelModuleName().str();
    std::string oldFuncName = op.getKernelName().str();
    
    auto funcKey = std::make_pair(oldModuleName, oldFuncName);
    auto renameIt = renameMap.find(funcKey);
    
    if (renameIt != renameMap.end()) {
      std::string newModuleName = renameIt->second.first;
      std::string newFuncName = renameIt->second.second;
      
      // Create new symbol reference
      auto newKernel = SymbolRefAttr::get(
          builder.getContext(),
          StringAttr::get(builder.getContext(), newModuleName),
          {SymbolRefAttr::get(builder.getContext(), newFuncName)});
      
      // Update kernel reference
      op->setAttr("kernel", newKernel);
    }
  });
  
  // Step 3: Delete old modules
  for (auto moduleOp : allModules) {
    moduleOp.erase();
  }
}

} // namespace onnx_mlir