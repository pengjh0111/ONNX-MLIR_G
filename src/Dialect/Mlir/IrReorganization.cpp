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

// Explicit Barrier Synchronization version
// void reorganizeIR(func::FuncOp funcOp, DependencyGraph &graph) {
//   OpBuilder builder(funcOp.getContext());
  
//   // Create mapping to track operation mapping relationships
//   IRMapping mapper;
  
//   // Group nodes by topological level
//   std::map<unsigned, llvm::SmallVector<DependencyNode*, 8>> nodesByLevel;
//   for (const auto &nodePair : graph.nodes) {
//     DependencyNode* node = nodePair.get();
//     nodesByLevel[node->topologicalLevel].push_back(node);
//   }
  
//   // Create new block
//   Block* oldBlock = &funcOp.getBody().front();
//   Block* newBlock = new Block();
  
//   // Map arguments
//   for (auto &blockArg : oldBlock->getArguments()) {
//     auto newArg = newBlock->addArgument(blockArg.getType(), blockArg.getLoc());
//     mapper.map(blockArg, newArg);
//   }
  
//   // Track processed operations
//   llvm::DenseSet<Operation*> processedOps;
  
//   // Collect all alloca operations, which need to be placed before use
//   llvm::SmallVector<Operation*, 16> allocaOps;
//   funcOp.walk([&](memref::AllocaOp allocaOp) {
//     allocaOps.push_back(allocaOp);
//   });
  
//   // Phase 1: First copy non-graph node prefix operations, while handling all allocas
//   for (auto &op : oldBlock->getOperations()) {
//     if (graph.opToNodeMap.count(&op)) {
//       // Stop copying prefix operations when encountering a node in the graph
//       break;
//     }
    
//     // Skip all GPU wait operations, as we will add necessary wait points according to the dependency graph
//     if (isa<gpu::WaitOp>(op)) {
//       processedOps.insert(&op);
//       continue;
//     }
    
//     // Record all alloca operations, to be processed together later
//     if (isa<memref::AllocaOp>(op)) {
//       processedOps.insert(&op);
//       continue; // Skip for now, process later
//     }
    
//     Operation *newOp = op.clone(mapper);
//     newBlock->push_back(newOp);
    
//     // Update mapping and mark as processed
//     for (unsigned i = 0; i < op.getNumResults(); ++i) {
//       mapper.map(op.getResult(i), newOp->getResult(i));
//     }
//     processedOps.insert(&op);
//   }
  
//   // Find maximum topological level
//   unsigned maxLevel = 0;
//   for (const auto &nodePair : graph.nodes) {
//     maxLevel = std::max(maxLevel, nodePair.get()->topologicalLevel);
//   }
  
//   // For tracking tokens from the final level
//   llvm::SmallVector<Value, 8> finalLevelTokens;
  
//   // Phase 2: Process nodes by topological level
//   for (unsigned level = 1; level <= maxLevel; level++) {
//     auto levelIt = nodesByLevel.find(level);
//     if (levelIt == nodesByLevel.end() || levelIt->second.empty())
//       continue;
      
//     auto &nodesAtLevel = levelIt->second;
    
//     // Collect async tokens for this level
//     llvm::SmallVector<Value, 8> levelTokens;
    
//     // Count kernels at current level
//     unsigned kernelCount = 0;
//     for (auto node : nodesAtLevel) {
//       if (node->type == NodeType::Kernel) {
//         kernelCount++;
//       }
//     }
    
//     // Step 1: Create async wait tokens for all kernels
//     llvm::SmallVector<Value, 8> waitTokens;
//     if (kernelCount > 0) {
//       builder.setInsertionPointToEnd(newBlock);
      
//       for (unsigned i = 0; i < kernelCount; i++) {
//         // Create async wait operation
//         auto waitOp = builder.create<gpu::WaitOp>(
//             funcOp.getLoc(),
//             builder.getType<gpu::AsyncTokenType>(),
//             ValueRange{});
//         waitTokens.push_back(waitOp.getAsyncToken());
//       }
//     }
    
//     // Step 2: Process all nodes at the current level
//     unsigned kernelIndex = 0;  // Used to track which kernel is currently being processed
    
//     for (auto node : nodesAtLevel) {
//       builder.setInsertionPointToEnd(newBlock);
      
//       if (node->type == NodeType::Kernel) {
//         auto kernelOp = cast<gpu::LaunchFuncOp>(node->op);
        
//         // Create kernel symbol reference
//         auto kernelSymbol = SymbolRefAttr::get(
//             builder.getContext(),
//             kernelOp.getKernelModuleName(),
//             {SymbolRefAttr::get(builder.getContext(), kernelOp.getKernelName())});
        
//         // Map operands
//         SmallVector<Value, 8> remappedOperands;
//         for (Value operand : kernelOp.getKernelOperands()) {
//           remappedOperands.push_back(mapper.lookupOrDefault(operand));
//         }
        
//         // Map grid and block sizes
//         auto gridSize = kernelOp.getGridSizeOperandValues();
//         auto blockSize = kernelOp.getBlockSizeOperandValues();
        
//         mlir::gpu::KernelDim3 mappedGridSize = {
//           mapper.lookupOrDefault(gridSize.x),
//           mapper.lookupOrDefault(gridSize.y),
//           mapper.lookupOrDefault(gridSize.z)
//         };
        
//         mlir::gpu::KernelDim3 mappedBlockSize = {
//           mapper.lookupOrDefault(blockSize.x),
//           mapper.lookupOrDefault(blockSize.y),
//           mapper.lookupOrDefault(blockSize.z)
//         };
        
//         // Use previously created async wait token
//         Value waitToken = waitTokens[kernelIndex++];
            
//         // Directly create async kernel launch
//         auto newLaunchOp = builder.create<gpu::LaunchFuncOp>(
//             kernelOp.getLoc(),
//             kernelSymbol,
//             mappedGridSize,
//             mappedBlockSize,
//             Value(),  // No dynamic shared memory
//             remappedOperands,
//             builder.getType<gpu::AsyncTokenType>(),  // Async token type
//             ValueRange{waitToken},  // Use previously created async wait token
//             std::nullopt);  // No cluster size
            
//         // Collect async tokens for this level
//         levelTokens.push_back(newLaunchOp.getAsyncToken());
            
//         // Map results
//         if (kernelOp->getNumResults() > 0) {
//           mapper.map(kernelOp->getResult(0), newLaunchOp->getResult(0));
//         }
        
//         // Mark as processed
//         processedOps.insert(node->op);
//       } 
//       else if (node->type == NodeType::Loop) {
//         // Find all memref.alloca operations associated with this loop
//         llvm::SmallVector<Operation*, 8> loopLocalAllocas;
//         for (auto allocaOp : allocaOps) {
//           // Check if this alloca is used by this loop
//           bool used = false;
//           Value allocaResult = allocaOp->getResult(0);
//           node->op->walk([&](Operation *user) {
//             for (Value operand : user->getOperands()) {
//               if (operand == allocaResult) {
//                 used = true;
//                 return WalkResult::interrupt();
//               }
//             }
//             return WalkResult::advance();
//           });
          
//           if (used) {
//             loopLocalAllocas.push_back(allocaOp);
//             processedOps.insert(allocaOp); // Mark as processed
//           }
//         }
        
//         // Recreate all local allocas before this loop
//         for (auto allocaOp : loopLocalAllocas) {
//           auto newAllocaOp = builder.clone(*allocaOp, mapper);
          
//           // Update mapping
//           for (unsigned i = 0; i < allocaOp->getNumResults(); ++i) {
//             mapper.map(allocaOp->getResult(i), newAllocaOp->getResult(i));
//           }
//         }
        
//         // Clone loop operation
//         Operation *newOp = builder.clone(*node->op, mapper);
        
//         // Update mapping
//         for (unsigned i = 0; i < node->op->getNumResults(); ++i) {
//           mapper.map(node->op->getResult(i), newOp->getResult(i));
//         }
        
//         // Mark as processed
//         processedOps.insert(node->op);
//       }
//     }
    
//     // If current level has operations
//     if (!levelTokens.empty()) {
//       // If not the last level, add a synchronization point
//       if (level < maxLevel) {
//         builder.setInsertionPointToEnd(newBlock);
        
//         // Non-async wait - ensure all operations at this level complete before moving to the next level
//         builder.create<gpu::WaitOp>(funcOp.getLoc(), TypeRange{}, levelTokens);
//       } 
//       // If it's the last level, save tokens to add a final sync point before function return
//       else {
//         finalLevelTokens = levelTokens;
//       }
//     }
//   }
  
//   // Process remaining unused alloca operations
//   for (auto allocaOp : allocaOps) {
//     if (!processedOps.count(allocaOp)) {
//       builder.setInsertionPointToEnd(newBlock);
//       auto newAllocaOp = builder.clone(*allocaOp, mapper);
      
//       // Update mapping
//       for (unsigned i = 0; i < allocaOp->getNumResults(); ++i) {
//         mapper.map(allocaOp->getResult(i), newAllocaOp->getResult(i));
//       }
      
//       processedOps.insert(allocaOp);
//     }
//   }
  
//   // Phase 3: Copy remaining unprocessed operations, but skip gpu.wait operations
//   bool hasReturnOp = false;
//   Operation* returnOp = nullptr;
  
//   for (auto &op : oldBlock->getOperations()) {
//     if (processedOps.count(&op))
//       continue;  // Skip already processed operations
      
//     // Skip all GPU wait operations
//     if (isa<gpu::WaitOp>(op)) {
//       processedOps.insert(&op);
//       continue;
//     }
    
//     // If it's a return operation, don't clone it yet, process it later
//     if (isa<func::ReturnOp>(op)) {
//       hasReturnOp = true;
//       returnOp = &op;
//       continue;
//     }
    
//     Operation *newOp = op.clone(mapper);
//     newBlock->push_back(newOp);
    
//     // Update mapping
//     for (unsigned i = 0; i < op.getNumResults(); ++i) {
//       mapper.map(op.getResult(i), newOp->getResult(i));
//     }
//   }
  
//   // If there are tokens from the final level, add a final sync point
//   if (!finalLevelTokens.empty()) {
//     builder.setInsertionPointToEnd(newBlock);
    
//     // Add final synchronization wait
//     builder.create<gpu::WaitOp>(funcOp.getLoc(), TypeRange{}, finalLevelTokens);
//   }
  
//   // If there's a return operation, clone it now
//   if (hasReturnOp) {
//     builder.setInsertionPointToEnd(newBlock);
//     Operation *newReturnOp = returnOp->clone(mapper);
//     newBlock->push_back(newReturnOp);
//   }
  
//   // Replace old block
//   // 1. Add new block to function body
//   funcOp.getBody().push_back(newBlock);
  
//   // 2. Update usage relationships
//   for (auto &op : oldBlock->getOperations()) {
//     for (unsigned i = 0; i < op.getNumResults(); ++i) {
//       Value oldResult = op.getResult(i);
//       if (mapper.contains(oldResult)) {
//         oldResult.replaceAllUsesWith(mapper.lookup(oldResult));
//       }
//     }
//   }
  
//   // 3. Remove old block
//   oldBlock->dropAllUses();
//   oldBlock->erase();
// }

// // Token-Chained Synchronization version
// void reorganizeIR(func::FuncOp funcOp, DependencyGraph &graph) {
//   OpBuilder builder(funcOp.getContext());
  
//   // 创建映射以跟踪操作映射关系
//   IRMapping mapper;
  
//   // 按拓扑级别分组节点
//   std::map<unsigned, llvm::SmallVector<DependencyNode*, 8>> nodesByLevel;
//   for (const auto &nodePair : graph.nodes) {
//     DependencyNode* node = nodePair.get();
//     nodesByLevel[node->topologicalLevel].push_back(node);
//   }
  
//   // 创建新块
//   Block* oldBlock = &funcOp.getBody().front();
//   Block* newBlock = new Block();
  
//   // 映射参数
//   for (auto &blockArg : oldBlock->getArguments()) {
//     auto newArg = newBlock->addArgument(blockArg.getType(), blockArg.getLoc());
//     mapper.map(blockArg, newArg);
//   }
  
//   // 跟踪已处理的操作
//   llvm::DenseSet<Operation*> processedOps;
  
//   // 收集所有alloca操作，需要在使用前放置
//   llvm::SmallVector<Operation*, 16> allocaOps;
//   funcOp.walk([&](memref::AllocaOp allocaOp) {
//     allocaOps.push_back(allocaOp);
//   });
  
//   // 阶段1：首先复制非图节点前缀操作，同时处理所有alloca
//   for (auto &op : oldBlock->getOperations()) {
//     if (graph.opToNodeMap.count(&op)) {
//       // 当遇到图中的节点时停止复制前缀操作
//       break;
//     }
    
//     // 跳过所有GPU等待操作，我们将根据依赖图添加必要的等待点
//     if (isa<gpu::WaitOp>(op)) {
//       processedOps.insert(&op);
//       continue;
//     }
    
//     // 记录所有alloca操作，稍后一起处理
//     if (isa<memref::AllocaOp>(op)) {
//       processedOps.insert(&op);
//       continue; // 暂时跳过，稍后处理
//     }
    
//     Operation *newOp = op.clone(mapper);
//     newBlock->push_back(newOp);
    
//     // 更新映射并标记为已处理
//     for (unsigned i = 0; i < op.getNumResults(); ++i) {
//       mapper.map(op.getResult(i), newOp->getResult(i));
//     }
//     processedOps.insert(&op);
//   }
  
//   // 找到最大拓扑级别
//   unsigned maxLevel = 0;
//   for (const auto &nodePair : graph.nodes) {
//     maxLevel = std::max(maxLevel, nodePair.get()->topologicalLevel);
//   }
  
//   // 用于跟踪前一级别的令牌
//   llvm::SmallVector<Value, 8> previousLevelTokens;
  
//   // 阶段2：按拓扑级别处理节点
//   for (unsigned level = 1; level <= maxLevel; level++) {
//     auto levelIt = nodesByLevel.find(level);
//     if (levelIt == nodesByLevel.end() || levelIt->second.empty())
//       continue;
      
//     auto &nodesAtLevel = levelIt->second;
    
//     // 收集此级别的异步令牌
//     llvm::SmallVector<Value, 8> levelTokens;
    
//     // 计算当前级别的内核数量
//     unsigned kernelCount = 0;
//     for (auto node : nodesAtLevel) {
//       if (node->type == NodeType::Kernel) {
//         kernelCount++;
//       }
//     }
    
//     // 如果没有内核，只需正常处理节点
//     if (kernelCount == 0) {
//       for (auto node : nodesAtLevel) {
//         builder.setInsertionPointToEnd(newBlock);
        
//         // 这将是一个循环节点
//         // 查找与此循环相关的所有memref.alloca操作
//         llvm::SmallVector<Operation*, 8> loopLocalAllocas;
//         for (auto allocaOp : allocaOps) {
//           // 检查此alloca是否被此循环使用
//           bool used = false;
//           Value allocaResult = allocaOp->getResult(0);
//           node->op->walk([&](Operation *user) {
//             for (Value operand : user->getOperands()) {
//               if (operand == allocaResult) {
//                 used = true;
//                 return WalkResult::interrupt();
//               }
//             }
//             return WalkResult::advance();
//           });
          
//           if (used) {
//             loopLocalAllocas.push_back(allocaOp);
//             processedOps.insert(allocaOp); // 标记为已处理
//           }
//         }
        
//         // 在此循环之前重新创建所有本地alloca
//         for (auto allocaOp : loopLocalAllocas) {
//           auto newAllocaOp = builder.clone(*allocaOp, mapper);
          
//           // 更新映射
//           for (unsigned i = 0; i < allocaOp->getNumResults(); ++i) {
//             mapper.map(allocaOp->getResult(i), newAllocaOp->getResult(i));
//           }
//         }
        
//         // 克隆循环操作
//         Operation *newOp = builder.clone(*node->op, mapper);
        
//         // 更新映射
//         for (unsigned i = 0; i < node->op->getNumResults(); ++i) {
//           mapper.map(node->op->getResult(i), newOp->getResult(i));
//         }
        
//         // 标记为已处理
//         processedOps.insert(node->op);
//       }
//       continue;
//     }
    
//     // 为此级别创建等待令牌
//     llvm::SmallVector<Value, 8> waitTokens;
    
//     // 第一级别或之前没有处理级别时，为每个内核创建独立的wait令牌
//     if (level == 1 || previousLevelTokens.empty()) {
//       builder.setInsertionPointToEnd(newBlock);
      
//       // 为每个内核创建独立的等待令牌
//       for (unsigned i = 0; i < kernelCount; i++) {
//         auto waitOp = builder.create<gpu::WaitOp>(
//             funcOp.getLoc(),
//             builder.getType<gpu::AsyncTokenType>(),
//             ValueRange{});
//         waitTokens.push_back(waitOp.getAsyncToken());
//       }
//     } 
//     // 对于后续级别，创建一个依赖于所有前一级别令牌的单个等待令牌
//     else {
//       builder.setInsertionPointToEnd(newBlock);
      
//       // 创建一个依赖于前一级别所有令牌的单个等待操作
//       auto waitOp = builder.create<gpu::WaitOp>(
//           funcOp.getLoc(),
//           builder.getType<gpu::AsyncTokenType>(),
//           previousLevelTokens);
      
//       // 对此级别的所有内核使用此单个令牌
//       for (unsigned i = 0; i < kernelCount; i++) {
//         waitTokens.push_back(waitOp.getAsyncToken());
//       }
//     }
    
//     // 处理此级别的所有节点
//     unsigned kernelIndex = 0;
//     for (auto node : nodesAtLevel) {
//       builder.setInsertionPointToEnd(newBlock);
      
//       if (node->type == NodeType::Kernel) {
//         auto kernelOp = cast<gpu::LaunchFuncOp>(node->op);
        
//         // 创建内核符号引用
//         auto kernelSymbol = SymbolRefAttr::get(
//             builder.getContext(),
//             kernelOp.getKernelModuleName(),
//             {SymbolRefAttr::get(builder.getContext(), kernelOp.getKernelName())});
        
//         // 映射操作数
//         SmallVector<Value, 8> remappedOperands;
//         for (Value operand : kernelOp.getKernelOperands()) {
//           remappedOperands.push_back(mapper.lookupOrDefault(operand));
//         }
        
//         // 映射网格和块大小
//         auto gridSize = kernelOp.getGridSizeOperandValues();
//         auto blockSize = kernelOp.getBlockSizeOperandValues();
        
//         mlir::gpu::KernelDim3 mappedGridSize = {
//           mapper.lookupOrDefault(gridSize.x),
//           mapper.lookupOrDefault(gridSize.y),
//           mapper.lookupOrDefault(gridSize.z)
//         };
        
//         mlir::gpu::KernelDim3 mappedBlockSize = {
//           mapper.lookupOrDefault(blockSize.x),
//           mapper.lookupOrDefault(blockSize.y),
//           mapper.lookupOrDefault(blockSize.z)
//         };
        
//         // 使用此内核的等待令牌
//         Value waitToken = waitTokens[kernelIndex++];
            
//         // 直接创建异步内核启动
//         auto newLaunchOp = builder.create<gpu::LaunchFuncOp>(
//             kernelOp.getLoc(),
//             kernelSymbol,
//             mappedGridSize,
//             mappedBlockSize,
//             Value(),  // 无动态共享内存
//             remappedOperands,
//             builder.getType<gpu::AsyncTokenType>(),  // 异步令牌类型
//             ValueRange{waitToken},  // 使用等待令牌
//             std::nullopt);  // 无集群大小
            
//         // 收集此级别的异步令牌
//         levelTokens.push_back(newLaunchOp.getAsyncToken());
            
//         // 映射结果
//         if (kernelOp->getNumResults() > 0) {
//           mapper.map(kernelOp->getResult(0), newLaunchOp->getResult(0));
//         }
        
//         // 标记为已处理
//         processedOps.insert(node->op);
//       } 
//       else if (node->type == NodeType::Loop) {
//         // 处理循环节点
//         // 查找与此循环相关的所有memref.alloca操作
//         llvm::SmallVector<Operation*, 8> loopLocalAllocas;
//         for (auto allocaOp : allocaOps) {
//           // 检查此alloca是否被此循环使用
//           bool used = false;
//           Value allocaResult = allocaOp->getResult(0);
//           node->op->walk([&](Operation *user) {
//             for (Value operand : user->getOperands()) {
//               if (operand == allocaResult) {
//                 used = true;
//                 return WalkResult::interrupt();
//               }
//             }
//             return WalkResult::advance();
//           });
          
//           if (used) {
//             loopLocalAllocas.push_back(allocaOp);
//             processedOps.insert(allocaOp); // 标记为已处理
//           }
//         }
        
//         // 在此循环之前重新创建所有本地alloca
//         for (auto allocaOp : loopLocalAllocas) {
//           auto newAllocaOp = builder.clone(*allocaOp, mapper);
          
//           // 更新映射
//           for (unsigned i = 0; i < allocaOp->getNumResults(); ++i) {
//             mapper.map(allocaOp->getResult(i), newAllocaOp->getResult(i));
//           }
//         }
        
//         // 克隆循环操作
//         Operation *newOp = builder.clone(*node->op, mapper);
        
//         // 更新映射
//         for (unsigned i = 0; i < node->op->getNumResults(); ++i) {
//           mapper.map(node->op->getResult(i), newOp->getResult(i));
//         }
        
//         // 标记为已处理
//         processedOps.insert(node->op);
//       }
//     }
    
//     // 更新前一级别令牌用于下一级别
//     previousLevelTokens = levelTokens;
//   }
  
//   // 处理剩余未使用的alloca操作
//   for (auto allocaOp : allocaOps) {
//     if (!processedOps.count(allocaOp)) {
//       builder.setInsertionPointToEnd(newBlock);
//       auto newAllocaOp = builder.clone(*allocaOp, mapper);
      
//       // 更新映射
//       for (unsigned i = 0; i < allocaOp->getNumResults(); ++i) {
//         mapper.map(allocaOp->getResult(i), newAllocaOp->getResult(i));
//       }
      
//       processedOps.insert(allocaOp);
//     }
//   }
  
//   // 阶段3：复制剩余未处理的操作，但跳过gpu.wait操作
//   bool hasReturnOp = false;
//   Operation* returnOp = nullptr;
  
//   for (auto &op : oldBlock->getOperations()) {
//     if (processedOps.count(&op))
//       continue;  // 跳过已处理的操作
      
//     // 跳过所有GPU等待操作
//     if (isa<gpu::WaitOp>(op)) {
//       processedOps.insert(&op);
//       continue;
//     }
    
//     // 如果是返回操作，先不克隆，稍后处理
//     if (isa<func::ReturnOp>(op)) {
//       hasReturnOp = true;
//       returnOp = &op;
//       continue;
//     }
    
//     Operation *newOp = op.clone(mapper);
//     newBlock->push_back(newOp);
    
//     // 更新映射
//     for (unsigned i = 0; i < op.getNumResults(); ++i) {
//       mapper.map(op.getResult(i), newOp->getResult(i));
//     }
//   }
  
//   // 如果有来自最终级别的令牌，添加最终同步点
//   if (!previousLevelTokens.empty()) {
//     builder.setInsertionPointToEnd(newBlock);
    
//     // 添加最终同步等待
//     builder.create<gpu::WaitOp>(funcOp.getLoc(), TypeRange{}, previousLevelTokens);
//   }
  
//   // 如果有返回操作，现在克隆它
//   if (hasReturnOp) {
//     builder.setInsertionPointToEnd(newBlock);
//     Operation *newReturnOp = returnOp->clone(mapper);
//     newBlock->push_back(newReturnOp);
//   }
  
//   // 替换旧块
//   // 1. 将新块添加到函数体
//   funcOp.getBody().push_back(newBlock);
  
//   // 2. 更新使用关系
//   for (auto &op : oldBlock->getOperations()) {
//     for (unsigned i = 0; i < op.getNumResults(); ++i) {
//       Value oldResult = op.getResult(i);
//       if (mapper.contains(oldResult)) {
//         oldResult.replaceAllUsesWith(mapper.lookup(oldResult));
//       }
//     }
//   }
  
//   // 3. 删除旧块
//   oldBlock->dropAllUses();
//   oldBlock->erase();
// }

// 混合方案：单个kernel使用链式令牌，多个kernel使用显式同步后创建独立异步令牌
void reorganizeIR(func::FuncOp funcOp, DependencyGraph &graph) {
  OpBuilder builder(funcOp.getContext());
  
  // 创建映射以跟踪操作映射关系
  IRMapping mapper;
  
  // 按拓扑级别分组节点
  std::map<unsigned, llvm::SmallVector<DependencyNode*, 8>> nodesByLevel;
  for (const auto &nodePair : graph.nodes) {
    DependencyNode* node = nodePair.get();
    nodesByLevel[node->topologicalLevel].push_back(node);
  }
  
  // 创建新块
  Block* oldBlock = &funcOp.getBody().front();
  Block* newBlock = new Block();
  
  // 映射参数
  for (auto &blockArg : oldBlock->getArguments()) {
    auto newArg = newBlock->addArgument(blockArg.getType(), blockArg.getLoc());
    mapper.map(blockArg, newArg);
  }
  
  // 跟踪已处理的操作
  llvm::DenseSet<Operation*> processedOps;
  
  // 收集所有alloca操作，需要在使用前放置
  llvm::SmallVector<Operation*, 16> allocaOps;
  funcOp.walk([&](memref::AllocaOp allocaOp) {
    allocaOps.push_back(allocaOp);
  });
  
  // 阶段1：首先复制非图节点前缀操作，同时处理所有alloca
  for (auto &op : oldBlock->getOperations()) {
    if (graph.opToNodeMap.count(&op)) {
      // 当遇到图中的节点时停止复制前缀操作
      break;
    }
    
    // 跳过所有GPU等待操作，我们将根据依赖图添加必要的等待点
    if (isa<gpu::WaitOp>(op)) {
      processedOps.insert(&op);
      continue;
    }
    
    // 记录所有alloca操作，稍后一起处理
    if (isa<memref::AllocaOp>(op)) {
      processedOps.insert(&op);
      continue; // 暂时跳过，稍后处理
    }
    
    Operation *newOp = op.clone(mapper);
    newBlock->push_back(newOp);
    
    // 更新映射并标记为已处理
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      mapper.map(op.getResult(i), newOp->getResult(i));
    }
    processedOps.insert(&op);
  }
  
  // 找到最大拓扑级别
  unsigned maxLevel = 0;
  for (const auto &nodePair : graph.nodes) {
    maxLevel = std::max(maxLevel, nodePair.get()->topologicalLevel);
  }
  
  // 用于跟踪前一级别的令牌
  llvm::SmallVector<Value, 8> previousLevelTokens;
  // 用于跟踪最终级别的令牌
  llvm::SmallVector<Value, 8> finalLevelTokens;
  
  // 阶段2：按拓扑级别处理节点
  for (unsigned level = 1; level <= maxLevel; level++) {
    auto levelIt = nodesByLevel.find(level);
    if (levelIt == nodesByLevel.end() || levelIt->second.empty())
      continue;
      
    auto &nodesAtLevel = levelIt->second;
    
    // 收集此级别的异步令牌
    llvm::SmallVector<Value, 8> levelTokens;
    
    // 计算当前级别的内核数量
    unsigned kernelCount = 0;
    for (auto node : nodesAtLevel) {
      if (node->type == NodeType::Kernel) {
        kernelCount++;
      }
    }
    
    // 如果没有内核，只需正常处理节点（如循环）
    if (kernelCount == 0) {
      for (auto node : nodesAtLevel) {
        builder.setInsertionPointToEnd(newBlock);
        
        // 查找与此循环相关的所有memref.alloca操作
        llvm::SmallVector<Operation*, 8> loopLocalAllocas;
        for (auto allocaOp : allocaOps) {
          // 检查此alloca是否被此循环使用
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
            processedOps.insert(allocaOp); // 标记为已处理
          }
        }
        
        // 在此循环之前重新创建所有本地alloca
        for (auto allocaOp : loopLocalAllocas) {
          auto newAllocaOp = builder.clone(*allocaOp, mapper);
          
          // 更新映射
          for (unsigned i = 0; i < allocaOp->getNumResults(); ++i) {
            mapper.map(allocaOp->getResult(i), newAllocaOp->getResult(i));
          }
        }
        
        // 克隆循环操作
        Operation *newOp = builder.clone(*node->op, mapper);
        
        // 更新映射
        for (unsigned i = 0; i < node->op->getNumResults(); ++i) {
          mapper.map(node->op->getResult(i), newOp->getResult(i));
        }
        
        // 标记为已处理
        processedOps.insert(node->op);
      }
      continue;
    }
    
    // **混合方案的核心**
    // 对于只有单个内核的级别，使用链式令牌
    // 对于有多个内核的级别，使用显式同步后创建独立令牌
    if (kernelCount == 1) {
      // === 单内核链式令牌方案 ===
      builder.setInsertionPointToEnd(newBlock);
      
      // 创建一个依赖于前一级别令牌的等待操作
      Value waitToken;
      if (level == 1 || previousLevelTokens.empty()) {
        // 第一级别无需等待
        auto waitOp = builder.create<gpu::WaitOp>(
            funcOp.getLoc(),
            builder.getType<gpu::AsyncTokenType>(),
            ValueRange{});
        waitToken = waitOp.getAsyncToken();
      } else {
        // 依赖于前一级别的所有令牌
        auto waitOp = builder.create<gpu::WaitOp>(
            funcOp.getLoc(),
            builder.getType<gpu::AsyncTokenType>(),
            previousLevelTokens);
        waitToken = waitOp.getAsyncToken();
      }
      
      // 处理单个内核节点
      DependencyNode* node = nullptr;
      for (auto n : nodesAtLevel) {
        if (n->type == NodeType::Kernel) {
          node = n;
          break;
        }
      }
      
      if (node) {
        auto kernelOp = cast<gpu::LaunchFuncOp>(node->op);
        
        // 创建内核符号引用
        auto kernelSymbol = SymbolRefAttr::get(
            builder.getContext(),
            kernelOp.getKernelModuleName(),
            {SymbolRefAttr::get(builder.getContext(), kernelOp.getKernelName())});
        
        // 映射操作数
        SmallVector<Value, 8> remappedOperands;
        for (Value operand : kernelOp.getKernelOperands()) {
          remappedOperands.push_back(mapper.lookupOrDefault(operand));
        }
        
        // 映射网格和块大小
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
        
        // 使用链式等待令牌
        auto newLaunchOp = builder.create<gpu::LaunchFuncOp>(
            kernelOp.getLoc(),
            kernelSymbol,
            mappedGridSize,
            mappedBlockSize,
            Value(),  // 无动态共享内存
            remappedOperands,
            builder.getType<gpu::AsyncTokenType>(),  // 异步令牌类型
            ValueRange{waitToken},  // 使用链式等待令牌
            std::nullopt);  // 无集群大小
            
        // 收集此级别的异步令牌
        levelTokens.push_back(newLaunchOp.getAsyncToken());
            
        // 映射结果
        if (kernelOp->getNumResults() > 0) {
          mapper.map(kernelOp->getResult(0), newLaunchOp->getResult(0));
        }
        
        // 标记为已处理
        processedOps.insert(node->op);
      }
      
      // 处理此级别的任何循环节点
      for (auto node : nodesAtLevel) {
        if (node->type == NodeType::Loop) {
          builder.setInsertionPointToEnd(newBlock);
          
          // 查找与此循环相关的所有memref.alloca操作
          llvm::SmallVector<Operation*, 8> loopLocalAllocas;
          for (auto allocaOp : allocaOps) {
            // 检查此alloca是否被此循环使用
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
              processedOps.insert(allocaOp); // 标记为已处理
            }
          }
          
          // 在此循环之前重新创建所有本地alloca
          for (auto allocaOp : loopLocalAllocas) {
            auto newAllocaOp = builder.clone(*allocaOp, mapper);
            
            // 更新映射
            for (unsigned i = 0; i < allocaOp->getNumResults(); ++i) {
              mapper.map(allocaOp->getResult(i), newAllocaOp->getResult(i));
            }
          }
          
          // 克隆循环操作
          Operation *newOp = builder.clone(*node->op, mapper);
          
          // 更新映射
          for (unsigned i = 0; i < node->op->getNumResults(); ++i) {
            mapper.map(node->op->getResult(i), newOp->getResult(i));
          }
          
          // 标记为已处理
          processedOps.insert(node->op);
        }
      }
    } 
    else {
      // === 多内核显式同步方案 ===
      
      // 如果有前一级别的令牌，添加同步点
      if (!previousLevelTokens.empty()) {
        builder.setInsertionPointToEnd(newBlock);
        
        // 显式等待所有前一级别操作完成
        builder.create<gpu::WaitOp>(funcOp.getLoc(), TypeRange{}, previousLevelTokens);
      }
      
      // 为每个内核创建独立的异步令牌
      llvm::SmallVector<Value, 8> kernelWaitTokens;
      for (unsigned i = 0; i < kernelCount; i++) {
        builder.setInsertionPointToEnd(newBlock);
        
        // 创建独立的异步等待操作
        auto waitOp = builder.create<gpu::WaitOp>(
            funcOp.getLoc(),
            builder.getType<gpu::AsyncTokenType>(),
            ValueRange{});  // 无依赖，但会创建一个新的异步令牌
            
        kernelWaitTokens.push_back(waitOp.getAsyncToken());
      }
      
      // 处理所有内核节点
      unsigned kernelIndex = 0;
      for (auto node : nodesAtLevel) {
        builder.setInsertionPointToEnd(newBlock);
        
        if (node->type == NodeType::Kernel) {
          auto kernelOp = cast<gpu::LaunchFuncOp>(node->op);
          
          // 创建内核符号引用
          auto kernelSymbol = SymbolRefAttr::get(
              builder.getContext(),
              kernelOp.getKernelModuleName(),
              {SymbolRefAttr::get(builder.getContext(), kernelOp.getKernelName())});
          
          // 映射操作数
          SmallVector<Value, 8> remappedOperands;
          for (Value operand : kernelOp.getKernelOperands()) {
            remappedOperands.push_back(mapper.lookupOrDefault(operand));
          }
          
          // 映射网格和块大小
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
          
          // 使用此内核的独立等待令牌
          Value waitToken = kernelWaitTokens[kernelIndex++];
              
          // 创建异步内核启动，使用独立等待令牌
          auto newLaunchOp = builder.create<gpu::LaunchFuncOp>(
              kernelOp.getLoc(),
              kernelSymbol,
              mappedGridSize,
              mappedBlockSize,
              Value(),  // 无动态共享内存
              remappedOperands,
              builder.getType<gpu::AsyncTokenType>(),  // 异步令牌类型
              ValueRange{waitToken},  // 使用内核独立的等待令牌
              std::nullopt);  // 无集群大小
              
          // 收集此级别的异步令牌
          levelTokens.push_back(newLaunchOp.getAsyncToken());
              
          // 映射结果
          if (kernelOp->getNumResults() > 0) {
            mapper.map(kernelOp->getResult(0), newLaunchOp->getResult(0));
          }
          
          // 标记为已处理
          processedOps.insert(node->op);
        } 
        else if (node->type == NodeType::Loop) {
          // 处理循环节点
          // 查找与此循环相关的所有memref.alloca操作
          llvm::SmallVector<Operation*, 8> loopLocalAllocas;
          for (auto allocaOp : allocaOps) {
            // 检查此alloca是否被此循环使用
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
              processedOps.insert(allocaOp); // 标记为已处理
            }
          }
          
          // 在此循环之前重新创建所有本地alloca
          for (auto allocaOp : loopLocalAllocas) {
            auto newAllocaOp = builder.clone(*allocaOp, mapper);
            
            // 更新映射
            for (unsigned i = 0; i < allocaOp->getNumResults(); ++i) {
              mapper.map(allocaOp->getResult(i), newAllocaOp->getResult(i));
            }
          }
          
          // 克隆循环操作
          Operation *newOp = builder.clone(*node->op, mapper);
          
          // 更新映射
          for (unsigned i = 0; i < node->op->getNumResults(); ++i) {
            mapper.map(node->op->getResult(i), newOp->getResult(i));
          }
          
          // 标记为已处理
          processedOps.insert(node->op);
        }
      }
      
      // 多内核级别完成后添加同步点
      if (!levelTokens.empty()) {
        builder.setInsertionPointToEnd(newBlock);
        
        // 添加显式同步点等待此级别所有内核完成
        builder.create<gpu::WaitOp>(funcOp.getLoc(), TypeRange{}, levelTokens);
        
        // 由于我们已经同步，清空令牌列表
        // 下一级别不需要等待这些已经同步的令牌
        levelTokens.clear();
      }
    }
    
    // 更新前一级别令牌用于下一级别
    // 对于多内核级别，这里levelTokens已经被清空
    previousLevelTokens = levelTokens;
    
    // 如果这是最后一级并且仍有活跃令牌，保存它们用于最终同步
    if (level == maxLevel && !previousLevelTokens.empty()) {
      finalLevelTokens = previousLevelTokens;
    }
  }
  
  // 处理剩余未使用的alloca操作
  for (auto allocaOp : allocaOps) {
    if (!processedOps.count(allocaOp)) {
      builder.setInsertionPointToEnd(newBlock);
      auto newAllocaOp = builder.clone(*allocaOp, mapper);
      
      // 更新映射
      for (unsigned i = 0; i < allocaOp->getNumResults(); ++i) {
        mapper.map(allocaOp->getResult(i), newAllocaOp->getResult(i));
      }
      
      processedOps.insert(allocaOp);
    }
  }
  
  // 阶段3：复制剩余未处理的操作，但跳过gpu.wait操作
  bool hasReturnOp = false;
  Operation* returnOp = nullptr;
  
  for (auto &op : oldBlock->getOperations()) {
    if (processedOps.count(&op))
      continue;  // 跳过已处理的操作
      
    // 跳过所有GPU等待操作
    if (isa<gpu::WaitOp>(op)) {
      processedOps.insert(&op);
      continue;
    }
    
    // 如果是返回操作，先不克隆，稍后处理
    if (isa<func::ReturnOp>(op)) {
      hasReturnOp = true;
      returnOp = &op;
      continue;
    }
    
    Operation *newOp = op.clone(mapper);
    newBlock->push_back(newOp);
    
    // 更新映射
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      mapper.map(op.getResult(i), newOp->getResult(i));
    }
  }
  
  // 如果有来自最终级别的令牌，添加最终同步点
  if (!finalLevelTokens.empty()) {
    builder.setInsertionPointToEnd(newBlock);
    
    // 添加最终同步等待
    builder.create<gpu::WaitOp>(funcOp.getLoc(), TypeRange{}, finalLevelTokens);
  }
  
  // 如果有返回操作，现在克隆它
  if (hasReturnOp) {
    builder.setInsertionPointToEnd(newBlock);
    Operation *newReturnOp = returnOp->clone(mapper);
    newBlock->push_back(newReturnOp);
  }
  
  // 替换旧块
  // 1. 将新块添加到函数体
  funcOp.getBody().push_back(newBlock);
  
  // 2. 更新使用关系
  for (auto &op : oldBlock->getOperations()) {
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      Value oldResult = op.getResult(i);
      if (mapper.contains(oldResult)) {
        oldResult.replaceAllUsesWith(mapper.lookup(oldResult));
      }
    }
  }
  
  // 3. 删除旧块
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