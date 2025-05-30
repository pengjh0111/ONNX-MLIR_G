#include "mlir/IR/Operation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <map>

#include "IrReorganization.h"
#include "DependencyGraph.h"

using namespace mlir;

namespace onnx_mlir {

// 简化方案：所有级别都使用显式同步，删除链式令牌逻辑
void reorganizeIR(func::FuncOp funcOp, DependencyGraph &graph) {
  OpBuilder builder(funcOp.getContext());
  
  // 安全检查：确保函数体不为空
  if (funcOp.getBody().empty()) {
    llvm::errs() << "Warning: Function body is empty, skipping reorganization\n";
    return;
  }
  
  // 创建映射以跟踪操作映射关系
  IRMapping mapper;
  
  // 按拓扑级别分组节点
  std::map<unsigned, llvm::SmallVector<DependencyNode*, 8>> nodesByLevel;
  for (const auto &nodePair : graph.nodes) {
    DependencyNode* node = nodePair.get();
    nodesByLevel[node->topologicalLevel].push_back(node);
  }
  
  // 如果没有节点需要处理，直接返回
  if (nodesByLevel.empty()) {
    llvm::errs() << "Warning: No nodes to reorganize, skipping\n";
    return;
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
  
  // 预先分析所有GPU wait操作的使用情况
  llvm::DenseMap<Operation*, bool> waitOpShouldKeep;
  // funcOp.walk([&](gpu::WaitOp waitOp) {
  //   bool usedByNonGraphOps = false;
  //   for (auto user : waitOp->getUsers()) {
  //     if (!graph.opToNodeMap.count(user)) {
  //       usedByNonGraphOps = true;
  //       break;
  //     }
  //   }
  //   waitOpShouldKeep[waitOp] = usedByNonGraphOps;
  // });
  funcOp.walk([&](gpu::WaitOp waitOp) {
    // 区分异步和同步wait操作
    bool isAsyncWait = waitOp.getAsyncToken() != nullptr;
    bool isSyncWait = !waitOp.getAsyncDependencies().empty() && !isAsyncWait;
    
    if (isSyncWait) {
      // 同步wait操作（如 gpu.wait [%token]）应该保留
      // 这些是用于同步异步操作的关键同步点
      waitOpShouldKeep[waitOp] = true;
    } else if (isAsyncWait) {
      // 异步wait操作（如 %token = gpu.wait async）
      // 检查是否被非图节点使用
      bool usedByNonGraphOps = false;
      for (auto user : waitOp->getUsers()) {
        if (!graph.opToNodeMap.count(user)) {
          usedByNonGraphOps = true;
          break;
        }
      }
      waitOpShouldKeep[waitOp] = usedByNonGraphOps;
    } else {
      // 其他类型的wait操作，保守起见保留
      waitOpShouldKeep[waitOp] = true;
    }
  });

  
  // 阶段1：首先复制非图节点前缀操作
  for (auto &op : oldBlock->getOperations()) {
    if (graph.opToNodeMap.count(&op)) {
      // 当遇到图中的节点时停止复制前缀操作
      break;
    }
    
    // 记录所有alloca操作，稍后一起处理
    if (isa<memref::AllocaOp>(op)) {
      processedOps.insert(&op);
      continue;
    }
    
    // 对于GPU wait操作，根据预分析结果决定是否保留
    if (auto waitOp = dyn_cast<gpu::WaitOp>(op)) {
      if (waitOpShouldKeep[waitOp]) {
        Operation *newOp = op.clone(mapper);
        newBlock->push_back(newOp);
        
        // 更新映射
        for (unsigned i = 0; i < op.getNumResults(); ++i) {
          mapper.map(op.getResult(i), newOp->getResult(i));
        }
      }
      processedOps.insert(&op);
      continue;
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
  
  // 用于跟踪前一级别的令牌（仅用于gpu.launch_func）
  llvm::SmallVector<Value, 8> previousLevelTokens;
  
  // 阶段2：按拓扑级别处理节点 - 统一使用显式同步方案
  for (unsigned level = 1; level <= maxLevel; level++) {
    auto levelIt = nodesByLevel.find(level);
    if (levelIt == nodesByLevel.end() || levelIt->second.empty())
      continue;
      
    auto &nodesAtLevel = levelIt->second;
    
    // === 统一的显式同步方案 ===
    
    // 步骤1：如果有前一级别的令牌，添加同步点
    if (!previousLevelTokens.empty()) {
      builder.setInsertionPointToEnd(newBlock);
      
      // 显式等待所有前一级别操作完成
      builder.create<gpu::WaitOp>(funcOp.getLoc(), TypeRange{}, previousLevelTokens);
    }
    
    // 步骤2：为当前级别的所有CuLibs节点创建streams
    llvm::SmallVector<Value, 8> culibsStreams;
    for (auto node : nodesAtLevel) {
      if (node->type == NodeType::CuLibs) {
        builder.setInsertionPointToEnd(newBlock);
        
        // 创建stream
        auto streamCreateOp = builder.create<func::CallOp>(
            funcOp.getLoc(),
            "mgpuStreamCreate",
            TypeRange{LLVM::LLVMPointerType::get(builder.getContext())},
            ValueRange{});
        culibsStreams.push_back(streamCreateOp.getResult(0));

        // 创建handles for stream
        builder.create<func::CallOp>(
          funcOp.getLoc(),
          "mgpuCreateHandlesForStream",
          TypeRange{},
          ValueRange{streamCreateOp.getResult(0)});
      }
    }
    
    // 步骤3：为当前级别的所有kernels创建独立的异步令牌
    llvm::SmallVector<Value, 8> kernelWaitTokens;
    llvm::SmallVector<Value, 8> currentLevelTokens;
    
    // 计算kernel数量
    unsigned kernelCount = 0;
    for (auto node : nodesAtLevel) {
      if (node->type == NodeType::Kernel) {
        kernelCount++;
      }
    }
    
    // 为每个kernel创建独立的异步令牌
    for (unsigned i = 0; i < kernelCount; i++) {
      builder.setInsertionPointToEnd(newBlock);
      
      // 创建独立的异步等待操作
      auto waitOp = builder.create<gpu::WaitOp>(
          funcOp.getLoc(),
          builder.getType<gpu::AsyncTokenType>(),
          ValueRange{});  // 无依赖，创建新的异步令牌
          
      kernelWaitTokens.push_back(waitOp.getAsyncToken());
    }
    
    // 步骤4：处理当前级别的所有节点
    // unsigned kernelIndex = 0;
    // unsigned culibsIndex = 0;
    
    // for (auto node : nodesAtLevel) {
    //   builder.setInsertionPointToEnd(newBlock);
      
    //   if (node->type == NodeType::Kernel) {
    //     Value waitToken = kernelWaitTokens[kernelIndex++];
    //     Value kernelToken = processKernelNode(node, builder, mapper, waitToken, processedOps);
    //     currentLevelTokens.push_back(kernelToken);
    //   } 
    //   else if (node->type == NodeType::Loop) {
    //     processLoopNode(node, builder, mapper, allocaOps, processedOps);
    //   } 
    //   else if (node->type == NodeType::CuLibs) {
    //     Value stream = culibsStreams[culibsIndex++];
    //     processCuLibsNodeWithStream(node, builder, mapper, processedOps, stream);
    //   }
    // }
    
    // 对当前级别的节点进行排序：Kernel 节点优先
    llvm::SmallVector<DependencyNode*, 8> sortedNodes;
    llvm::SmallVector<DependencyNode*, 8> kernelNodes;
    llvm::SmallVector<DependencyNode*, 8> loopNodes;  
    llvm::SmallVector<DependencyNode*, 8> culibsNodes;

    // 分离不同类型的节点
    for (auto node : nodesAtLevel) {
      switch (node->type) {
        case NodeType::Kernel:
          kernelNodes.push_back(node);
          break;
        case NodeType::Loop:
          loopNodes.push_back(node);
          break;
        case NodeType::CuLibs:
          culibsNodes.push_back(node);
          break;
        default:
          // 处理未知类型，保守地放在最后
          llvm::errs() << "Warning: Unknown node type encountered\n";
          break;
      }
    }

    // 按优先级排序：Kernel > Loop > CuLibs
    // Kernel 节点优先，因为它们通常执行时间更长，先启动可以获得更好的并行效果
    sortedNodes.append(kernelNodes.begin(), kernelNodes.end());
    sortedNodes.append(loopNodes.begin(), loopNodes.end());
    sortedNodes.append(culibsNodes.begin(), culibsNodes.end());

    // llvm::errs() << "Level " << level << " execution order optimization: " 
    //              << kernelNodes.size() << " kernels first, then " 
    //              << loopNodes.size() << " loops, then " 
    //              << culibsNodes.size() << " culibs calls\n";

    unsigned kernelIndex = 0;
    unsigned culibsIndex = 0;

    // 按排序后的顺序处理节点
    for (auto node : sortedNodes) {
      builder.setInsertionPointToEnd(newBlock);
      
      if (node->type == NodeType::Kernel) {
        Value waitToken = kernelWaitTokens[kernelIndex++];
        Value kernelToken = processKernelNode(node, builder, mapper, waitToken, processedOps);
        currentLevelTokens.push_back(kernelToken);
        
        // llvm::errs() << "  Processed kernel: " << node->kernelModuleName 
        //              << "::" << node->kernelName << " (index " << (kernelIndex-1) << ")\n";
      } 
      else if (node->type == NodeType::Loop) {
        processLoopNode(node, builder, mapper, allocaOps, processedOps);
        
        // llvm::errs() << "  Processed loop node\n";
      } 
      else if (node->type == NodeType::CuLibs) {
        Value stream = culibsStreams[culibsIndex++];
        processCuLibsNodeWithStream(node, builder, mapper, processedOps, stream);
        
        // llvm::errs() << "  Processed culibs call: " << node->culibsFunctionName 
        //              << " (index " << (culibsIndex-1) << ")\n";
      }
    }

    // 步骤5：同步和销毁所有streams
    for (Value stream : culibsStreams) {
      builder.setInsertionPointToEnd(newBlock);
      
      // 同步stream
      builder.create<func::CallOp>(
          funcOp.getLoc(),
          "mgpuStreamSynchronize",
          TypeRange{},
          ValueRange{stream});
      
      // 销毁stream
      builder.create<func::CallOp>(
          funcOp.getLoc(),
          "mgpuStreamDestroy", 
          TypeRange{},
          ValueRange{stream});
    }
    
    // 步骤6：级别完成后的同步处理
    if (!currentLevelTokens.empty()) {
      // 如果不是最后一级，添加显式同步点
      if (level < maxLevel) {
        builder.setInsertionPointToEnd(newBlock);
        
        // 添加显式同步点等待此级别所有kernel完成
        builder.create<gpu::WaitOp>(funcOp.getLoc(), TypeRange{}, currentLevelTokens);
        
        // 清空令牌，下一级别将创建新的独立令牌
        currentLevelTokens.clear();
      }
    }
    
    // 更新前一级别令牌用于下一级别
    previousLevelTokens = currentLevelTokens;
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
  
  // 阶段3：复制剩余未处理的操作
  bool hasReturnOp = false;
  Operation* returnOp = nullptr;
  
  for (auto &op : oldBlock->getOperations()) {
    if (processedOps.count(&op))
      continue;
    
    // 如果是返回操作，先不克隆，稍后处理
    if (isa<func::ReturnOp>(op)) {
      hasReturnOp = true;
      returnOp = &op;
      continue;
    }
    
    // 对于GPU wait操作，根据预分析结果决定是否需要保留
    if (auto waitOp = dyn_cast<gpu::WaitOp>(op)) {
      if (waitOpShouldKeep[waitOp]) {
        Operation *newOp = op.clone(mapper);
        newBlock->push_back(newOp);
        
        // 更新映射
        for (unsigned i = 0; i < op.getNumResults(); ++i) {
          mapper.map(op.getResult(i), newOp->getResult(i));
        }
      }
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
  if (!previousLevelTokens.empty()) {
    builder.setInsertionPointToEnd(newBlock);
    
    // 添加最终同步等待
    builder.create<gpu::WaitOp>(funcOp.getLoc(), TypeRange{}, previousLevelTokens);
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
  
  // 2. 更新使用关系 - 确保所有映射都正确建立
  for (auto &op : oldBlock->getOperations()) {
    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      Value oldResult = op.getResult(i);
      if (mapper.contains(oldResult)) {
        Value newResult = mapper.lookup(oldResult);
        // 替换所有不在旧块中的使用
        llvm::SmallVector<mlir::OpOperand*, 4> usesToReplace;
        for (auto &use : oldResult.getUses()) {
          Operation* userOp = use.getOwner();
          if (userOp->getBlock() != oldBlock) {
            usesToReplace.push_back(&use);
          }
        }
        
        for (auto *use : usesToReplace) {
          use->set(newResult);
        }
      }
    }
  }
  
  // 3. 检查外部引用情况
  bool hasExternalUses = false;
  for (auto &op : oldBlock->getOperations()) {
    for (auto result : op.getResults()) {
      for (auto &use : result.getUses()) {
        if (use.getOwner()->getBlock() != oldBlock) {
          // 只对那些我们没有为其创建映射的操作报告警告
          if (!mapper.contains(result)) {
            llvm::errs() << "Warning: Operation still has external uses: ";
            op.print(llvm::errs());
            llvm::errs() << "\n";
            hasExternalUses = true;
          }
        }
      }
    }
  }
  
  if (hasExternalUses) {
    llvm::errs() << "Error: Cannot safely delete old block due to external references\n";
    return;
  }
  
  // 4. 现在可以安全地删除旧块
  oldBlock->dropAllUses();
  oldBlock->erase();
}


// Helper function to process kernel nodes
Value processKernelNode(DependencyNode* node, OpBuilder& builder, IRMapping& mapper, 
                       Value waitToken, llvm::DenseSet<Operation*>& processedOps) {
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
  
  // 使用等待令牌创建异步内核启动
  auto newLaunchOp = builder.create<gpu::LaunchFuncOp>(
      kernelOp.getLoc(),
      kernelSymbol,
      mappedGridSize,
      mappedBlockSize,
      Value(),  // 无动态共享内存
      remappedOperands,
      builder.getType<gpu::AsyncTokenType>(),  // 异步令牌类型
      ValueRange{waitToken},  // 使用等待令牌
      std::nullopt);  // 无集群大小
      
  // 映射结果
  if (kernelOp->getNumResults() > 0) {
    mapper.map(kernelOp->getResult(0), newLaunchOp->getResult(0));
  }
  
  // 标记为已处理
  processedOps.insert(node->op);
  
  return newLaunchOp.getAsyncToken();
}

// Helper function to process loop nodes
void processLoopNode(DependencyNode* node, OpBuilder& builder, IRMapping& mapper,
                    llvm::SmallVector<Operation*, 16>& allocaOps,
                    llvm::DenseSet<Operation*>& processedOps) {
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

// Helper function to process CuLibs nodes (legacy version with integrated stream management)
void processCuLibsNode(DependencyNode* node, OpBuilder& builder, IRMapping& mapper,
                      llvm::DenseSet<Operation*>& processedOps) {
  // 首先收集所有需要的参数准备操作
  llvm::SetVector<Operation*> requiredOps;
  
  // 为每个CuLibs操作收集其依赖的操作
  for (Operation* culibsOp : node->culibsOps) {
    // 递归收集这个操作的所有依赖
    collectDependentOps(culibsOp, requiredOps, processedOps);
  }
  
  // 按依赖顺序处理所有必需的操作
  for (Operation* requiredOp : requiredOps) {
    if (!processedOps.count(requiredOp)) {
      Operation* newOp = builder.clone(*requiredOp, mapper);
      
      // 更新映射
      for (unsigned i = 0; i < requiredOp->getNumResults(); ++i) {
        mapper.map(requiredOp->getResult(i), newOp->getResult(i));
      }
      
      // 标记为已处理
      processedOps.insert(requiredOp);
    }
  }
  
  // 然后按顺序克隆CuLibs操作序列
  for (Operation* culibsOp : node->culibsOps) {
    if (!processedOps.count(culibsOp)) {
      // 克隆每个操作
      Operation* newOp = builder.clone(*culibsOp, mapper);
      
      // 更新映射
      for (unsigned i = 0; i < culibsOp->getNumResults(); ++i) {
        mapper.map(culibsOp->getResult(i), newOp->getResult(i));
      }
      
      // 标记为已处理
      processedOps.insert(culibsOp);
    }
  }
}

// Helper function to process CuLibs nodes with a pre-created stream
void processCuLibsNodeWithStream(DependencyNode* node, OpBuilder& builder, IRMapping& mapper,
                                llvm::DenseSet<Operation*>& processedOps, Value stream) {
  // 首先收集所有需要的参数准备操作
  llvm::SetVector<Operation*> requiredOps;
  
  // 为主要的CuLibs调用收集其依赖的操作（跳过stream管理操作）
  Operation* mainCall = nullptr;
  for (Operation* culibsOp : node->culibsOps) {
    if (isCuLibsCall(culibsOp)) {
      mainCall = culibsOp;
      break;
    }
  }
  
  if (mainCall) {
    // 递归收集这个操作的所有依赖
    collectDependentOps(mainCall, requiredOps, processedOps);
    
    // 按依赖顺序处理所有必需的操作
    for (Operation* requiredOp : requiredOps) {
      if (!processedOps.count(requiredOp)) {
        Operation* newOp = builder.clone(*requiredOp, mapper);
        
        // 更新映射
        for (unsigned i = 0; i < requiredOp->getNumResults(); ++i) {
          mapper.map(requiredOp->getResult(i), newOp->getResult(i));
        }
        
        // 标记为已处理
        processedOps.insert(requiredOp);
      }
    }
    
    // 克隆主要的CuLibs调用，但使用预先创建的stream
    if (!processedOps.count(mainCall)) {
      auto callOp = cast<func::CallOp>(mainCall);
      
      // 准备操作数，将最后一个操作数（原来的stream）替换为我们的stream
      llvm::SmallVector<Value, 8> newOperands;
      for (unsigned i = 0; i < callOp.getNumOperands() - 1; ++i) {
        newOperands.push_back(mapper.lookupOrDefault(callOp.getOperand(i)));
      }
      newOperands.push_back(stream); // 使用预先创建的stream
      
      // 创建新的调用
      auto newCallOp = builder.create<func::CallOp>(
          callOp.getLoc(),
          callOp.getCallee(),
          callOp.getResultTypes(),
          newOperands);
      
      // 更新映射
      for (unsigned i = 0; i < mainCall->getNumResults(); ++i) {
        mapper.map(mainCall->getResult(i), newCallOp.getResult(i));
      }
      
      // 标记为已处理
      processedOps.insert(mainCall);
    }
  }
  
  // 标记所有相关操作为已处理（包括stream管理操作）
  for (Operation* culibsOp : node->culibsOps) {
    processedOps.insert(culibsOp);
  }
}

// Helper function to collect dependent operations
void collectDependentOps(Operation* op, llvm::SetVector<Operation*>& requiredOps, 
                        const llvm::DenseSet<Operation*>& processedOps) {
  // 遍历操作的所有操作数
  for (Value operand : op->getOperands()) {
    if (Operation* definingOp = operand.getDefiningOp()) {
      // 如果定义操作还没有被处理，且不是Block参数
      if (!processedOps.count(definingOp) && !isa<BlockArgument>(operand)) {
        // 检查是否是我们需要移动的操作类型
        if (shouldMoveWithCuLibs(definingOp)) {
          // 递归收集这个操作的依赖
          collectDependentOps(definingOp, requiredOps, processedOps);
          // 添加到必需操作列表
          requiredOps.insert(definingOp);
        }
      }
    }
  }
}

// Helper function to determine if an operation should be moved with CuLibs calls
bool shouldMoveWithCuLibs(Operation* op) {
  // 这些操作通常是为CuLibs调用准备参数的
  return isa<memref::ExtractAlignedPointerAsIndexOp>(op) ||
         isa<arith::IndexCastOp>(op) ||
         isa<mlir::LLVM::IntToPtrOp>(op) ||
         isa<memref::AllocOp>(op) ||
         isa<arith::ConstantOp>(op);  // 常量也可能需要
}

// multi_gpu_module version
// void reorganizeGPUModules(ModuleOp moduleOp, DependencyGraph &graph) {
//   OpBuilder builder(moduleOp.getContext());
  
//   // Scan all existing modules and functions
//   llvm::SmallVector<gpu::GPUModuleOp, 4> allModules;
//   std::map<std::pair<std::string, std::string>, gpu::GPUFuncOp> funcMap;
  
//   moduleOp.walk([&](gpu::GPUModuleOp op) {
//     allModules.push_back(op);
    
//     // Collect all functions in this module
//     std::string moduleName = op.getName().str();
//     op.walk([&](gpu::GPUFuncOp funcOp) {
//       std::string funcName = funcOp.getName().str();
//       funcMap[{moduleName, funcName}] = funcOp;
//     });
//   });
  
//   // Exit if no modules
//   if (allModules.empty())
//     return;
  
//   // Group kernel nodes by topological level
//   std::map<unsigned, llvm::SmallVector<DependencyNode*, 8>> kernelsByLevel;
//   for (const auto &nodePair : graph.nodes) {
//     DependencyNode* node = nodePair.get();
//     if (node->type == NodeType::Kernel) {
//       kernelsByLevel[node->topologicalLevel].push_back(node);
//     }
//   }
  
//   // Create renaming map: <old module name, old function name> -> <new module name, new function name>
//   using ModuleFuncKey = std::pair<std::string, std::string>;
//   std::map<ModuleFuncKey, ModuleFuncKey> renameMap;
  
//   // Create a counter for each topological level to ensure function name uniqueness
//   std::map<unsigned, int> levelFuncCounter;
  
//   // Step 1: Create a module for each topological level and copy corresponding functions
//   for (const auto &levelPair : kernelsByLevel) {
//     unsigned level = levelPair.first;
//     const auto &kernels = levelPair.second;
    
//     // Skip empty levels
//     if (kernels.empty())
//       continue;
    
//     // Initialize function counter for this level
//     levelFuncCounter[level] = 0;
    
//     // Create a new module for this level
//     std::string newModuleName = "level_" + std::to_string(level) + "_module";
//     builder.setInsertionPointToStart(moduleOp.getBody());
    
//     auto levelModule = builder.create<gpu::GPUModuleOp>(
//         moduleOp.getLoc(),
//         builder.getStringAttr(newModuleName));
    
//     builder.setInsertionPointToStart(levelModule.getBody());
    
//     // Copy all kernel functions for this level
//     for (DependencyNode* kernel : kernels) {
//       std::string oldModuleName = kernel->kernelModuleName.str();
//       std::string oldFuncName = kernel->kernelName.str();
      
//       // Find the original function
//       auto funcKey = std::make_pair(oldModuleName, oldFuncName);
//       auto funcIt = funcMap.find(funcKey);
      
//       if (funcIt != funcMap.end()) {
//         // Create a new unique function name
//         std::string newFuncName = "kernel_" + std::to_string(level) + "_" + 
//                                   std::to_string(levelFuncCounter[level]++);
        
//         // Clone the function to the new module
//         auto clonedFunc = cast<gpu::GPUFuncOp>(builder.clone(*funcIt->second));
        
//         // Set the new function name
//         clonedFunc.setName(newFuncName);
        
//         // Save mapping relationship
//         renameMap[funcKey] = {newModuleName, newFuncName};
//       }
//     }
//   }
  
//   // Step 2: Update all kernel launch references
//   moduleOp.walk([&](gpu::LaunchFuncOp op) {
//     std::string oldModuleName = op.getKernelModuleName().str();
//     std::string oldFuncName = op.getKernelName().str();
    
//     auto funcKey = std::make_pair(oldModuleName, oldFuncName);
//     auto renameIt = renameMap.find(funcKey);
    
//     if (renameIt != renameMap.end()) {
//       std::string newModuleName = renameIt->second.first;
//       std::string newFuncName = renameIt->second.second;
      
//       // Create new symbol reference
//       auto newKernel = SymbolRefAttr::get(
//           builder.getContext(),
//           StringAttr::get(builder.getContext(), newModuleName),
//           {SymbolRefAttr::get(builder.getContext(), newFuncName)});
      
//       // Update kernel reference
//       op->setAttr("kernel", newKernel);
//     }
//   });
  
//   // Step 3: Delete old modules
//   for (auto moduleOp : allModules) {
//     moduleOp.erase();
//   }
// }

// Single combined module version
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
  
  // Collect all kernel nodes from the dependency graph
  llvm::SmallVector<DependencyNode*, 16> allKernels;
  for (const auto &nodePair : graph.nodes) {
    DependencyNode* node = nodePair.get();
    if (node->type == NodeType::Kernel) {
      allKernels.push_back(node);
    }
  }
  
  // Exit if no kernels to process
  if (allKernels.empty()) {
    // Still need to clean up old modules even if no kernels
    for (auto moduleOp : allModules) {
      moduleOp.erase();
    }
    return;
  }
  
  // Create renaming map: <old module name, old function name> -> <new module name, new function name>
  using ModuleFuncKey = std::pair<std::string, std::string>;
  std::map<ModuleFuncKey, ModuleFuncKey> renameMap;
  
  // Step 1: Create a single combined module for all kernels
  std::string combinedModuleName = "combined_kernels_module";
  builder.setInsertionPointToStart(moduleOp.getBody());
  
  auto combinedModule = builder.create<gpu::GPUModuleOp>(
      moduleOp.getLoc(),
      builder.getStringAttr(combinedModuleName));
  
  builder.setInsertionPointToStart(combinedModule.getBody());
  
  // Function counter to ensure uniqueness across all kernels
  int globalFuncCounter = 0;
  
  // Step 2: Copy all kernel functions to the combined module
  for (DependencyNode* kernel : allKernels) {
    std::string oldModuleName = kernel->kernelModuleName.str();
    std::string oldFuncName = kernel->kernelName.str();
    
    // Find the original function
    auto funcKey = std::make_pair(oldModuleName, oldFuncName);
    auto funcIt = funcMap.find(funcKey);
    
    if (funcIt != funcMap.end()) {
      // Create a new unique function name
      // Include level information in the name for easier identification
      std::string newFuncName = "kernel_L" + std::to_string(kernel->topologicalLevel) + 
                                "_" + std::to_string(globalFuncCounter++);
      
      // Clone the function to the combined module
      auto clonedFunc = cast<gpu::GPUFuncOp>(builder.clone(*funcIt->second));
      
      // Set the new function name
      clonedFunc.setName(newFuncName);
      
      // Save mapping relationship
      renameMap[funcKey] = {combinedModuleName, newFuncName};
    }
  }
  
  // Step 3: Update all kernel launch references to point to the combined module
  moduleOp.walk([&](gpu::LaunchFuncOp op) {
    std::string oldModuleName = op.getKernelModuleName().str();
    std::string oldFuncName = op.getKernelName().str();
    
    auto funcKey = std::make_pair(oldModuleName, oldFuncName);
    auto renameIt = renameMap.find(funcKey);
    
    if (renameIt != renameMap.end()) {
      std::string newModuleName = renameIt->second.first;
      std::string newFuncName = renameIt->second.second;
      
      // Create new symbol reference pointing to the combined module
      auto newKernel = SymbolRefAttr::get(
          builder.getContext(),
          StringAttr::get(builder.getContext(), newModuleName),
          {SymbolRefAttr::get(builder.getContext(), newFuncName)});
      
      // Update kernel reference
      op->setAttr("kernel", newKernel);
    }
  });
  
  // Step 4: Delete all old modules
  for (auto moduleOp : allModules) {
    moduleOp.erase();
  }
}

} // namespace onnx_mlir

// #include "mlir/IR/Operation.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/IR/IRMapping.h"
// #include "mlir/Dialect/GPU/IR/GPUDialect.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// #include "mlir/Dialect/SCF/IR/SCF.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "llvm/ADT/DenseMap.h"
// #include "llvm/ADT/SmallVector.h"
// #include <map>

// #include "IrReorganization.h"
// #include "DependencyGraph.h"

// using namespace mlir;

// namespace onnx_mlir {

// // Helper function to ensure stream pool function declarations exist
// void ensureStreamPoolFunctionDeclarations(ModuleOp moduleOp, OpBuilder& builder) {
//   // 检查并添加stream pool相关函数声明
//   llvm::SmallVector<std::pair<StringRef, FunctionType>, 4> requiredFunctions;
  
//   auto context = builder.getContext();
//   auto ptrType = LLVM::LLVMPointerType::get(context);
//   auto voidType = builder.getNoneType();
  
//   // 添加需要的函数声明
//   requiredFunctions.push_back({
//     "mgpuInitStreamPool", 
//     builder.getFunctionType({}, {})
//   });
  
//   requiredFunctions.push_back({
//     "mgpuDestroyStreamPool", 
//     builder.getFunctionType({}, {})
//   });
  
//   requiredFunctions.push_back({
//     "mgpuAcquirePooledStream", 
//     builder.getFunctionType({}, {ptrType})
//   });
  
//   requiredFunctions.push_back({
//     "mgpuReleasePooledStream", 
//     builder.getFunctionType({ptrType}, {})
//   });
  
//   // 检查函数是否已存在，如果不存在则添加声明
//   for (auto &funcInfo : requiredFunctions) {
//     StringRef funcName = funcInfo.first;
//     FunctionType funcType = funcInfo.second;
    
//     if (!moduleOp.lookupSymbol<func::FuncOp>(funcName)) {
//       // 函数不存在，添加声明
//       builder.setInsertionPointToStart(moduleOp.getBody());
      
//       auto funcOp = builder.create<func::FuncOp>(
//           moduleOp.getLoc(),
//           funcName,
//           funcType);
      
//       // 标记为私有，表示这是外部函数
//       funcOp.setPrivate();
//     }
//   }
// }

// // 简化方案：所有级别都使用显式同步，删除链式令牌逻辑，使用stream pool
// void reorganizeIR(func::FuncOp funcOp, DependencyGraph &graph) {
//   OpBuilder builder(funcOp.getContext());
  
//   // 安全检查：确保函数体不为空
//   if (funcOp.getBody().empty()) {
//     llvm::errs() << "Warning: Function body is empty, skipping reorganization\n";
//     return;
//   }
  
//   // **新增：确保stream pool函数声明存在**
//   ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
//   if (moduleOp) {
//     ensureStreamPoolFunctionDeclarations(moduleOp, builder);
//   }
  
//   // 创建映射以跟踪操作映射关系
//   IRMapping mapper;
  
//   // 按拓扑级别分组节点
//   std::map<unsigned, llvm::SmallVector<DependencyNode*, 8>> nodesByLevel;
//   for (const auto &nodePair : graph.nodes) {
//     DependencyNode* node = nodePair.get();
//     nodesByLevel[node->topologicalLevel].push_back(node);
//   }
  
//   // 如果没有节点需要处理，直接返回
//   if (nodesByLevel.empty()) {
//     llvm::errs() << "Warning: No nodes to reorganize, skipping\n";
//     return;
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
  
//   // 预先分析所有GPU wait操作的使用情况
//   llvm::DenseMap<Operation*, bool> waitOpShouldKeep;
//   funcOp.walk([&](gpu::WaitOp waitOp) {
//     // 区分异步和同步wait操作
//     bool isAsyncWait = waitOp.getAsyncToken() != nullptr;
//     bool isSyncWait = !waitOp.getAsyncDependencies().empty() && !isAsyncWait;
    
//     if (isSyncWait) {
//       // 同步wait操作（如 gpu.wait [%token]）应该保留
//       // 这些是用于同步异步操作的关键同步点
//       waitOpShouldKeep[waitOp] = true;
//     } else if (isAsyncWait) {
//       // 异步wait操作（如 %token = gpu.wait async）
//       // 检查是否被非图节点使用
//       bool usedByNonGraphOps = false;
//       for (auto user : waitOp->getUsers()) {
//         if (!graph.opToNodeMap.count(user)) {
//           usedByNonGraphOps = true;
//           break;
//         }
//       }
//       waitOpShouldKeep[waitOp] = usedByNonGraphOps;
//     } else {
//       // 其他类型的wait操作，保守起见保留
//       waitOpShouldKeep[waitOp] = true;
//     }
//   });

  
//   // 阶段1：首先复制非图节点前缀操作
//   for (auto &op : oldBlock->getOperations()) {
//     if (graph.opToNodeMap.count(&op)) {
//       // 当遇到图中的节点时停止复制前缀操作
//       break;
//     }
    
//     // 记录所有alloca操作，稍后一起处理
//     if (isa<memref::AllocaOp>(op)) {
//       processedOps.insert(&op);
//       continue;
//     }
    
//     // 对于GPU wait操作，根据预分析结果决定是否保留
//     if (auto waitOp = dyn_cast<gpu::WaitOp>(op)) {
//       if (waitOpShouldKeep[waitOp]) {
//         Operation *newOp = op.clone(mapper);
//         newBlock->push_back(newOp);
        
//         // 更新映射
//         for (unsigned i = 0; i < op.getNumResults(); ++i) {
//           mapper.map(op.getResult(i), newOp->getResult(i));
//         }
//       }
//       processedOps.insert(&op);
//       continue;
//     }
    
//     Operation *newOp = op.clone(mapper);
//     newBlock->push_back(newOp);
    
//     // 更新映射并标记为已处理
//     for (unsigned i = 0; i < op.getNumResults(); ++i) {
//       mapper.map(op.getResult(i), newOp->getResult(i));
//     }
//     processedOps.insert(&op);
//   }
  
//   // **新增：在处理计算节点之前初始化stream pool**
//   builder.setInsertionPointToEnd(newBlock);
//   builder.create<func::CallOp>(
//       funcOp.getLoc(),
//       "mgpuInitStreamPool",
//       TypeRange{},
//       ValueRange{});
  
//   // 找到最大拓扑级别
//   unsigned maxLevel = 0;
//   for (const auto &nodePair : graph.nodes) {
//     maxLevel = std::max(maxLevel, nodePair.get()->topologicalLevel);
//   }
  
//   // 用于跟踪前一级别的令牌（仅用于gpu.launch_func）
//   llvm::SmallVector<Value, 8> previousLevelTokens;
  
//   // 阶段2：按拓扑级别处理节点 - 统一使用显式同步方案
//   for (unsigned level = 1; level <= maxLevel; level++) {
//     auto levelIt = nodesByLevel.find(level);
//     if (levelIt == nodesByLevel.end() || levelIt->second.empty())
//       continue;
      
//     auto &nodesAtLevel = levelIt->second;
    
//     // === 统一的显式同步方案 ===
    
//     // 步骤1：如果有前一级别的令牌，添加同步点
//     if (!previousLevelTokens.empty()) {
//       builder.setInsertionPointToEnd(newBlock);
      
//       // 显式等待所有前一级别操作完成
//       builder.create<gpu::WaitOp>(funcOp.getLoc(), TypeRange{}, previousLevelTokens);
//     }
    
//     // 步骤2：为当前级别的所有CuLibs节点获取streams
//     llvm::SmallVector<Value, 8> culibsStreams;
//     for (auto node : nodesAtLevel) {
//       if (node->type == NodeType::CuLibs) {
//         builder.setInsertionPointToEnd(newBlock);
        
//         // **修改：使用stream pool获取stream**
//         auto streamAcquireOp = builder.create<func::CallOp>(
//             funcOp.getLoc(),
//             "mgpuAcquirePooledStream",
//             TypeRange{LLVM::LLVMPointerType::get(builder.getContext())},
//             ValueRange{});
//         culibsStreams.push_back(streamAcquireOp.getResult(0));

//         // 创建handles for stream
//         builder.create<func::CallOp>(
//           funcOp.getLoc(),
//           "mgpuAcquirePooledHandles",
//           TypeRange{},
//           ValueRange{streamAcquireOp.getResult(0)});
//       }
//     }
    
//     // 步骤3：为当前级别的所有kernels创建独立的异步令牌
//     llvm::SmallVector<Value, 8> kernelWaitTokens;
//     llvm::SmallVector<Value, 8> currentLevelTokens;
    
//     // 计算kernel数量
//     unsigned kernelCount = 0;
//     for (auto node : nodesAtLevel) {
//       if (node->type == NodeType::Kernel) {
//         kernelCount++;
//       }
//     }
    
//     // 为每个kernel创建独立的异步令牌
//     for (unsigned i = 0; i < kernelCount; i++) {
//       builder.setInsertionPointToEnd(newBlock);
      
//       // 创建独立的异步等待操作
//       auto waitOp = builder.create<gpu::WaitOp>(
//           funcOp.getLoc(),
//           builder.getType<gpu::AsyncTokenType>(),
//           ValueRange{});  // 无依赖，创建新的异步令牌
          
//       kernelWaitTokens.push_back(waitOp.getAsyncToken());
//     }
    
//     // 步骤4：处理当前级别的所有节点
//     unsigned kernelIndex = 0;
//     unsigned culibsIndex = 0;
    
//     for (auto node : nodesAtLevel) {
//       builder.setInsertionPointToEnd(newBlock);
      
//       if (node->type == NodeType::Kernel) {
//         Value waitToken = kernelWaitTokens[kernelIndex++];
//         Value kernelToken = processKernelNode(node, builder, mapper, waitToken, processedOps);
//         currentLevelTokens.push_back(kernelToken);
//       } 
//       else if (node->type == NodeType::Loop) {
//         processLoopNode(node, builder, mapper, allocaOps, processedOps);
//       } 
//       else if (node->type == NodeType::CuLibs) {
//         Value stream = culibsStreams[culibsIndex++];
//         processCuLibsNodeWithStreamPool(node, builder, mapper, processedOps, stream);
//       }
//     }
    
//     // 步骤5：同步和释放所有streams（使用stream pool）
//     for (Value stream : culibsStreams) {
//       builder.setInsertionPointToEnd(newBlock);
      
//       // 同步stream
//       builder.create<func::CallOp>(
//           funcOp.getLoc(),
//           "mgpuStreamSynchronize",
//           TypeRange{},
//           ValueRange{stream});
      
//       // **修改：使用stream pool释放stream**
//       builder.create<func::CallOp>(
//           funcOp.getLoc(),
//           "mgpuReleasePooledStream", 
//           TypeRange{},
//           ValueRange{stream});
//     }
    
//     // 步骤6：级别完成后的同步处理
//     if (!currentLevelTokens.empty()) {
//       // 如果不是最后一级，添加显式同步点
//       if (level < maxLevel) {
//         builder.setInsertionPointToEnd(newBlock);
        
//         // 添加显式同步点等待此级别所有kernel完成
//         builder.create<gpu::WaitOp>(funcOp.getLoc(), TypeRange{}, currentLevelTokens);
        
//         // 清空令牌，下一级别将创建新的独立令牌
//         currentLevelTokens.clear();
//       }
//     }
    
//     // 更新前一级别令牌用于下一级别
//     previousLevelTokens = currentLevelTokens;
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
  
//   // 阶段3：复制剩余未处理的操作
//   bool hasReturnOp = false;
//   Operation* returnOp = nullptr;
  
//   for (auto &op : oldBlock->getOperations()) {
//     if (processedOps.count(&op))
//       continue;
    
//     // 如果是返回操作，先不克隆，稍后处理
//     if (isa<func::ReturnOp>(op)) {
//       hasReturnOp = true;
//       returnOp = &op;
//       continue;
//     }
    
//     // 对于GPU wait操作，根据预分析结果决定是否需要保留
//     if (auto waitOp = dyn_cast<gpu::WaitOp>(op)) {
//       if (waitOpShouldKeep[waitOp]) {
//         Operation *newOp = op.clone(mapper);
//         newBlock->push_back(newOp);
        
//         // 更新映射
//         for (unsigned i = 0; i < op.getNumResults(); ++i) {
//           mapper.map(op.getResult(i), newOp->getResult(i));
//         }
//       }
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
  
//   // **新增：在返回前销毁stream pool**
//   builder.setInsertionPointToEnd(newBlock);
//   builder.create<func::CallOp>(
//       funcOp.getLoc(),
//       "mgpuDestroyStreamPool",
//       TypeRange{},
//       ValueRange{});
  
//   // 如果有返回操作，现在克隆它
//   if (hasReturnOp) {
//     builder.setInsertionPointToEnd(newBlock);
//     Operation *newReturnOp = returnOp->clone(mapper);
//     newBlock->push_back(newReturnOp);
//   }
  
//   // 替换旧块
//   // 1. 将新块添加到函数体
//   funcOp.getBody().push_back(newBlock);
  
//   // 2. 更新使用关系 - 确保所有映射都正确建立
//   for (auto &op : oldBlock->getOperations()) {
//     for (unsigned i = 0; i < op.getNumResults(); ++i) {
//       Value oldResult = op.getResult(i);
//       if (mapper.contains(oldResult)) {
//         Value newResult = mapper.lookup(oldResult);
//         // 替换所有不在旧块中的使用
//         llvm::SmallVector<mlir::OpOperand*, 4> usesToReplace;
//         for (auto &use : oldResult.getUses()) {
//           Operation* userOp = use.getOwner();
//           if (userOp->getBlock() != oldBlock) {
//             usesToReplace.push_back(&use);
//           }
//         }
        
//         for (auto *use : usesToReplace) {
//           use->set(newResult);
//         }
//       }
//     }
//   }
  
//   // 3. 检查外部引用情况
//   bool hasExternalUses = false;
//   for (auto &op : oldBlock->getOperations()) {
//     for (auto result : op.getResults()) {
//       for (auto &use : result.getUses()) {
//         if (use.getOwner()->getBlock() != oldBlock) {
//           // 只对那些我们没有为其创建映射的操作报告警告
//           if (!mapper.contains(result)) {
//             llvm::errs() << "Warning: Operation still has external uses: ";
//             op.print(llvm::errs());
//             llvm::errs() << "\n";
//             hasExternalUses = true;
//           }
//         }
//       }
//     }
//   }
  
//   if (hasExternalUses) {
//     llvm::errs() << "Error: Cannot safely delete old block due to external references\n";
//     return;
//   }
  
//   // 4. 现在可以安全地删除旧块
//   oldBlock->dropAllUses();
//   oldBlock->erase();
// }


// // Helper function to process kernel nodes
// Value processKernelNode(DependencyNode* node, OpBuilder& builder, IRMapping& mapper, 
//                        Value waitToken, llvm::DenseSet<Operation*>& processedOps) {
//   auto kernelOp = cast<gpu::LaunchFuncOp>(node->op);
  
//   // 创建内核符号引用
//   auto kernelSymbol = SymbolRefAttr::get(
//       builder.getContext(),
//       kernelOp.getKernelModuleName(),
//       {SymbolRefAttr::get(builder.getContext(), kernelOp.getKernelName())});
  
//   // 映射操作数
//   SmallVector<Value, 8> remappedOperands;
//   for (Value operand : kernelOp.getKernelOperands()) {
//     remappedOperands.push_back(mapper.lookupOrDefault(operand));
//   }
  
//   // 映射网格和块大小
//   auto gridSize = kernelOp.getGridSizeOperandValues();
//   auto blockSize = kernelOp.getBlockSizeOperandValues();
  
//   mlir::gpu::KernelDim3 mappedGridSize = {
//     mapper.lookupOrDefault(gridSize.x),
//     mapper.lookupOrDefault(gridSize.y),
//     mapper.lookupOrDefault(gridSize.z)
//   };
  
//   mlir::gpu::KernelDim3 mappedBlockSize = {
//     mapper.lookupOrDefault(blockSize.x),
//     mapper.lookupOrDefault(blockSize.y),
//     mapper.lookupOrDefault(blockSize.z)
//   };
  
//   // 使用等待令牌创建异步内核启动
//   auto newLaunchOp = builder.create<gpu::LaunchFuncOp>(
//       kernelOp.getLoc(),
//       kernelSymbol,
//       mappedGridSize,
//       mappedBlockSize,
//       Value(),  // 无动态共享内存
//       remappedOperands,
//       builder.getType<gpu::AsyncTokenType>(),  // 异步令牌类型
//       ValueRange{waitToken},  // 使用等待令牌
//       std::nullopt);  // 无集群大小
      
//   // 映射结果
//   if (kernelOp->getNumResults() > 0) {
//     mapper.map(kernelOp->getResult(0), newLaunchOp->getResult(0));
//   }
  
//   // 标记为已处理
//   processedOps.insert(node->op);
  
//   return newLaunchOp.getAsyncToken();
// }

// // Helper function to process loop nodes
// void processLoopNode(DependencyNode* node, OpBuilder& builder, IRMapping& mapper,
//                     llvm::SmallVector<Operation*, 16>& allocaOps,
//                     llvm::DenseSet<Operation*>& processedOps) {
//   // 查找与此循环相关的所有memref.alloca操作
//   llvm::SmallVector<Operation*, 8> loopLocalAllocas;
//   for (auto allocaOp : allocaOps) {
//     // 检查此alloca是否被此循环使用
//     bool used = false;
//     Value allocaResult = allocaOp->getResult(0);
//     node->op->walk([&](Operation *user) {
//       for (Value operand : user->getOperands()) {
//         if (operand == allocaResult) {
//           used = true;
//           return WalkResult::interrupt();
//         }
//       }
//       return WalkResult::advance();
//     });
    
//     if (used) {
//       loopLocalAllocas.push_back(allocaOp);
//       processedOps.insert(allocaOp); // 标记为已处理
//     }
//   }
  
//   // 在此循环之前重新创建所有本地alloca
//   for (auto allocaOp : loopLocalAllocas) {
//     auto newAllocaOp = builder.clone(*allocaOp, mapper);
    
//     // 更新映射
//     for (unsigned i = 0; i < allocaOp->getNumResults(); ++i) {
//       mapper.map(allocaOp->getResult(i), newAllocaOp->getResult(i));
//     }
//   }
  
//   // 克隆循环操作
//   Operation *newOp = builder.clone(*node->op, mapper);
  
//   // 更新映射
//   for (unsigned i = 0; i < node->op->getNumResults(); ++i) {
//     mapper.map(node->op->getResult(i), newOp->getResult(i));
//   }
  
//   // 标记为已处理
//   processedOps.insert(node->op);
// }

// // Helper function to process CuLibs nodes (legacy version with integrated stream management)
// void processCuLibsNode(DependencyNode* node, OpBuilder& builder, IRMapping& mapper,
//                       llvm::DenseSet<Operation*>& processedOps) {
//   // 首先收集所有需要的参数准备操作
//   llvm::SetVector<Operation*> requiredOps;
  
//   // 为每个CuLibs操作收集其依赖的操作
//   for (Operation* culibsOp : node->culibsOps) {
//     // 递归收集这个操作的所有依赖
//     collectDependentOps(culibsOp, requiredOps, processedOps);
//   }
  
//   // 按依赖顺序处理所有必需的操作
//   for (Operation* requiredOp : requiredOps) {
//     if (!processedOps.count(requiredOp)) {
//       Operation* newOp = builder.clone(*requiredOp, mapper);
      
//       // 更新映射
//       for (unsigned i = 0; i < requiredOp->getNumResults(); ++i) {
//         mapper.map(requiredOp->getResult(i), newOp->getResult(i));
//       }
      
//       // 标记为已处理
//       processedOps.insert(requiredOp);
//     }
//   }
  
//   // 然后按顺序克隆CuLibs操作序列
//   for (Operation* culibsOp : node->culibsOps) {
//     if (!processedOps.count(culibsOp)) {
//       // 克隆每个操作
//       Operation* newOp = builder.clone(*culibsOp, mapper);
      
//       // 更新映射
//       for (unsigned i = 0; i < culibsOp->getNumResults(); ++i) {
//         mapper.map(culibsOp->getResult(i), newOp->getResult(i));
//       }
      
//       // 标记为已处理
//       processedOps.insert(culibsOp);
//     }
//   }
// }

// // **新增：Helper function to process CuLibs nodes with stream pool**
// void processCuLibsNodeWithStreamPool(DependencyNode* node, OpBuilder& builder, IRMapping& mapper,
//                                     llvm::DenseSet<Operation*>& processedOps, Value stream) {
//   // 首先收集所有需要的参数准备操作
//   llvm::SetVector<Operation*> requiredOps;
  
//   // 为主要的CuLibs调用收集其依赖的操作（跳过stream管理操作）
//   Operation* mainCall = nullptr;
//   for (Operation* culibsOp : node->culibsOps) {
//     if (isCuLibsCall(culibsOp)) {
//       mainCall = culibsOp;
//       break;
//     }
//   }
  
//   if (mainCall) {
//     // 递归收集这个操作的所有依赖
//     collectDependentOps(mainCall, requiredOps, processedOps);
    
//     // 按依赖顺序处理所有必需的操作
//     for (Operation* requiredOp : requiredOps) {
//       if (!processedOps.count(requiredOp)) {
//         Operation* newOp = builder.clone(*requiredOp, mapper);
        
//         // 更新映射
//         for (unsigned i = 0; i < requiredOp->getNumResults(); ++i) {
//           mapper.map(requiredOp->getResult(i), newOp->getResult(i));
//         }
        
//         // 标记为已处理
//         processedOps.insert(requiredOp);
//       }
//     }
    
//     // 克隆主要的CuLibs调用，但使用来自stream pool的stream
//     if (!processedOps.count(mainCall)) {
//       auto callOp = cast<func::CallOp>(mainCall);
      
//       // 准备操作数，将最后一个操作数（原来的stream）替换为我们的stream
//       llvm::SmallVector<Value, 8> newOperands;
//       for (unsigned i = 0; i < callOp.getNumOperands() - 1; ++i) {
//         newOperands.push_back(mapper.lookupOrDefault(callOp.getOperand(i)));
//       }
//       newOperands.push_back(stream); // 使用来自stream pool的stream
      
//       // 创建新的调用
//       auto newCallOp = builder.create<func::CallOp>(
//           callOp.getLoc(),
//           callOp.getCallee(),
//           callOp.getResultTypes(),
//           newOperands);
      
//       // 更新映射
//       for (unsigned i = 0; i < mainCall->getNumResults(); ++i) {
//         mapper.map(mainCall->getResult(i), newCallOp.getResult(i));
//       }
      
//       // 标记为已处理
//       processedOps.insert(mainCall);
//     }
//   }
  
//   // 标记所有相关操作为已处理（包括原来的stream管理操作）
//   for (Operation* culibsOp : node->culibsOps) {
//     processedOps.insert(culibsOp);
//   }
// }

// // Helper function to process CuLibs nodes with a pre-created stream (保留原版本以兼容)
// void processCuLibsNodeWithStream(DependencyNode* node, OpBuilder& builder, IRMapping& mapper,
//                                 llvm::DenseSet<Operation*>& processedOps, Value stream) {
//   // 调用新的stream pool版本
//   processCuLibsNodeWithStreamPool(node, builder, mapper, processedOps, stream);
// }

// // Helper function to collect dependent operations
// void collectDependentOps(Operation* op, llvm::SetVector<Operation*>& requiredOps, 
//                         const llvm::DenseSet<Operation*>& processedOps) {
//   // 遍历操作的所有操作数
//   for (Value operand : op->getOperands()) {
//     if (Operation* definingOp = operand.getDefiningOp()) {
//       // 如果定义操作还没有被处理，且不是Block参数
//       if (!processedOps.count(definingOp) && !isa<BlockArgument>(operand)) {
//         // 检查是否是我们需要移动的操作类型
//         if (shouldMoveWithCuLibs(definingOp)) {
//           // 递归收集这个操作的依赖
//           collectDependentOps(definingOp, requiredOps, processedOps);
//           // 添加到必需操作列表
//           requiredOps.insert(definingOp);
//         }
//       }
//     }
//   }
// }

// // Helper function to determine if an operation should be moved with CuLibs calls
// bool shouldMoveWithCuLibs(Operation* op) {
//   // 这些操作通常是为CuLibs调用准备参数的
//   return isa<memref::ExtractAlignedPointerAsIndexOp>(op) ||
//          isa<arith::IndexCastOp>(op) ||
//          isa<mlir::LLVM::IntToPtrOp>(op) ||
//          isa<memref::AllocOp>(op) ||
//          isa<arith::ConstantOp>(op);  // 常量也可能需要
// }

// // Single combined module version
// void reorganizeGPUModules(ModuleOp moduleOp, DependencyGraph &graph) {
//   OpBuilder builder(moduleOp.getContext());
  
//   // Scan all existing modules and functions
//   llvm::SmallVector<gpu::GPUModuleOp, 4> allModules;
//   std::map<std::pair<std::string, std::string>, gpu::GPUFuncOp> funcMap;
  
//   moduleOp.walk([&](gpu::GPUModuleOp op) {
//     allModules.push_back(op);
    
//     // Collect all functions in this module
//     std::string moduleName = op.getName().str();
//     op.walk([&](gpu::GPUFuncOp funcOp) {
//       std::string funcName = funcOp.getName().str();
//       funcMap[{moduleName, funcName}] = funcOp;
//     });
//   });
  
//   // Exit if no modules
//   if (allModules.empty())
//     return;
  
//   // Collect all kernel nodes from the dependency graph
//   llvm::SmallVector<DependencyNode*, 16> allKernels;
//   for (const auto &nodePair : graph.nodes) {
//     DependencyNode* node = nodePair.get();
//     if (node->type == NodeType::Kernel) {
//       allKernels.push_back(node);
//     }
//   }
  
//   // Exit if no kernels to process
//   if (allKernels.empty()) {
//     // Still need to clean up old modules even if no kernels
//     for (auto moduleOp : allModules) {
//       moduleOp.erase();
//     }
//     return;
//   }
  
//   // Create renaming map: <old module name, old function name> -> <new module name, new function name>
//   using ModuleFuncKey = std::pair<std::string, std::string>;
//   std::map<ModuleFuncKey, ModuleFuncKey> renameMap;
  
//   // Step 1: Create a single combined module for all kernels
//   std::string combinedModuleName = "combined_kernels_module";
//   builder.setInsertionPointToStart(moduleOp.getBody());
  
//   auto combinedModule = builder.create<gpu::GPUModuleOp>(
//       moduleOp.getLoc(),
//       builder.getStringAttr(combinedModuleName));
  
//   builder.setInsertionPointToStart(combinedModule.getBody());
  
//   // Function counter to ensure uniqueness across all kernels
//   int globalFuncCounter = 0;
  
//   // Step 2: Copy all kernel functions to the combined module
//   for (DependencyNode* kernel : allKernels) {
//     std::string oldModuleName = kernel->kernelModuleName.str();
//     std::string oldFuncName = kernel->kernelName.str();
    
//     // Find the original function
//     auto funcKey = std::make_pair(oldModuleName, oldFuncName);
//     auto funcIt = funcMap.find(funcKey);
    
//     if (funcIt != funcMap.end()) {
//       // Create a new unique function name
//       // Include level information in the name for easier identification
//       std::string newFuncName = "kernel_L" + std::to_string(kernel->topologicalLevel) + 
//                                 "_" + std::to_string(globalFuncCounter++);
      
//       // Clone the function to the combined module
//       auto clonedFunc = cast<gpu::GPUFuncOp>(builder.clone(*funcIt->second));
      
//       // Set the new function name
//       clonedFunc.setName(newFuncName);
      
//       // Save mapping relationship
//       renameMap[funcKey] = {combinedModuleName, newFuncName};
//     }
//   }
  
//   // Step 3: Update all kernel launch references to point to the combined module
//   moduleOp.walk([&](gpu::LaunchFuncOp op) {
//     std::string oldModuleName = op.getKernelModuleName().str();
//     std::string oldFuncName = op.getKernelName().str();
    
//     auto funcKey = std::make_pair(oldModuleName, oldFuncName);
//     auto renameIt = renameMap.find(funcKey);
    
//     if (renameIt != renameMap.end()) {
//       std::string newModuleName = renameIt->second.first;
//       std::string newFuncName = renameIt->second.second;
      
//       // Create new symbol reference pointing to the combined module
//       auto newKernel = SymbolRefAttr::get(
//           builder.getContext(),
//           StringAttr::get(builder.getContext(), newModuleName),
//           {SymbolRefAttr::get(builder.getContext(), newFuncName)});
      
//       // Update kernel reference
//       op->setAttr("kernel", newKernel);
//     }
//   });
  
//   // Step 4: Delete all old modules
//   for (auto moduleOp : allModules) {
//     moduleOp.erase();
//   }
// }

// } // namespace onnx_mlir