// IrReorganization.cpp 修改
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

// 修复版的 reorganizeIR 函数
void reorganizeIR(func::FuncOp funcOp, DependencyGraph &graph) {
  OpBuilder builder(funcOp.getContext());
  
  // 创建映射来跟踪操作映射关系
  IRMapping mapper;
  
  // 按拓扑级别对节点分组
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
  
  // 第一阶段：先复制非图节点的前置操作，但跳过gpu.wait操作
  for (auto &op : oldBlock->getOperations()) {
    if (graph.opToNodeMap.count(&op)) {
      // 遇到图中节点，停止复制前置操作
      break;
    }
    
    // 跳过所有GPU等待操作，因为我们会按照依赖图添加必要的等待点
    if (isa<gpu::WaitOp>(op)) {
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
  
  // 用于跟踪最后一级别的令牌
  llvm::SmallVector<Value, 8> finalLevelTokens;
  
  // 第二阶段：按拓扑级别处理节点
  for (unsigned level = 1; level <= maxLevel; level++) {
    auto levelIt = nodesByLevel.find(level);
    if (levelIt == nodesByLevel.end() || levelIt->second.empty())
      continue;
      
    auto &nodesAtLevel = levelIt->second;
    
    // 收集本级别的异步令牌
    llvm::SmallVector<Value, 8> levelTokens;
    
    // 处理当前级别的所有节点
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
        
        // 映射grid和block尺寸
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
        
        // 创建异步等待操作
        auto waitOp = builder.create<gpu::WaitOp>(
            funcOp.getLoc(),
            builder.getType<gpu::AsyncTokenType>(),
            ValueRange{});
            
        // 直接创建异步内核启动
        auto newLaunchOp = builder.create<gpu::LaunchFuncOp>(
            kernelOp.getLoc(),
            kernelSymbol,
            mappedGridSize,
            mappedBlockSize,
            Value(),  // 没有动态共享内存
            remappedOperands,
            builder.getType<gpu::AsyncTokenType>(),  // 异步令牌类型
            ValueRange{waitOp.getAsyncToken()},  // 使用前面创建的异步等待令牌
            std::nullopt);  // 没有集群大小
            
        // 收集本级别的异步令牌
        levelTokens.push_back(newLaunchOp.getAsyncToken());
            
        // 映射结果
        if (kernelOp->getNumResults() > 0) {
          mapper.map(kernelOp->getResult(0), newLaunchOp->getResult(0));
        }
        
        // 标记为已处理
        processedOps.insert(node->op);
      } 
      else if (node->type == NodeType::Loop) {
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
    
    // 如果当前级别有操作
    if (!levelTokens.empty()) {
      // 如果不是最后一级，添加一个同步点
      if (level < maxLevel) {
        builder.setInsertionPointToEnd(newBlock);
        
        // 非异步等待 - 确保本级别所有操作完成后才进入下一级别
        builder.create<gpu::WaitOp>(funcOp.getLoc(), TypeRange{}, levelTokens);
      } 
      // 如果是最后一级，保存令牌以便在函数返回前添加最终同步点
      else {
        finalLevelTokens = levelTokens;
      }
    }
  }
  
  // 第三阶段：复制剩余未处理的操作，但跳过gpu.wait操作
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
  
  // 如果有最终级别的令牌，添加最终同步点
  if (!finalLevelTokens.empty()) {
    builder.setInsertionPointToEnd(newBlock);
    
    // 添加最终的同步等待
    builder.create<gpu::WaitOp>(funcOp.getLoc(), TypeRange{}, finalLevelTokens);
  }
  
  // 如果有返回操作，现在克隆它
  if (hasReturnOp) {
    builder.setInsertionPointToEnd(newBlock);
    Operation *newReturnOp = returnOp->clone(mapper);
    newBlock->push_back(newReturnOp);
  }
  
  // 替换旧块
  // 1. 添加新块到函数体
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
  
  // 3. 移除旧块
  oldBlock->dropAllUses();
  oldBlock->erase();
}

// 修复版的 reorganizeGPUModules 函数
// void reorganizeGPUModules(ModuleOp moduleOp, DependencyGraph &graph) {
//   OpBuilder builder(moduleOp.getContext());
  
//   // 按拓扑级别分组模块
//   std::map<unsigned, llvm::SmallVector<StringRef, 8>> modulesByLevel;
//   std::set<std::pair<unsigned, StringRef>> processedModules;
  
//   for (const auto &nodePair : graph.nodes) {
//     DependencyNode* node = nodePair.get();
//     if (node->type == NodeType::Kernel) {
//       auto level = node->topologicalLevel;
//       auto moduleName = node->kernelModuleName;
      
//       // 每个级别中每个模块只添加一次
//       auto key = std::make_pair(level, moduleName);
//       if (processedModules.insert(key).second) {
//         modulesByLevel[level].push_back(moduleName);
//       }
//     }
//   }
  
//   // 记录原始模块和函数名，用于后续更新
//   struct ModuleInfo {
//     llvm::SmallVector<Operation*, 4> toRemove;
//     llvm::SmallVector<std::pair<gpu::GPUFuncOp, std::string>, 4> funcRenameMap;
//   };
  
//   // 为每个级别创建组合模块
//   for (const auto &levelPair : modulesByLevel) {
//     unsigned level = levelPair.first;
//     const auto &modules = levelPair.second;
    
//     // 如果该级别只有一个模块则跳过
//     if (modules.size() <= 1)
//       continue;
      
//     // 创建新的组合模块
//     std::string combinedName = "main_graph_kernel_level_" + std::to_string(level);
//     builder.setInsertionPointToStart(moduleOp.getBody());
    
//     auto combinedModule = builder.create<gpu::GPUModuleOp>(
//         moduleOp.getLoc(), 
//         builder.getStringAttr(combinedName));
    
//     // 添加该级别所有模块中的内核函数到组合模块
//     builder.setInsertionPointToStart(combinedModule.getBody());
    
//     ModuleInfo info;
//     int funcCounter = 0;  // 简单计数器
    
//     // 第一阶段：克隆所有函数并收集重命名信息
//     for (auto moduleName : modules) {
//       bool found = false;
      
//       // 查找模块
//       moduleOp.walk([&](gpu::GPUModuleOp op) {
//         if (op.getName() == moduleName) {
//           found = true;
          
//           // 遍历模块中的所有函数
//           for (Operation &op : op.getBody()->getOperations()) {
//             if (auto funcOp = dyn_cast<gpu::GPUFuncOp>(op)) {
//               // 为函数创建新名称
//               std::string newFuncName = "func_" + std::to_string(funcCounter++);
              
//               // 克隆并重命名函数
//               auto clonedFunc = cast<gpu::GPUFuncOp>(builder.clone(op));
//               clonedFunc.setName(newFuncName);
              
//               // 记录重命名信息
//               info.funcRenameMap.push_back({cast<gpu::GPUFuncOp>(op), newFuncName});
//             }
//           }
          
//           // 标记模块准备删除
//           info.toRemove.push_back(op);
//         }
//       });
//     }
    
//     // 第二阶段：更新所有内核启动
//     moduleOp.walk([&](gpu::LaunchFuncOp op) {
//       StringRef opModuleName = op.getKernelModuleName();
//       StringRef opKernelName = op.getKernelName();
      
//       // 检查此启动是否使用当前级别的任何模块
//       for (auto moduleName : modules) {
//         if (opModuleName == moduleName) {
//           // 查找对应的重命名信息
//           for (auto &pair : info.funcRenameMap) {
//             gpu::GPUFuncOp origFunc = pair.first;
//             std::string newFuncName = pair.second;
            
//             if (origFunc.getName() == opKernelName) {
//               // 创建新的符号引用
//               auto newKernel = SymbolRefAttr::get(
//                   builder.getContext(),
//                   StringAttr::get(builder.getContext(), combinedName),
//                   {SymbolRefAttr::get(builder.getContext(), newFuncName)});
              
//               // 更新kernel属性
//               op->setAttr("kernel", newKernel);
//               break;
//             }
//           }
//           break;
//         }
//       }
//     });
    
//     // 第三阶段：删除原始模块
//     for (Operation *op : info.toRemove) {
//       op->erase();
//     }
//   }
// }

void reorganizeGPUModules(ModuleOp moduleOp, DependencyGraph &graph) {
  OpBuilder builder(moduleOp.getContext());
  
  // 直接计数器 - 确保生成唯一名称
  int moduleCounter = 0;
  int funcCounter = 0;
  
  // 扫描一次并获取所有模块
  llvm::SmallVector<gpu::GPUModuleOp, 4> allModules;
  moduleOp.walk([&](gpu::GPUModuleOp op) {
    allModules.push_back(op);
  });
  
  // 没有模块则退出
  if (allModules.empty())
    return;
  
  // 创建一个新的合并模块
  std::string combinedName = "merged_module_" + std::to_string(moduleCounter++);
  builder.setInsertionPointToStart(moduleOp.getBody());
  
  auto combinedModule = builder.create<gpu::GPUModuleOp>(
      moduleOp.getLoc(),
      builder.getStringAttr(combinedName));
  
  builder.setInsertionPointToStart(combinedModule.getBody());
  
  // 创建映射: <旧模块名, 旧函数名> -> 新函数名
  std::map<std::pair<std::string, std::string>, std::string> renameMap;
  
  // 第一步：复制所有函数并重命名
  for (auto moduleOp : allModules) {
    std::string oldModuleName = moduleOp.getName().str();
    
    for (Operation &op : moduleOp.getBody()->getOperations()) {
      if (auto funcOp = dyn_cast<gpu::GPUFuncOp>(op)) {
        std::string oldFuncName = funcOp.getName().str();
        
        // 创建新函数名
        std::string newFuncName = "kernel_" + std::to_string(funcCounter++);
        
        // 克隆并重命名函数
        auto clonedFunc = cast<gpu::GPUFuncOp>(builder.clone(op));
        clonedFunc.setName(newFuncName);
        
        // 保存重命名映射
        renameMap[{oldModuleName, oldFuncName}] = newFuncName;
      }
    }
  }
  
  // 第二步：更新所有kernel启动引用
  moduleOp.walk([&](gpu::LaunchFuncOp op) {
    std::string oldModuleName = op.getKernelModuleName().str();
    std::string oldFuncName = op.getKernelName().str();
    
    auto it = renameMap.find({oldModuleName, oldFuncName});
    if (it != renameMap.end()) {
      std::string newFuncName = it->second;
      
      // 创建新符号引用
      auto newKernel = SymbolRefAttr::get(
          builder.getContext(),
          StringAttr::get(builder.getContext(), combinedName),
          {SymbolRefAttr::get(builder.getContext(), newFuncName)});
      
      // 更新属性
      op->setAttr("kernel", newKernel);
    }
  });
  
  // 第三步：删除旧模块
  for (auto moduleOp : allModules) {
    moduleOp.erase();
  }
}

} // namespace onnx_mlir