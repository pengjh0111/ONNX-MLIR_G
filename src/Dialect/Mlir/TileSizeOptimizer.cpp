#include "TileSizeOptimizer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"

#define DEBUG_TYPE "tile-size-optimizer"

using namespace mlir;
using namespace mlir::scf;
using namespace llvm;

//===----------------------------------------------------------------------===//
// TileSizeOptimizer实现
//===----------------------------------------------------------------------===//

// CompleteTileConfig TileSizeOptimizer::optimizeTileSize(scf::ParallelOp parallelOp) {
//   // 步骤1: 提取循环信息
//   std::vector<LoopInfo> loopInfos = extractLoopInfo(parallelOp);
  
//   // 步骤2: 分析内存访问模式
//   std::vector<MemoryAccessInfo> memAccesses = analyzeMemoryAccesses(parallelOp);
  
//   // 步骤3: 分析计算操作
//   std::vector<ComputationInfo> computations = analyzeComputations(parallelOp);
  
//   // 步骤4: 使用启发式规则生成候选tile大小
//   std::vector<std::vector<int64_t>> tileSizeCandidates = generateTileSizeCandidates(loopInfos);
  
//   // 步骤5: 使用动态规划查找最优tile配置
//   CompleteTileConfig optimalConfig = findOptimalTileConfig(
//       loopInfos, memAccesses, computations, tileSizeCandidates);
  
//   return optimalConfig;
// }

CompleteTileConfig TileSizeOptimizer::optimizeTileSize(scf::ParallelOp parallelOp) {
  // Step 1: Extract loop information
  std::vector<LoopInfo> loopInfos = extractLoopInfo(parallelOp);
  
  // Step 2: Analyze memory access patterns
  std::vector<MemoryAccessInfo> memAccesses = analyzeMemoryAccesses(parallelOp);
  
  // Step 3: Analyze computation operations
  std::vector<ComputationInfo> computations = analyzeComputations(parallelOp);
  
  // Step 4: Detect computational pattern
  ComputationalPattern pattern = detectComputationalPattern(parallelOp, memAccesses);
  
  // Step 5: Generate candidate tile sizes with pattern awareness
  std::vector<std::vector<int64_t>> tileSizeCandidates = 
      generateTileSizeCandidates(loopInfos, pattern);
  
  // Step 6: Find optimal tile configuration using dynamic programming
  CompleteTileConfig optimalConfig = findOptimalTileConfig(
      loopInfos, memAccesses, computations, tileSizeCandidates);
  
  LLVM_DEBUG(llvm::dbgs() << "Optimal tile config: ";
           for (const auto &dim : optimalConfig.perDimConfig) {
             llvm::dbgs() << dim.tileSize << " ";
           }
           llvm::dbgs() << "Score: " << optimalConfig.overallPerformanceScore << "\n");
  
  return optimalConfig;
}

std::vector<LoopInfo> TileSizeOptimizer::extractLoopInfo(scf::ParallelOp parallelOp) {
  std::vector<LoopInfo> loopInfos;
  
  // 提取每个循环维度的信息
  for (unsigned i = 0; i < parallelOp.getNumLoops(); ++i) {
    LoopInfo info;
    info.dimension = i;
    info.lowerBound = parallelOp.getLowerBound()[i];
    info.upperBound = parallelOp.getUpperBound()[i];
    info.step = parallelOp.getStep()[i];
    
    // 尝试提取常量值（如果可用）
    if (auto constLB = dyn_cast_or_null<arith::ConstantIndexOp>(info.lowerBound.getDefiningOp())) {
      info.constantLowerBound = constLB.value();
      
      if (auto constUB = dyn_cast_or_null<arith::ConstantIndexOp>(info.upperBound.getDefiningOp())) {
        info.constantUpperBound = constUB.value();
        
        if (auto constStep = dyn_cast_or_null<arith::ConstantIndexOp>(info.step.getDefiningOp())) {
          info.constantStep = constStep.value();
          info.hasConstantBounds = true;
          
          // 计算总迭代次数
          info.tripCount = (info.constantUpperBound - info.constantLowerBound + info.constantStep - 1) / 
                           info.constantStep;
        }
      }
    }
    
    loopInfos.push_back(info);
  }
  
  return loopInfos;
}

std::vector<MemoryAccessInfo> TileSizeOptimizer::analyzeMemoryAccesses(scf::ParallelOp parallelOp) {
  std::vector<MemoryAccessInfo> memAccesses;
  
  // 收集循环中的所有内存操作
  parallelOp.getBody()->walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      MemoryAccessInfo info;
      info.memref = loadOp.getMemref();
      info.indices = SmallVector<Value, 4>(loadOp.getIndices().begin(), loadOp.getIndices().end());
      info.isLoad = true;
      info.isStore = false;
      
      // 确定数据类型大小
      if (auto memrefType = loadOp.getMemref().getType().dyn_cast<MemRefType>()) {
        Type elementType = memrefType.getElementType();
        if (elementType.isF32()) {
          info.dataTypeSizeInBytes = 4;
        } else if (elementType.isF64()) {
          info.dataTypeSizeInBytes = 8;
        } else if (elementType.isInteger(32)) {
          info.dataTypeSizeInBytes = 4;
        } else if (elementType.isInteger(64)) {
          info.dataTypeSizeInBytes = 8;
        } else {
          // 其他类型的默认值
          info.dataTypeSizeInBytes = 4;
        }
      } else {
        info.dataTypeSizeInBytes = 4;  // 默认值
      }
      
      // 分析访问模式（简化版）
      info.pattern = MemoryAccessPattern::SEQUENTIAL;
      
      // 寻找合并访问模式
      // 检查是否最内层循环索引用于最右侧维度
      if (!info.indices.empty() && 
          info.indices.back() == parallelOp.getInductionVars().back()) {
        info.pattern = MemoryAccessPattern::COALESCED;
      }
      
      memAccesses.push_back(info);
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      MemoryAccessInfo info;
      info.memref = storeOp.getMemref();
      info.indices = SmallVector<Value, 4>(storeOp.getIndices().begin(), storeOp.getIndices().end());
      info.isLoad = false;
      info.isStore = true;
      
      // 确定数据类型大小（类似于loadOp）
      if (auto memrefType = storeOp.getMemref().getType().dyn_cast<MemRefType>()) {
        Type elementType = memrefType.getElementType();
        if (elementType.isF32()) {
          info.dataTypeSizeInBytes = 4;
        } else if (elementType.isF64()) {
          info.dataTypeSizeInBytes = 8;
        } else if (elementType.isInteger(32)) {
          info.dataTypeSizeInBytes = 4;
        } else if (elementType.isInteger(64)) {
          info.dataTypeSizeInBytes = 8;
        } else {
          info.dataTypeSizeInBytes = 4;  // 默认值
        }
      } else {
        info.dataTypeSizeInBytes = 4;  // 默认值
      }
      
      // 分析访问模式（简化版）
      info.pattern = MemoryAccessPattern::SEQUENTIAL;
      
      // 检查合并访问模式
      if (!info.indices.empty() && 
          info.indices.back() == parallelOp.getInductionVars().back()) {
        info.pattern = MemoryAccessPattern::COALESCED;
      }
      
      memAccesses.push_back(info);
    }
  });
  
  return memAccesses;
}

std::vector<ComputationInfo> TileSizeOptimizer::analyzeComputations(scf::ParallelOp parallelOp) {
  std::vector<ComputationInfo> computations;
  
  // 收集循环中的计算操作
  parallelOp.getBody()->walk([&](Operation *op) {
    ComputationInfo info;
    info.op = op;
    info.opCount = 1;  // 每个操作的默认计数
    
    // 识别浮点操作
    if (isa<arith::AddFOp>(op) || isa<arith::SubFOp>(op) ||
        isa<arith::MulFOp>(op) || isa<arith::DivFOp>(op)) {
      info.isFloatingPoint = true;
      computations.push_back(info);
    } 
    // 识别整数操作
    else if (isa<arith::AddIOp>(op) || isa<arith::SubIOp>(op) ||
             isa<arith::MulIOp>(op) || isa<arith::DivSIOp>(op) ||
             isa<arith::DivUIOp>(op)) {
      info.isFloatingPoint = false;
      computations.push_back(info);
    }
  });
  
  return computations;
}

// std::vector<std::vector<int64_t>> TileSizeOptimizer::generateTileSizeCandidates(
//     const std::vector<LoopInfo> &loopInfos) {
  
//   std::vector<std::vector<int64_t>> candidatesPerDim;
  
//   for (const auto &loopInfo : loopInfos) {
//     std::vector<int64_t> candidatesForDim;
    
//     // 如果我们有常量边界，使用它们生成候选值
//     if (loopInfo.hasConstantBounds) {
//       int64_t loopRange = loopInfo.constantUpperBound - loopInfo.constantLowerBound;
      
//       // 添加2的幂次方候选值
//       for (int64_t size = 4; size <= 1024; size *= 2) {
//         // 跳过大于循环范围的尺寸
//         if (size > loopRange) {
//           break;
//         }
//         candidatesForDim.push_back(size);
//       }
      
//       // 添加循环范围的除数
//       for (int64_t i = 1; i * i <= loopRange; ++i) {
//         if (loopRange % i == 0) {
//           if (i >= 4 && i <= 1024) {
//             candidatesForDim.push_back(i);
//           }
          
//           int64_t j = loopRange / i;
//           if (j != i && j >= 4 && j <= 1024) {
//             candidatesForDim.push_back(j);
//           }
//         }
//       }
      
//       // 对候选值进行排序
//       std::sort(candidatesForDim.begin(), candidatesForDim.end());
      
//       // 移除重复项
//       candidatesForDim.erase(
//           std::unique(candidatesForDim.begin(), candidatesForDim.end()),
//           candidatesForDim.end());
//     } else {
//       // 对于没有常量边界的情况，使用标准的2的幂次方尺寸
//       candidatesForDim = {8, 16, 32, 64, 128, 256, 512};
//     }
    
//     // 过滤掉无效的tile尺寸
//     std::vector<int64_t> validCandidates;
//     for (int64_t size : candidatesForDim) {
//       // 检查硬件约束
//       if (size <= hwParams.maxThreadsPerBlock) {
//         validCandidates.push_back(size);
//       }
//     }
    
//     // 如果没有有效候选值，添加一个默认值
//     if (validCandidates.empty()) {
//       validCandidates.push_back(32);  // 默认使用warp大小
//     }
    
//     candidatesPerDim.push_back(validCandidates);
//   }
  
//   return candidatesPerDim;
// }

// std::vector<std::vector<int64_t>> TileSizeOptimizer::generateTileSizeCandidates(
//     const std::vector<LoopInfo> &loopInfos) {
  
//   std::vector<std::vector<int64_t>> candidatesPerDim;
  
//   for (const auto &loopInfo : loopInfos) {
//     std::vector<int64_t> candidatesForDim;
    
//     // 如果我们有常量边界，使用它们生成候选值
//     if (loopInfo.hasConstantBounds) {
//       int64_t loopRange = loopInfo.constantUpperBound - loopInfo.constantLowerBound;
      
//       // 小维度特殊处理：直接使用维度大小作为首选候选值
//       if (loopRange <= 16) {
//         candidatesForDim.push_back(loopRange);
//       }
      
//       // 对于较小维度，考虑更细粒度的候选值
//       if (loopRange < 32) {
//         for (int64_t size = 1; size <= loopRange; ++size) {
//           candidatesForDim.push_back(size);
//         }
//       } else {
//         // 添加更多细粒度值
//         for (int64_t size = 1; size <= std::min(loopRange, (int64_t)16); ++size) {
//           candidatesForDim.push_back(size);
//         }
//       }
      
//       // 添加2的幂次方候选值
//       for (int64_t size = 2; size <= loopRange && size <= 1024; size *= 2) {
//         candidatesForDim.push_back(size);
//       }
      
//       // 添加warp相关优化的值
//       for (int mul = 1; mul <= 32; mul++) {
//         int64_t size = hwParams.warpSize * mul;
//         if (size <= loopRange && size <= 1024) {
//           candidatesForDim.push_back(size);
//         } else {
//           break;
//         }
//       }
      
//       // 添加循环范围的除数（可以整除的值通常是好的tile size）
//       for (int64_t i = 1; i * i <= loopRange; ++i) {
//         if (loopRange % i == 0) {
//           candidatesForDim.push_back(i);
          
//           int64_t j = loopRange / i;
//           if (j != i) {
//             candidatesForDim.push_back(j);
//           }
//         }
//       }
      
//       // 添加一些常见的CNN kernel大小相关的值
//       std::vector<int64_t> commonCNNSizes = {3, 5, 7, 9, 11};
//       for (auto size : commonCNNSizes) {
//         if (size <= loopRange) {
//           candidatesForDim.push_back(size);
//         }
//       }
      
//       // 对候选值进行排序
//       std::sort(candidatesForDim.begin(), candidatesForDim.end());
      
//       // 移除重复项
//       candidatesForDim.erase(
//           std::unique(candidatesForDim.begin(), candidatesForDim.end()),
//           candidatesForDim.end());
          
//     } else {
//       // 对于没有常量边界的情况，使用更全面的候选集
//       candidatesForDim = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024};
//     }
    
//     // 过滤掉无效的tile尺寸，确保不超过循环范围和硬件约束
//     std::vector<int64_t> validCandidates;
//     for (int64_t size : candidatesForDim) {
//       if (size <= hwParams.maxThreadsPerBlock && 
//           (!loopInfo.hasConstantBounds || size <= loopInfo.constantUpperBound - loopInfo.constantLowerBound)) {
//         validCandidates.push_back(size);
//       }
//     }
    
//     // 如果没有有效候选值，添加一个默认值
//     if (validCandidates.empty()) {
//       // 对于小维度，使用维度大小作为默认值
//       if (loopInfo.hasConstantBounds && 
//           loopInfo.constantUpperBound - loopInfo.constantLowerBound < hwParams.warpSize) {
//         validCandidates.push_back(loopInfo.constantUpperBound - loopInfo.constantLowerBound);
//       } else {
//         validCandidates.push_back(std::min(hwParams.warpSize, (int)32));  // 默认使用warp大小
//       }
//     }
    
//     candidatesPerDim.push_back(validCandidates);
//   }
  
//   return candidatesPerDim;
// }

TileSizeOptimizer::ComputationalPattern TileSizeOptimizer::detectComputationalPattern(
    scf::ParallelOp parallelOp,
    const std::vector<MemoryAccessInfo> &memAccesses) {
  
  // Default pattern
  ComputationalPattern pattern = ComputationalPattern::GENERIC;
  
  // Get loop dimensions
  size_t numDims = parallelOp.getNumLoops();
  
  // Count memory operations
  int loadCount = 0;
  int storeCount = 0;
  llvm::DenseSet<Value> uniqueLoadMemrefs;
  llvm::DenseSet<Value> uniqueStoreMemrefs;
  
  for (const auto &access : memAccesses) {
    if (access.isLoad) {
      loadCount++;
      uniqueLoadMemrefs.insert(access.memref);
    }
    if (access.isStore) {
      storeCount++;
      uniqueStoreMemrefs.insert(access.memref);
    }
  }
  
  // Count arithmetic operations
  int addCount = 0, mulCount = 0, maxCount = 0;
  
  parallelOp.getBody()->walk([&](Operation *op) {
    if (isa<arith::AddFOp>(op) || isa<arith::AddIOp>(op)) {
      addCount++;
    } else if (isa<arith::MulFOp>(op) || isa<arith::MulIOp>(op)) {
      mulCount++;
    } else if (isa<arith::MaximumFOp>(op) || isa<arith::MaxSIOp>(op) || 
        isa<arith::MaxNumFOp>(op)) {
      maxCount++;
    }
  });
  
  // Matrix multiplication pattern detection
  // - Usually 3 dimensions (M, N, K)
  // - High ratio of multiply-adds
  // - Specific memory access patterns
  if (numDims >= 2 && 
      mulCount > 0 && 
      addCount > 0 && 
      mulCount + addCount > loadCount) {
    pattern = ComputationalPattern::MATMUL;
  }
  
  // Convolution pattern detection
  // - Usually have window/kernel dimensions
  // - Often use max operations (for pooling)
  // - High spatial locality
  if (numDims >= 3 && 
      (mulCount > 0 || maxCount > 0) && 
      uniqueLoadMemrefs.size() >= 1 && 
      uniqueStoreMemrefs.size() >= 1) {
    pattern = ComputationalPattern::CONV;
  }
  
  // Stencil pattern detection
  // - Multiple loads from adjacent locations
  // - Usually few arithmetic ops per load
  if (loadCount > 3 * (addCount + mulCount) && 
      uniqueLoadMemrefs.size() <= 2 &&
      uniqueStoreMemrefs.size() <= 2) {
    pattern = ComputationalPattern::STENCIL;
  }
  
  // Reduction pattern detection
  // - Usually has scf.reduce operation
  // - High ratio of adds or maxes
  if (numDims >= 1 && 
      (addCount > loadCount / 2 || maxCount > loadCount / 2)) {
    
    // Check for reduce operations
    bool hasReduce = false;
    parallelOp.getBody()->walk([&](Operation *op) {
      if (isa<scf::ReduceOp>(op)) {
        hasReduce = true;
      }
    });
    
    if (hasReduce) {
      pattern = ComputationalPattern::REDUCTION;
    }
  }
  
  // Element-wise pattern detection
  // - One load, one store per iteration
  // - Simple operations (add, mul, etc.)
  if (loadCount <= 2 * numDims && 
      storeCount <= numDims && 
      addCount + mulCount <= 3 * numDims) {
    pattern = ComputationalPattern::ELEMENTWISE;
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Detected computational pattern: " 
            << static_cast<int>(pattern) << "\n");
  
  return pattern;
}

std::vector<std::vector<int64_t>> TileSizeOptimizer::generateTileSizeCandidates(
    const std::vector<LoopInfo> &loopInfos,
    ComputationalPattern pattern) {
  
  std::vector<std::vector<int64_t>> candidatesPerDim;
  
  // Analyze memory access patterns to determine which dimensions are most important
  // This would ideally be done by analyzing memAccesses, but for now we'll use
  // heuristics based on loop position
  
  // For CUDA/GPU optimization, innermost dimensions should be prioritized for coalescing
  
  for (int dimIdx = 0; dimIdx < loopInfos.size(); ++dimIdx) {
    const auto &loopInfo = loopInfos[dimIdx];
    bool isInnermostDim = (dimIdx == loopInfos.size() - 1);
    bool isOutermostDim = (dimIdx == 0);
    std::vector<int64_t> candidatesForDim;
    
    // Basic set to include in all cases - powers of two are generally good
    for (int64_t size = 1; size <= 1024; size *= 2) {
      candidatesForDim.push_back(size);
    }
    
    // Special handling for different dimensions
    if (loopInfo.hasConstantBounds) {
      int64_t loopRange = loopInfo.constantUpperBound - loopInfo.constantLowerBound;
      
      // If dimension size is small, include it as a candidate
      if (loopRange <= 64) {
        candidatesForDim.push_back(loopRange);
      }
      
      // For small dimensions, consider all divisors
      if (loopRange <= 32) {
        for (int64_t i = 1; i <= loopRange; ++i) {
          if (loopRange % i == 0) {
            candidatesForDim.push_back(i);
          }
        }
      }
      
      // For larger dimensions, include key divisors
      else {
        for (int64_t i = 1; i * i <= loopRange; ++i) {
          if (loopRange % i == 0) {
            candidatesForDim.push_back(i);
            if (i != loopRange / i) {
              candidatesForDim.push_back(loopRange / i);
            }
          }
        }
      }
      
      // Hardware-specific tile size candidates
      
      // For innermost dimension, prioritize coalescing - multiples of warp size
      if (isInnermostDim) {
        for (int i = 1; i <= 4; ++i) {
          int64_t size = hwParams.warpSize * i;
          if (size <= loopRange && size <= hwParams.maxBlockDimX) {
            candidatesForDim.push_back(size);
          }
        }
        
        // For innermost dimension, also try half-warp sizes
        candidatesForDim.push_back(hwParams.warpSize / 2);
        
        // Special case: if innermost dimension is small, include values 
        // that are near multiples of warp size
        if (loopRange < hwParams.warpSize) {
          candidatesForDim.push_back(loopRange);
        } else if (loopRange < hwParams.warpSize * 2) {
          // For sizes between 32-64, try non-standard sizes like 48
          candidatesForDim.push_back(48);
        }
      }
      
      // For middle dimensions, balance between parallelism and data reuse
      if (!isInnermostDim && !isOutermostDim) {
        // Common values that often work well for middle dimensions
        std::vector<int64_t> midDimCandidates = {4, 8, 12, 16, 24, 32};
        for (auto size : midDimCandidates) {
          if (size <= loopRange) {
            candidatesForDim.push_back(size);
          }
        }
        
        // Special handling for convolutional patterns - add kernel-size related values
        std::vector<int64_t> convKernelSizes = {3, 5, 7, 9, 11};
        for (auto kernelSize : convKernelSizes) {
          if (kernelSize <= loopRange) {
            candidatesForDim.push_back(kernelSize);
            // Also add kernel size with halo regions for stencil/conv ops
            if (kernelSize + 2 <= loopRange) {
              candidatesForDim.push_back(kernelSize + 2);
            }
          }
        }
      }
      
      // For outermost dimension, prioritize load balancing across blocks
      if (isOutermostDim) {
        // Try to find values that evenly divide the dimension for load balancing
        for (int div = 2; div <= 16; ++div) {
          if (loopRange % div == 0) {
            int64_t size = loopRange / div;
            if (size <= hwParams.maxBlockDimX) {
              candidatesForDim.push_back(size);
            }
          }
        }
        
        // For small batch dimensions (often the outermost in ML workloads)
        // include typical batch sizes
        std::vector<int64_t> batchSizes = {1, 2, 4, 8, 16, 32, 64, 128};
        for (auto size : batchSizes) {
          if (size <= loopRange && size <= hwParams.maxBlockDimX) {
            candidatesForDim.push_back(size);
          }
        }
      }
      
      // For all dimensions, include values close to sqrt of the range
      // (often a good starting point for balanced tiling)
      int64_t sqrtRange = static_cast<int64_t>(std::sqrt(loopRange));
      candidatesForDim.push_back(sqrtRange);
      if (sqrtRange > 1) candidatesForDim.push_back(sqrtRange - 1);
      if (sqrtRange < loopRange) candidatesForDim.push_back(sqrtRange + 1);
      
      // For dimensions that could be part of matrix multiplication patterns,
      // include sizes that are good for matrix multiply (multiples of 8, 16, 32)
      std::vector<int64_t> matmulSizes = {8, 16, 32, 64};
      for (auto size : matmulSizes) {
        if (size <= loopRange) {
          candidatesForDim.push_back(size);
        }
      }
      
      // For very large dimensions, include some larger tile sizes
      if (loopRange > 1024) {
        std::vector<int64_t> largeSizes = {128, 256, 384, 512, 768, 1024};
        for (auto size : largeSizes) {
          if (size <= loopRange && size <= hwParams.maxBlockDimX) {
            candidatesForDim.push_back(size);
          }
        }
      }
    } else {
      // For non-constant loop bounds, use a comprehensive set of candidates
      candidatesForDim = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 
                         256, 384, 512, 768, 1024};
      
      // Special handling for different dimensions position
      if (isInnermostDim) {
        // For innermost, prioritize multiples of warp size
        candidatesForDim.push_back(hwParams.warpSize);
        candidatesForDim.push_back(hwParams.warpSize * 2);
        candidatesForDim.push_back(hwParams.warpSize * 4);
      }
    }
    
    // Sort and remove duplicates
    std::sort(candidatesForDim.begin(), candidatesForDim.end());
    candidatesForDim.erase(
        std::unique(candidatesForDim.begin(), candidatesForDim.end()),
        candidatesForDim.end());
    
    // Filter candidates based on hardware constraints
    std::vector<int64_t> validCandidates;
    for (int64_t size : candidatesForDim) {
      bool isValid = true;
      
      // Basic constraint: max threads per dimension
      if (dimIdx == 0 && size > hwParams.maxBlockDimX) isValid = false;
      else if (dimIdx == 1 && size > hwParams.maxBlockDimY) isValid = false;
      else if (dimIdx == 2 && size > hwParams.maxBlockDimZ) isValid = false;
      
      // If this is a constant-bound loop, enforce loop range constraint
      if (loopInfo.hasConstantBounds && 
          size > loopInfo.constantUpperBound - loopInfo.constantLowerBound) {
        isValid = false;
      }
      
      if (isValid) {
        validCandidates.push_back(size);
      }
    }
    
    // Edge case: if no valid candidates, add basic defaults
    if (validCandidates.empty()) {
      if (loopInfo.hasConstantBounds) {
        int64_t loopRange = loopInfo.constantUpperBound - loopInfo.constantLowerBound;
        // Use smallest of: loop range, warp size, or max dimension size
        int64_t defaultSize = std::min({loopRange, 
                                      static_cast<int64_t>(hwParams.warpSize),
                                      isOutermostDim ? 
                                        static_cast<int64_t>(hwParams.maxBlockDimX) : 
                                        static_cast<int64_t>(hwParams.maxThreadsPerBlock)});
        validCandidates.push_back(std::max(static_cast<int64_t>(1), defaultSize));
      } else {
        validCandidates.push_back(isInnermostDim ? hwParams.warpSize : 16);
      }
    }
    
    // Validate combined thread counts when building multi-dimensional candidates
    // This needs to be done in findOptimalTileConfig since we need to check
    // the product across dimensions
    
    candidatesPerDim.push_back(validCandidates);
    
    LLVM_DEBUG(llvm::dbgs() << "Dimension " << dimIdx 
              << " candidates (" << validCandidates.size() << "): ");
    LLVM_DEBUG(for (auto size : validCandidates) {
      llvm::dbgs() << size << " ";
    });
    LLVM_DEBUG(llvm::dbgs() << "\n");
  }
  
  // Pattern-specific adjustments
  switch (pattern) {
    case ComputationalPattern::MATMUL: {
      // For matrix multiplication, optimize for register blocking and shared memory
      if (loopInfos.size() >= 3) {
        // Common GEMM tile sizes for dimensions M, N
        std::vector<int64_t> matmulTileSizes = {16, 32, 64, 128};
        
        // Add these candidates for the first two dimensions (usually M, N)
        for (int i = 0; i < std::min(size_t(2), loopInfos.size()); ++i) {
          for (auto size : matmulTileSizes) {
            if (loopInfos[i].hasConstantBounds) {
              int64_t loopRange = loopInfos[i].constantUpperBound - loopInfos[i].constantLowerBound;
              if (size <= loopRange && size <= (i == 0 ? hwParams.maxBlockDimX : hwParams.maxBlockDimY)) {
                candidatesPerDim[i].push_back(size);
              }
            } else {
              candidatesPerDim[i].push_back(size);
            }
          }
          
          // Sort and remove duplicates
          std::sort(candidatesPerDim[i].begin(), candidatesPerDim[i].end());
          candidatesPerDim[i].erase(
              std::unique(candidatesPerDim[i].begin(), candidatesPerDim[i].end()),
              candidatesPerDim[i].end());
        }
        
        // For K dimension (reduction), use smaller values to control register pressure
        if (loopInfos.size() > 2) {
          int kDim = 2; // Typically the 3rd dimension in GEMM
          std::vector<int64_t> kDimSizes = {4, 8, 16};
          
          if (loopInfos[kDim].hasConstantBounds) {
            int64_t loopRange = loopInfos[kDim].constantUpperBound - loopInfos[kDim].constantLowerBound;
            for (auto size : kDimSizes) {
              if (size <= loopRange) {
                candidatesPerDim[kDim].push_back(size);
              }
            }
            
            // Sort and remove duplicates
            std::sort(candidatesPerDim[kDim].begin(), candidatesPerDim[kDim].end());
            candidatesPerDim[kDim].erase(
                std::unique(candidatesPerDim[kDim].begin(), candidatesPerDim[kDim].end()),
                candidatesPerDim[kDim].end());
          }
        }
      }
      break;
    }
    
    case ComputationalPattern::CONV: {
      // For convolutions, optimize spatial dimensions differently
      if (loopInfos.size() >= 4) {
        // Batch dimension (typically outermost)
        // This is often small and best tiled at size 1 or the exact batch size
        int batchDim = 0;
        if (loopInfos[batchDim].hasConstantBounds) {
          int64_t batchSize = loopInfos[batchDim].constantUpperBound - loopInfos[batchDim].constantLowerBound;
          if (batchSize <= 32) {
            // If batch size is small, consider using exact batch size
            candidatesPerDim[batchDim].push_back(batchSize);
          }
        }
        
        // Channel dimensions
        int channelDim = 1;
        if (loopInfos[channelDim].hasConstantBounds) {
          int64_t channels = loopInfos[channelDim].constantUpperBound - loopInfos[channelDim].constantLowerBound;
          // Common channel grouping sizes for channel dimensions
          std::vector<int64_t> channelSizes = {1, 3, 4, 8, 16};
          for (auto size : channelSizes) {
            if (size <= channels) {
              candidatesPerDim[channelDim].push_back(size);
            }
          }
        }
        
        // Spatial dimensions (typically innermost 2 dimensions)
        for (int i = std::max(2, static_cast<int>(loopInfos.size()) - 2); 
             i < loopInfos.size(); ++i) {
          if (loopInfos[i].hasConstantBounds) {
            int64_t spatialDim = loopInfos[i].constantUpperBound - loopInfos[i].constantLowerBound;
            
            // For spatial dimensions, include values that match common tensor core sizes
            std::vector<int64_t> spatialSizes = {4, 8, 16, 32};
            for (auto size : spatialSizes) {
              if (size <= spatialDim) {
                candidatesPerDim[i].push_back(size);
              }
            }
            
            // Add sizes based on common kernel dimensions plus padding
            std::vector<int64_t> kernelSizes = {3, 5, 7};
            for (auto kSize : kernelSizes) {
              for (int padding = 0; padding <= 2; padding++) {
                int64_t tileSize = kSize + 2 * padding;
                if (tileSize <= spatialDim) {
                  candidatesPerDim[i].push_back(tileSize);
                }
              }
            }
          }
          
          // Sort and remove duplicates
          std::sort(candidatesPerDim[i].begin(), candidatesPerDim[i].end());
          candidatesPerDim[i].erase(
              std::unique(candidatesPerDim[i].begin(), candidatesPerDim[i].end()),
              candidatesPerDim[i].end());
        }
      }
      break;
    }
    
    case ComputationalPattern::REDUCTION: {
      // For reductions, the reduction dimensions should use larger tiles
      if (!loopInfos.empty()) {
        // Assume last dimension is reduction dimension (common pattern)
        int redDim = loopInfos.size() - 1;
        
        if (loopInfos[redDim].hasConstantBounds) {
          int64_t redRange = loopInfos[redDim].constantUpperBound - loopInfos[redDim].constantLowerBound;
          
          // For reduction, try larger tiles to reduce synchronization overhead
          std::vector<int64_t> redSizes = {32, 64, 128, 256, 512};
          for (auto size : redSizes) {
            if (size <= redRange) {
              candidatesPerDim[redDim].push_back(size);
            }
          }
          
          // Also try full reduction in one tile
          candidatesPerDim[redDim].push_back(redRange);
        }
        
        // Sort and remove duplicates
        std::sort(candidatesPerDim[redDim].begin(), candidatesPerDim[redDim].end());
        candidatesPerDim[redDim].erase(
            std::unique(candidatesPerDim[redDim].begin(), candidatesPerDim[redDim].end()),
            candidatesPerDim[redDim].end());
      }
      break;
    }
    
    case ComputationalPattern::STENCIL: {
      // For stencils, consider halo regions
      for (int i = 0; i < loopInfos.size(); ++i) {
        if (loopInfos[i].hasConstantBounds) {
          // Add tile sizes that account for typical stencil radii (1, 2, 3)
          for (int radius = 1; radius <= 3; radius++) {
            // For a radius-R stencil, a good tile size is often a multiple of 16 or 32, plus 2*R
            for (int base : {16, 32}) {
              int64_t tileSize = base + 2 * radius;
              if (tileSize <= loopInfos[i].constantUpperBound - loopInfos[i].constantLowerBound) {
                candidatesPerDim[i].push_back(tileSize);
              }
            }
          }
          
          // Sort and remove duplicates
          std::sort(candidatesPerDim[i].begin(), candidatesPerDim[i].end());
          candidatesPerDim[i].erase(
              std::unique(candidatesPerDim[i].begin(), candidatesPerDim[i].end()),
              candidatesPerDim[i].end());
        }
      }
      break;
    }
    
    case ComputationalPattern::ELEMENTWISE: {
      // For element-wise, prioritize coalesced memory access for innermost dim
      if (!loopInfos.empty()) {
        int innerDim = loopInfos.size() - 1;
        
        // For innermost dimension, focus on warp and multiple-warp sizes
        for (int warps = 1; warps <= 8; warps *= 2) {
          int64_t size = hwParams.warpSize * warps;
          if (loopInfos[innerDim].hasConstantBounds) {
            int64_t dimSize = loopInfos[innerDim].constantUpperBound - loopInfos[innerDim].constantLowerBound;
            if (size <= dimSize) {
              candidatesPerDim[innerDim].push_back(size);
            }
          } else {
            candidatesPerDim[innerDim].push_back(size);
          }
        }
        
        // Sort and remove duplicates
        std::sort(candidatesPerDim[innerDim].begin(), candidatesPerDim[innerDim].end());
        candidatesPerDim[innerDim].erase(
            std::unique(candidatesPerDim[innerDim].begin(), candidatesPerDim[innerDim].end()),
            candidatesPerDim[innerDim].end());
      }
      break;
    }
    
    case ComputationalPattern::GENERIC:
    default:
      // No additional adjustments for generic pattern
      break;
  }
  
  return candidatesPerDim;
}

bool TileSizeOptimizer::isValidTileConfig(const CompleteTileConfig &config) {
  // 检查每个维度的tile大小是否有效
  for (const auto &tileConfig : config.perDimConfig) {
    // 确保tile大小在合理范围内
    if (tileConfig.tileSize < 1 || tileConfig.tileSize > hwParams.maxThreadsPerBlock) {
      return false;
    }
  }
  
  return true;
}

CompleteTileConfig TileSizeOptimizer::findOptimalTileConfig(
    const std::vector<LoopInfo> &loopInfos,
    const std::vector<MemoryAccessInfo> &memAccesses,
    const std::vector<ComputationInfo> &computations,
    const std::vector<std::vector<int64_t>> &tileSizeCandidates) {
  
  const int numDimensions = loopInfos.size();
  
  // 初始化DP表
  // dp[d][i] = 当第d维使用索引i对应的配置，且前d-1维使用最优配置时的最佳得分
  std::vector<std::vector<float>> dp(numDimensions);
  std::vector<std::vector<int>> prevConfig(numDimensions);
  
  // 保存最佳配置
  std::vector<std::vector<CompleteTileConfig>> bestConfigs(numDimensions);
  
  // 初始化第一维
  dp[0].resize(tileSizeCandidates[0].size(), -std::numeric_limits<float>::infinity());
  prevConfig[0].resize(tileSizeCandidates[0].size(), -1);
  bestConfigs[0].resize(tileSizeCandidates[0].size());
  
  for (size_t i = 0; i < tileSizeCandidates[0].size(); ++i) {
    CompleteTileConfig config;
    
    // 创建第一维配置
    TileConfig dimConfig;
    dimConfig.tileSize = tileSizeCandidates[0][i];
    
    config.perDimConfig.push_back(dimConfig);
    
    // 检查有效性并评估
    if (isValidTileConfig(config)) {
      // 评估单维度性能
      float score = evaluateConfig(config, loopInfos, memAccesses, computations);
      dp[0][i] = score;
      bestConfigs[0][i] = config;
    }
  }
  
  // 填充DP表，处理后续维度
  for (int dim = 1; dim < numDimensions; ++dim) {
    dp[dim].resize(tileSizeCandidates[dim].size(), -std::numeric_limits<float>::infinity());
    prevConfig[dim].resize(tileSizeCandidates[dim].size(), -1);
    bestConfigs[dim].resize(tileSizeCandidates[dim].size());
    
    for (size_t i = 0; i < tileSizeCandidates[dim].size(); ++i) {
      float bestScore = -std::numeric_limits<float>::infinity();
      int bestPrevIdx = -1;
      
      // 尝试与前一维度的每种配置组合
      for (size_t j = 0; j < dp[dim-1].size(); ++j) {
        // 跳过无效的前一配置
        if (dp[dim-1][j] == -std::numeric_limits<float>::infinity()) {
          continue;
        }
        
        // 创建新配置，扩展前一维度的最佳配置
        CompleteTileConfig newConfig = bestConfigs[dim-1][j];
        
        // 添加当前维度
        TileConfig dimConfig;
        dimConfig.tileSize = tileSizeCandidates[dim][i];
        
        newConfig.perDimConfig.push_back(dimConfig);
        
        // 检查有效性
        if (isValidTileConfig(newConfig)) {
          // 评估组合性能
          float score = evaluateConfig(newConfig, loopInfos, memAccesses, computations);
          
          if (score > bestScore) {
            bestScore = score;
            bestPrevIdx = j;
            bestConfigs[dim][i] = newConfig;
          }
        }
      }
      
      dp[dim][i] = bestScore;
      prevConfig[dim][i] = bestPrevIdx;
    }
  }
  
  // 在最终维度找到最佳配置
  float bestFinalScore = -std::numeric_limits<float>::infinity();
  int bestFinalIdx = -1;
  
  for (size_t i = 0; i < dp[numDimensions-1].size(); ++i) {
    if (dp[numDimensions-1][i] > bestFinalScore) {
      bestFinalScore = dp[numDimensions-1][i];
      bestFinalIdx = i;
    }
  }
  
  // 返回最佳配置
  CompleteTileConfig optimalConfig;
  
  if (bestFinalIdx != -1) {
    optimalConfig = bestConfigs[numDimensions-1][bestFinalIdx];
    optimalConfig.overallPerformanceScore = bestFinalScore;
  } else {
    // 如果没有找到有效配置，使用默认配置
    optimalConfig.perDimConfig.resize(numDimensions);
    for (int i = 0; i < numDimensions; ++i) {
      optimalConfig.perDimConfig[i].tileSize = (i == 0 || i == 1) ? 32 : 1;
    }
    optimalConfig.overallPerformanceScore = 0.0f;
  }
  
  return optimalConfig;
}

//===----------------------------------------------------------------------===//
// 性能评估模型实现
//===----------------------------------------------------------------------===//

// float TileSizeOptimizer::evaluateConfig(
//     const CompleteTileConfig &config,
//     const std::vector<LoopInfo> &loopInfos,
//     const std::vector<MemoryAccessInfo> &memAccesses,
//     const std::vector<ComputationInfo> &computations) {
  
//   // 计算各个性能因素
//   float arithmeticIntensityScore = evaluateArithmeticIntensity(computations, memAccesses);
//   float occupancyScore = evaluateOccupancy(config);
//   float memoryEfficiencyScore = evaluateMemoryEfficiency(config, loopInfos, memAccesses);
//   float loadBalancingScore = evaluateLoadBalancing(config, loopInfos);
  
//   // 各因素的权重（可基于经验结果调整）
//   const float w_roofline = 0.4f;
//   const float w_occupancy = 0.3f;
//   const float w_memory = 0.2f;
//   const float w_balance = 0.1f;
  
//   // 组合所有因素得到最终评分
//   float totalScore = (
//       w_roofline * arithmeticIntensityScore +
//       w_occupancy * occupancyScore +
//       w_memory * memoryEfficiencyScore +
//       w_balance * loadBalancingScore
//   ) / (w_roofline + w_occupancy + w_memory + w_balance);
  
//   return totalScore;
// }

// float TileSizeOptimizer::evaluateArithmeticIntensity(
//     const std::vector<ComputationInfo> &computations,
//     const std::vector<MemoryAccessInfo> &memAccesses) {
  
//   // 计算总浮点操作数
//   int64_t totalFLOPs = 0;
//   for (const auto &comp : computations) {
//     if (comp.isFloatingPoint) {
//       totalFLOPs += comp.opCount;
//     }
//   }
  
//   // 计算总内存访问字节数
//   int64_t totalBytesAccessed = 0;
//   for (const auto &access : memAccesses) {
//     totalBytesAccessed += access.dataTypeSizeInBytes;
//   }
  
//   // 避免除零
//   if (totalBytesAccessed == 0) {
//     return 0.0f;
//   }
  
//   // 计算算术强度（每字节的浮点操作数）
//   float arithmeticIntensity = static_cast<float>(totalFLOPs) / totalBytesAccessed;
  
//   // 应用Roofline模型
//   float computeBound = hwParams.peakComputePerformance;
//   float memoryBound = arithmeticIntensity * hwParams.memoryBandwidth;
  
//   // 根据Roofline模型可达到的性能
//   float attainablePerformance = std::min(computeBound, memoryBound);
  
//   // 归一化到[0, 1]范围，除以峰值计算性能
//   float normalizedPerformance = attainablePerformance / hwParams.peakComputePerformance;
  
//   return normalizedPerformance;
// }

// float TileSizeOptimizer::evaluateOccupancy(const CompleteTileConfig &config) {
//   // 计算总tile大小（所有维度的乘积）
//   int64_t totalTileSize = 1;
//   for (const auto &tileConfig : config.perDimConfig) {
//     totalTileSize *= tileConfig.tileSize;
//   }
  
//   // 检查硬件约束
//   if (totalTileSize > hwParams.maxThreadsPerBlock) {
//     return 0.0f;  // 无效配置
//   }
  
//   // 估计每线程寄存器使用量
//   int registersPerThread = 32;  // 默认估计，可进一步细化
  
//   // 估计每块共享内存
//   int sharedMemoryPerBlock = totalTileSize * sizeof(float);  // 简化估计
  
//   // 计算理论占用率
//   int maxBlocksPerSM_byThreadCount = hwParams.maxThreadsPerSM / totalTileSize;
//   int maxBlocksPerSM_byRegisters = 
//       hwParams.maxRegistersPerSM / (registersPerThread * totalTileSize);
//   int maxBlocksPerSM_bySharedMem = 
//       (sharedMemoryPerBlock > 0) ? 
//       hwParams.maxSharedMemoryPerSM / sharedMemoryPerBlock : 
//       hwParams.maxBlocksPerSM;
  
//   int maxBlocksPerSM = std::min({
//     hwParams.maxBlocksPerSM,
//     maxBlocksPerSM_byThreadCount,
//     maxBlocksPerSM_byRegisters,
//     maxBlocksPerSM_bySharedMem
//   });
  
//   float occupancy = 
//       static_cast<float>(maxBlocksPerSM * totalTileSize) / hwParams.maxThreadsPerSM;
  
//   return occupancy;
// }

// float TileSizeOptimizer::evaluateMemoryEfficiency(
//     const CompleteTileConfig &config,
//     const std::vector<LoopInfo> &loopInfos,
//     const std::vector<MemoryAccessInfo> &memAccesses) {
  
//   // 评估合并内存访问效率（权重：0.7）
//   float coalescedScore = estimateCoalescedAccess(config, memAccesses);
  
//   // 评估缓存利用率（权重：0.3）
//   float cacheScore = estimateCacheUtilization(config, memAccesses);
  
//   // 使用适当权重组合评分
//   float memoryEfficiency = 0.7f * coalescedScore + 0.3f * cacheScore;
  
//   return memoryEfficiency;
// }

// float TileSizeOptimizer::estimateCoalescedAccess(
//     const CompleteTileConfig &config,
//     const std::vector<MemoryAccessInfo> &memAccesses) {
  
//   float totalScore = 0.0f;
//   int totalAccesses = 0;
  
//   for (const auto &access : memAccesses) {
//     totalAccesses++;
    
//     // 不可分析模式的默认评分
//     float accessScore = 0.1f;
    
//     switch (access.pattern) {
//       case MemoryAccessPattern::COALESCED:
//         // 最内层循环tile大小是32的倍数时，合并访问效率更高
//         if (!config.perDimConfig.empty() && config.perDimConfig.back().tileSize % 32 == 0) {
//           accessScore = 1.0f;
//         } else {
//           accessScore = 0.8f;
//         }
//         break;
//       case MemoryAccessPattern::SEQUENTIAL:
//         accessScore = 0.5f;
//         break;
//       case MemoryAccessPattern::STRIDED:
//         // 跨步访问的较低评分，基于估计的步长
//         accessScore = 0.3f;
//         break;
//       case MemoryAccessPattern::RANDOM:
//         accessScore = 0.1f;
//         break;
//     }
    
//     totalScore += accessScore;
//   }
  
//   // 避免除零
//   if (totalAccesses == 0) {
//     return 0.5f;  // 没有内存访问时的默认中等评分
//   }
  
//   return totalScore / totalAccesses;
// }

// float TileSizeOptimizer::estimateCacheUtilization(
//     const CompleteTileConfig &config,
//     const std::vector<MemoryAccessInfo> &memAccesses) {
  
//   // 估计数据重用因子
//   float dataReuse = 1.0f;  // 默认：每块数据使用一次
  
//   // 对于矩阵乘法类型模式，数据重用与tile大小成正比
//   // 这是一个简化模型，需要针对其他模式进行细化
//   int64_t totalTileSize = 1;
//   for (const auto &tileConfig : config.perDimConfig) {
//     totalTileSize *= tileConfig.tileSize;
//   }
  
//   // 简单模型：数据重用随tile大小增加而增加，直到达到某个点
//   dataReuse = std::min(5.0f, std::sqrt(static_cast<float>(totalTileSize)) / 10.0f + 1.0f);
  
//   // 缓存行通常为128字节
//   const int cacheLineSize = 128;
  
//   // 估计缓存利用率（上限为1.0）
//   float cacheUtilization = std::min(1.0f, dataReuse / (cacheLineSize / 4.0f));
  
//   return cacheUtilization;
// }

// float TileSizeOptimizer::evaluateLoadBalancing(
//     const CompleteTileConfig &config,
//     const std::vector<LoopInfo> &loopInfos) {
  
//   float balanceScore = 1.0f;
  
//   // 检查每个维度的负载均衡情况
//   for (size_t i = 0; i < config.perDimConfig.size() && i < loopInfos.size(); ++i) {
//     const auto &tileConfig = config.perDimConfig[i];
//     const auto &loopInfo = loopInfos[i];
    
//     // 跳过没有常量边界的循环
//     if (!loopInfo.hasConstantBounds) {
//       continue;
//     }
    
//     // 计算循环范围和块数
//     int64_t range = loopInfo.constantUpperBound - loopInfo.constantLowerBound;
//     int64_t numBlocks = (range + tileConfig.tileSize - 1) / tileConfig.tileSize;
    
//     // 检查是否有余数
//     if (range % tileConfig.tileSize != 0) {
//       int64_t lastBlockSize = range % tileConfig.tileSize;
//       float dimensionBalanceScore = static_cast<float>(lastBlockSize) / tileConfig.tileSize;
      
//       // 更新总体平衡评分（使用所有维度的最小值）
//       balanceScore = std::min(balanceScore, dimensionBalanceScore);
//     }
//   }
  
//   return balanceScore;
// }

// int TileSizeOptimizer::estimateRegistersPerThread(
//     const std::vector<ComputationInfo> &computations) {
  
//   // 基于操作数量的基本寄存器估计
//   int baseRegisters = 16;  // 循环开销等的基础寄存器
//   int opRegisters = 0;
  
//   for (const auto &comp : computations) {
//     // 不同操作需要不同数量的寄存器
//     if (auto arithOp = dyn_cast<arith::AddFOp>(comp.op)) {
//       opRegisters += 3 * comp.opCount;  // 源操作数 + 结果
//     } else if (auto arithOp = dyn_cast<arith::MulFOp>(comp.op)) {
//       opRegisters += 3 * comp.opCount;
//     } else {
//       // 其他操作的默认值
//       opRegisters += 2 * comp.opCount;
//     }
//   }
  
//   // 估计总寄存器数，上限为最大允许值
//   int totalRegisters = baseRegisters + std::min(opRegisters, 100);
//   return std::min(totalRegisters, hwParams.maxRegistersPerThread);
// }

// int TileSizeOptimizer::calculateSharedMemoryUsage(
//     const CompleteTileConfig &config,
//     const std::vector<MemoryAccessInfo> &memAccesses) {
  
//   // 简化估计：基于tile大小和典型数据类型
//   int bytesPerElement = 4;  // 假设大多数是float类型 (4字节)
  
//   // 计算tile的总元素数
//   int64_t totalElements = 1;
//   for (const auto &tileConfig : config.perDimConfig) {
//     totalElements *= tileConfig.tileSize;
//   }
  
//   // 估计共享内存用量（简化模型）
//   int sharedMemory = bytesPerElement * totalElements;
  
//   return sharedMemory;
// }

float TileSizeOptimizer::evaluateConfig(
    const CompleteTileConfig &config,
    const std::vector<LoopInfo> &loopInfos,
    const std::vector<MemoryAccessInfo> &memAccesses,
    const std::vector<ComputationInfo> &computations) {
  
  // Calculate individual performance factors with improved models
  float arithmeticIntensityScore = evaluateArithmeticIntensity(config, computations, memAccesses);
  float occupancyScore = evaluateOccupancy(config, computations);
  float memoryEfficiencyScore = evaluateMemoryEfficiency(config, loopInfos, memAccesses);
  float loadBalancingScore = evaluateLoadBalancing(config, loopInfos);
  float dataReuseScore = evaluateDataReuse(config, loopInfos, memAccesses);
  
  // Weights for different architectures (could be adjusted based on target)
  const float w_roofline = 0.35f;   // Arithmetic intensity importance
  const float w_occupancy = 0.25f;  // Resource utilization importance
  const float w_memory = 0.25f;     // Memory access pattern importance
  const float w_balance = 0.05f;    // Load balancing importance
  const float w_reuse = 0.10f;      // Data reuse importance
  
  // Combined score with all factors
  float totalScore = (
      w_roofline * arithmeticIntensityScore +
      w_occupancy * occupancyScore +
      w_memory * memoryEfficiencyScore +
      w_balance * loadBalancingScore +
      w_reuse * dataReuseScore
  ) / (w_roofline + w_occupancy + w_memory + w_balance + w_reuse);
  
  LLVM_DEBUG(llvm::dbgs() << "Tile Config: ";
             for (const auto &dim : config.perDimConfig) {
               llvm::dbgs() << dim.tileSize << " ";
             }
             llvm::dbgs() << "\n  AI Score: " << arithmeticIntensityScore
                         << ", Occ: " << occupancyScore
                         << ", Mem: " << memoryEfficiencyScore
                         << ", Bal: " << loadBalancingScore
                         << ", Reuse: " << dataReuseScore
                         << ", Total: " << totalScore << "\n");
  
  return totalScore;
}

float TileSizeOptimizer::evaluateArithmeticIntensity(
    const CompleteTileConfig &config,
    const std::vector<ComputationInfo> &computations,
    const std::vector<MemoryAccessInfo> &memAccesses) {
  
  // Calculate total tile size
  int64_t totalTileSize = 1;
  for (const auto &tileConfig : config.perDimConfig) {
    totalTileSize *= tileConfig.tileSize;
  }
  
  // Calculate total operations with operation-specific weights
  int64_t totalOps = 0;
  for (const auto &comp : computations) {
    float opWeight = 1.0;
    
    // Assign different weights to different operations
    if (isa<arith::MulFOp>(comp.op) || isa<arith::MulIOp>(comp.op)) {
      opWeight = 1.5;  // Multiplications typically more expensive
    } else if (isa<arith::DivFOp>(comp.op) || isa<arith::DivSIOp>(comp.op) || 
               isa<arith::DivUIOp>(comp.op)) {
      opWeight = 3.0;  // Divisions are very expensive
    }
    
    totalOps += comp.opCount * opWeight;
  }
  
  // Estimate memory transactions with cache effects
  int64_t totalMemoryTransactions = 0;
  llvm::DenseMap<Value, float> memrefAccessCount;
  
  for (const auto &access : memAccesses) {
    // Count accesses per memref to model cache effects
    memrefAccessCount[access.memref] += 1.0;
  }
  
  // Calculate memory transactions with cache effects
  for (const auto &pair : memrefAccessCount) {
    Value memref = pair.first;
    float accessCount = pair.second;
    
    // Find the element size for this memref
    int elementSize = 4;  // Default to 4 bytes
    for (const auto &access : memAccesses) {
      if (access.memref == memref) {
        elementSize = access.dataTypeSizeInBytes;
        break;
      }
    }
    
    // Model cache behavior: first access is full cost, subsequent accesses benefit from cache
    float cacheFactor = std::min(1.0f, 1.0f / std::sqrt(accessCount));
    float effectiveBytes = totalTileSize * elementSize * cacheFactor;
    
    totalMemoryTransactions += static_cast<int64_t>(effectiveBytes);
  }
  
  // Avoid division by zero
  if (totalMemoryTransactions == 0) {
    return 0.5f;  // Default medium score
  }
  
  // Calculate arithmetic intensity (operations per byte)
  float arithmeticIntensity = static_cast<float>(totalOps) / totalMemoryTransactions;
  
  // Apply Roofline model with realistic performance curve
  float peakComputeThroughput = hwParams.peakComputePerformance;
  float peakMemoryBandwidth = hwParams.memoryBandwidth;
  
  // Calculate ridge point (arithmetic intensity where compute = memory bandwidth)
  float ridgePoint = peakComputeThroughput / peakMemoryBandwidth;
  
  // Calculate attainable performance as percentage of peak
  float attainablePerformance;
  if (arithmeticIntensity < ridgePoint) {
    // Memory-bound region
    attainablePerformance = arithmeticIntensity * peakMemoryBandwidth / peakComputeThroughput;
    
    // Add penalty for very low arithmetic intensity (heavy memory bottleneck)
    if (arithmeticIntensity < 0.1f * ridgePoint) {
      attainablePerformance *= 0.8f;
    }
  } else {
    // Compute-bound region
    attainablePerformance = 1.0f;
    
    // Add bonus for balanced computation (near ridge point)
    if (arithmeticIntensity < 2.0f * ridgePoint) {
      attainablePerformance *= 1.1f;
    }
  }
  
  // Normalize to [0,1] range
  return std::min(1.0f, attainablePerformance);
}

float TileSizeOptimizer::evaluateOccupancy(
    const CompleteTileConfig &config,
    const std::vector<ComputationInfo> &computations) {
  
  // Calculate total threads per block
  int64_t totalThreadsPerBlock = 1;
  for (const auto &tileConfig : config.perDimConfig) {
    totalThreadsPerBlock *= tileConfig.tileSize;
  }
  
  // Check if thread count exceeds maximum
  if (totalThreadsPerBlock > hwParams.maxThreadsPerBlock) {
    return 0.0f;  // Invalid configuration
  }
  
  // Better register usage estimation based on operation types
  int registersPerThread = estimateRegistersPerThread(computations);
  
  // Better shared memory usage estimation
  int sharedMemoryPerBlock = 0;
  // Could be enhanced with more detailed analysis
  
  // Calculate theoretical occupancy limited by different factors
  
  // Threads per SM limit
  int maxBlocksPerSM_byThreadCount = hwParams.maxThreadsPerSM / totalThreadsPerBlock;
  
  // Registers per SM limit
  int maxBlocksPerSM_byRegisters = hwParams.maxRegistersPerSM / 
                                 (registersPerThread * totalThreadsPerBlock);
  
  // Shared memory per SM limit
  int maxBlocksPerSM_bySharedMem = (sharedMemoryPerBlock > 0) ?
                                 hwParams.maxSharedMemoryPerSM / sharedMemoryPerBlock :
                                 hwParams.maxBlocksPerSM;
  
  // Hardware blocks per SM limit
  int maxBlocksPerSM = std::min({
    hwParams.maxBlocksPerSM,
    maxBlocksPerSM_byThreadCount,
    maxBlocksPerSM_byRegisters,
    maxBlocksPerSM_bySharedMem
  });
  
  // Calculate occupancy as fraction of maximum possible warps
  int warpsPerBlock = (totalThreadsPerBlock + hwParams.warpSize - 1) / hwParams.warpSize;
  int activeWarps = maxBlocksPerSM * warpsPerBlock;
  int maxWarpsPerSM = hwParams.maxThreadsPerSM / hwParams.warpSize;
  
  float occupancy = static_cast<float>(activeWarps) / maxWarpsPerSM;
  
  // Apply occupancy-performance curve
  // Research shows that ~70% occupancy is often sufficient for good performance
  float normalizedOccupancy;
  if (occupancy < 0.2f) {
    // Very low occupancy is bad
    normalizedOccupancy = occupancy * 2.5f;
  } else if (occupancy < 0.7f) {
    // Medium occupancy - linear improvement
    normalizedOccupancy = 0.5f + (occupancy - 0.2f) * 0.7f;
  } else {
    // Diminishing returns after 70% occupancy
    normalizedOccupancy = 0.85f + (occupancy - 0.7f) * 0.5f;
  }
  
  // Warp utilization factor - bonus for sizes that are multiples of warp size
  float warpUtilization = 1.0f;
  if (totalThreadsPerBlock % hwParams.warpSize != 0) {
    // Partial warps reduce efficiency
    int lastWarpSize = totalThreadsPerBlock % hwParams.warpSize;
    warpUtilization = 1.0f - 0.2f * (1.0f - static_cast<float>(lastWarpSize) / hwParams.warpSize);
  }
  
  // Combine factors
  return normalizedOccupancy * warpUtilization;
}

float TileSizeOptimizer::evaluateMemoryEfficiency(
    const CompleteTileConfig &config,
    const std::vector<LoopInfo> &loopInfos,
    const std::vector<MemoryAccessInfo> &memAccesses) {
  
  // Evaluate coalesced memory access (weight: 0.6)
  float coalescedScore = estimateCoalescedAccess(config, memAccesses);
  
  // Evaluate cache utilization (weight: 0.3)
  float cacheScore = estimateCacheUtilization(config, memAccesses);
  
  // Evaluate memory bank conflicts (weight: 0.1)
  float bankConflictScore = estimateMemoryBankConflicts(config, memAccesses);
  
  // Combined memory efficiency score
  float memoryEfficiency = 0.6f * coalescedScore + 
                         0.3f * cacheScore + 
                         0.1f * bankConflictScore;
  
  return memoryEfficiency;
}

float TileSizeOptimizer::estimateCoalescedAccess(
    const CompleteTileConfig &config,
    const std::vector<MemoryAccessInfo> &memAccesses) {
  
  float totalScore = 0.0f;
  int totalAccesses = 0;
  
  // If no memory accesses, return neutral score
  if (memAccesses.empty()) {
    return 0.5f;
  }
  
  // Process each memory access
  for (const auto &access : memAccesses) {
    totalAccesses++;
    float accessScore = 0.1f;  // Default low score
    
    // Evaluate based on access pattern
    switch (access.pattern) {
      case MemoryAccessPattern::COALESCED:
        // Check if innermost dimension tile size is warp-friendly
        if (!config.perDimConfig.empty()) {
          int64_t innermostTileSize = config.perDimConfig.back().tileSize;
          
          if (innermostTileSize == hwParams.warpSize) {
            // Perfect coalescing with warp size
            accessScore = 1.0f;
          } else if (innermostTileSize % hwParams.warpSize == 0) {
            // Multiple of warp size - still good
            accessScore = 0.95f;
          } else if (innermostTileSize % 16 == 0) {
            // Multiple of half-warp - reasonable
            accessScore = 0.85f;
          } else if (innermostTileSize % 8 == 0) {
            // Multiple of quarter-warp - acceptable
            accessScore = 0.75f;
          } else if (innermostTileSize >= hwParams.warpSize) {
            // Larger than warp size but not aligned - some inefficiency
            accessScore = 0.7f;
          } else {
            // Smaller than warp size - sub-optimal
            accessScore = 0.6f * static_cast<float>(innermostTileSize) / hwParams.warpSize;
          }
        }
        break;
        
      case MemoryAccessPattern::SEQUENTIAL:
        // Sequential but not coalesced - medium score
        accessScore = 0.5f;
        break;
        
      case MemoryAccessPattern::STRIDED:
        // Strided access - low score with some consideration for stride
        accessScore = 0.3f;
        break;
        
      case MemoryAccessPattern::RANDOM:
        // Random access - very low score
        accessScore = 0.1f;
        break;
    }
    
    totalScore += accessScore;
  }
  
  return totalScore / totalAccesses;
}

float TileSizeOptimizer::estimateCacheUtilization(
    const CompleteTileConfig &config,
    const std::vector<MemoryAccessInfo> &memAccesses) {
  
  // Calculate total tile size in elements
  int64_t totalTileSize = 1;
  for (const auto &tileConfig : config.perDimConfig) {
    totalTileSize *= tileConfig.tileSize;
  }
  
  // Estimate working set size
  int64_t workingSetBytes = 0;
  llvm::DenseSet<Value> uniqueMemrefs;
  
  for (const auto &access : memAccesses) {
    if (uniqueMemrefs.insert(access.memref).second) {
      workingSetBytes += totalTileSize * access.dataTypeSizeInBytes;
    }
  }
  
  // Cache sizes - typical L1 and L2 sizes
  const int L1CacheSize = hwParams.l1CacheSize;
  const int L2CacheSize = hwParams.l2CacheSize;
  
  // Score based on working set vs cache size
  float cacheScore = 0.0f;
  
  if (workingSetBytes <= L1CacheSize) {
    // Working set fits in L1 cache - excellent
    cacheScore = 1.0f;
  } else if (workingSetBytes <= L2CacheSize) {
    // Working set fits in L2 cache - good
    float l2Ratio = static_cast<float>(L2CacheSize - workingSetBytes) / 
                   (L2CacheSize - L1CacheSize);
    cacheScore = 0.7f + 0.3f * l2Ratio;
  } else {
    // Working set exceeds L2 cache - poor
    float excessRatio = std::min(1.0f, static_cast<float>(L2CacheSize) / workingSetBytes);
    cacheScore = 0.3f * excessRatio;
  }
  
  // Spatial locality bonus - prefer larger tile sizes for better spatial locality
  float spatialLocalityFactor = std::min(1.0f, std::log2f(totalTileSize) / 8.0f);
  
  return cacheScore * (0.7f + 0.3f * spatialLocalityFactor);
}

float TileSizeOptimizer::estimateMemoryBankConflicts(
    const CompleteTileConfig &config,
    const std::vector<MemoryAccessInfo> &memAccesses) {
  
  // Simple model: Penalize tile sizes that are likely to cause bank conflicts
  
  // In NVIDIA GPUs, shared memory is divided into 32 banks
  const int numBanks = 32;
  
  // Check if any dimension size is likely to cause conflicts
  float conflictScore = 1.0f;
  
  for (const auto &tileConfig : config.perDimConfig) {
    int64_t tileSize = tileConfig.tileSize;
    
    // Check for patterns known to cause bank conflicts
    if (tileSize % numBanks == 0) {
      // Perfect - no conflicts
      continue;
    } else if (tileSize % 16 == 0) {
      // Reasonably good
      conflictScore *= 0.95f;
    } else if (tileSize % 8 == 0) {
      // Some conflicts likely
      conflictScore *= 0.9f;
    } else if (tileSize % 2 == 0) {
      // More conflicts likely
      conflictScore *= 0.8f;
    } else {
      // Odd sizes can be better than even sizes for bank conflicts
      conflictScore *= 0.85f;
    }
  }
  
  return conflictScore;
}

float TileSizeOptimizer::evaluateLoadBalancing(
    const CompleteTileConfig &config,
    const std::vector<LoopInfo> &loopInfos) {
  
  float balanceScore = 1.0f;
  
  // Check each dimension for load balancing issues
  for (size_t i = 0; i < config.perDimConfig.size() && i < loopInfos.size(); ++i) {
    const auto &tileConfig = config.perDimConfig[i];
    const auto &loopInfo = loopInfos[i];
    
    // Skip dimensions without constant bounds
    if (!loopInfo.hasConstantBounds) {
      continue;
    }
    
    // Calculate loop range and number of blocks
    int64_t range = loopInfo.constantUpperBound - loopInfo.constantLowerBound;
    int64_t tileSize = tileConfig.tileSize;
    int64_t numBlocks = (range + tileSize - 1) / tileSize;
    
    // Check for uneven division (remainder)
    if (range % tileSize != 0) {
      int64_t lastBlockSize = range % tileSize;
      
      // Calculate load imbalance factor
      float dimensionBalanceScore;
      
      if (numBlocks == 1) {
        // Only one block - no imbalance
        dimensionBalanceScore = 1.0f;
      } else {
        // Multiple blocks with one smaller block
        float fullBlocks = numBlocks - 1;
        float totalWork = fullBlocks * tileSize + lastBlockSize;
        float idealWork = totalWork / numBlocks;
        float maxDeviation = std::max(
            std::abs(tileSize - idealWork),
            std::abs(lastBlockSize - idealWork)
        ) / idealWork;
        
        // Score based on deviation from ideal
        dimensionBalanceScore = 1.0f - 0.5f * maxDeviation;
      }
      
      // Update overall balance score (use minimum across dimensions)
      balanceScore = std::min(balanceScore, dimensionBalanceScore);
    }
  }
  
  return balanceScore;
}

float TileSizeOptimizer::evaluateDataReuse(
    const CompleteTileConfig &config,
    const std::vector<LoopInfo> &loopInfos,
    const std::vector<MemoryAccessInfo> &memAccesses) {
  
  // This new function evaluates data reuse potential
  
  // Calculate tile sizes and total tile elements
  int64_t totalTileElements = 1;
  for (const auto &tileConfig : config.perDimConfig) {
    totalTileElements *= tileConfig.tileSize;
  }
  
  // Count memory references and accesses
  llvm::DenseMap<Value, int> memrefAccessCounts;
  int totalAccesses = 0;
  
  for (const auto &access : memAccesses) {
    memrefAccessCounts[access.memref]++;
    totalAccesses++;
  }
  
  // No accesses means no reuse
  if (totalAccesses == 0) {
    return 0.5f;
  }
  
  // Calculate average accesses per memref
  float avgAccessesPerMemref = static_cast<float>(totalAccesses) / memrefAccessCounts.size();
  
  // Higher average accesses means more potential data reuse
  float reuseScore = std::min(1.0f, avgAccessesPerMemref / 10.0f);
  
  // Adjust based on tile size - larger tiles can increase reuse
  float tileSizeFactor = std::min(1.0f, std::log2f(totalTileElements) / 10.0f);
  
  // Adjust based on access patterns - sequential/coalesced patterns often have better reuse
  float patternFactor = 0.5f;
  int coalescedCount = 0;
  
  for (const auto &access : memAccesses) {
    if (access.pattern == MemoryAccessPattern::COALESCED ||
        access.pattern == MemoryAccessPattern::SEQUENTIAL) {
      coalescedCount++;
    }
  }
  
  if (totalAccesses > 0) {
    patternFactor = 0.5f + 0.5f * static_cast<float>(coalescedCount) / totalAccesses;
  }
  
  // Combine factors
  return reuseScore * 0.6f + tileSizeFactor * 0.2f + patternFactor * 0.2f;
}

int TileSizeOptimizer::estimateRegistersPerThread(
    const std::vector<ComputationInfo> &computations) {
  
  // Base registers for loop overhead
  int baseRegisters = 16;
  
  // Maps to track unique operations and operands
  llvm::DenseSet<Operation*> uniqueOps;
  llvm::DenseSet<Value> uniqueOperands;
  int intermediateResults = 0;
  
  for (const auto &comp : computations) {
    // Track unique operations
    uniqueOps.insert(comp.op);
    
    // Track operands
    for (Value operand : comp.op->getOperands()) {
      uniqueOperands.insert(operand);
    }
    
    // Track results that need registers
    intermediateResults += comp.op->getNumResults();
    
    // Operation-specific register estimates
    if (isa<arith::MulFOp>(comp.op) || isa<arith::DivFOp>(comp.op)) {
      // Complex operations may need temporary registers
      baseRegisters += 1;
    }
  }
  
  // Calculate total register estimate - conservative approach
  int totalRegisters = baseRegisters + 
                     uniqueOperands.size() +
                     intermediateResults;
  
  // Apply upper bound limit
  return std::min(totalRegisters, hwParams.maxRegistersPerThread);
}

