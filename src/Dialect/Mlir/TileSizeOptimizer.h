#ifndef MLIR_DIALECT_SCF_TRANSFORMS_TILESIZEOPTIMIZER_H_
#define MLIR_DIALECT_SCF_TRANSFORMS_TILESIZEOPTIMIZER_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"

#include <vector>
#include <unordered_map>
#include <set>
#include <cmath>
#include <limits>
#include <algorithm>


using namespace mlir;
using namespace mlir::scf;
using namespace llvm;

/// GPU硬件参数结构体，用于性能模型计算
struct GpuHardwareParameters {
  // 常见NVIDIA GPU参数
  int maxThreadsPerBlock = 1024;
  int maxBlockDimX = 1024;
  int maxBlockDimY = 1024;
  int maxBlockDimZ = 64;
  int warpSize = 32;
  int maxRegistersPerThread = 255;
  int maxRegistersPerSM = 65536;
  int maxThreadsPerSM = 2048;
  int maxBlocksPerSM = 32;
  int maxSharedMemoryPerSM = 49152; // 字节
  float peakComputePerformance = 9.7e12;  // FLOPS
  float memoryBandwidth = 1.6e12;         // 字节/秒
  int l1CacheSize = 128 * 1024;    // 字节
  int l2CacheSize = 6 * 1024 * 1024; // 字节
};

/// 内存访问模式枚举
enum class MemoryAccessPattern {
  SEQUENTIAL,  // 相同线程的顺序访问
  COALESCED,   // 相邻线程访问相邻内存
  STRIDED,     // 跨步访问
  RANDOM       // 不规则访问模式
};

/// 从MLIR并行循环提取的循环信息
struct LoopInfo {
  int dimension;
  Value lowerBound;
  Value upperBound;
  Value step;
  int64_t constantLowerBound = -1;
  int64_t constantUpperBound = -1;
  int64_t constantStep = -1;
  bool hasConstantBounds = false;
  int64_t tripCount = -1;
};

/// 内存访问信息，用于性能建模
struct MemoryAccessInfo {
  Value memref;
  SmallVector<Value, 4> indices;
  bool isLoad;
  bool isStore;
  MemoryAccessPattern pattern;
  int dataTypeSizeInBytes;
};

/// 计算操作信息
struct ComputationInfo {
  Operation *op;
  int64_t opCount;
  bool isFloatingPoint;
};

/// 单维度tile配置
struct TileConfig {
  int64_t tileSize;
  float performanceScore;
};

/// 所有维度的完整tile配置
struct CompleteTileConfig {
  std::vector<TileConfig> perDimConfig;
  float overallPerformanceScore;
};

/// Tile大小优化器 - 使用启发式规则和动态规划
class TileSizeOptimizer {
public:
  TileSizeOptimizer(MLIRContext *context, const GpuHardwareParameters &params = GpuHardwareParameters())
      : context(context), hwParams(params) {}

  /// 分析并行循环并返回最优tile配置
  CompleteTileConfig optimizeTileSize(scf::ParallelOp parallelOp);

    /// Recognized computational patterns for specialized tuning
    enum class ComputationalPattern {
    GENERIC,     // General case
    MATMUL,      // Matrix multiplication patterns
    CONV,        // Convolution patterns
    REDUCTION,   // Reduction operations (sum, max, etc.)
    STENCIL,     // Stencil computations (e.g., 2D/3D neighbor operations)
    ELEMENTWISE  // Element-wise operations (map operations)
    };

private:
  MLIRContext *context;
  GpuHardwareParameters hwParams;

  /// 从并行循环中提取循环信息
  std::vector<LoopInfo> extractLoopInfo(scf::ParallelOp parallelOp);
  
  /// 分析循环体中的内存访问模式
  std::vector<MemoryAccessInfo> analyzeMemoryAccesses(scf::ParallelOp parallelOp);
  
  /// 分析循环体中的计算操作
  std::vector<ComputationInfo> analyzeComputations(scf::ParallelOp parallelOp);
  
/// Additional method declarations
ComputationalPattern detectComputationalPattern(
    scf::ParallelOp parallelOp,
    const std::vector<MemoryAccessInfo> &memAccesses);

/// Updated method signature for pattern-aware candidate generation
std::vector<std::vector<int64_t>> generateTileSizeCandidates(
    const std::vector<LoopInfo> &loopInfos,
    ComputationalPattern pattern = ComputationalPattern::GENERIC);

//   /// 使用启发式规则为每个维度生成候选tile大小
//   std::vector<std::vector<int64_t>> generateTileSizeCandidates(const std::vector<LoopInfo> &loopInfos);
  
  /// 使用动态规划查找最优tile配置
  CompleteTileConfig findOptimalTileConfig(
      const std::vector<LoopInfo> &loopInfos,
      const std::vector<MemoryAccessInfo> &memAccesses,
      const std::vector<ComputationInfo> &computations,
      const std::vector<std::vector<int64_t>> &tileSizeCandidates);
  
  /// 检查tile配置是否满足硬件约束
  bool isValidTileConfig(const CompleteTileConfig &config);

  // ===== 性能评估模型方法 =====
  
  /// 评估完整tile配置的性能
    float evaluateConfig(
        const CompleteTileConfig &config,
        const std::vector<LoopInfo> &loopInfos,
        const std::vector<MemoryAccessInfo> &memAccesses,
        const std::vector<ComputationInfo> &computations);

    /// 基于计算和内存访问评估算术强度 - 已修改
    float evaluateArithmeticIntensity(
        const CompleteTileConfig &config,
        const std::vector<ComputationInfo> &computations,
        const std::vector<MemoryAccessInfo> &memAccesses);

    /// 基于资源使用评估占用率 - 已修改
    float evaluateOccupancy(
        const CompleteTileConfig &config,
        const std::vector<ComputationInfo> &computations);

    /// 评估内存访问效率
    float evaluateMemoryEfficiency(
        const CompleteTileConfig &config,
        const std::vector<LoopInfo> &loopInfos,
        const std::vector<MemoryAccessInfo> &memAccesses);

    /// 评估负载均衡
    float evaluateLoadBalancing(
        const CompleteTileConfig &config,
        const std::vector<LoopInfo> &loopInfos);

    /// 估计合并内存访问效率
    float estimateCoalescedAccess(
        const CompleteTileConfig &config,
        const std::vector<MemoryAccessInfo> &memAccesses);

    /// 估计缓存利用率
    float estimateCacheUtilization(
        const CompleteTileConfig &config,
        const std::vector<MemoryAccessInfo> &memAccesses);

    /// 估计每线程寄存器使用量
    int estimateRegistersPerThread(const std::vector<ComputationInfo> &computations);

    /// 计算共享内存使用量
    int calculateSharedMemoryUsage(
        const CompleteTileConfig &config,
        const std::vector<MemoryAccessInfo> &memAccesses);

    /// Evaluate data reuse potential
    float evaluateDataReuse(
        const CompleteTileConfig &config,
        const std::vector<LoopInfo> &loopInfos,
        const std::vector<MemoryAccessInfo> &memAccesses);

    /// Estimate shared memory bank conflicts
    float estimateMemoryBankConflicts(
        const CompleteTileConfig &config,
        const std::vector<MemoryAccessInfo> &memAccesses);

    // Add any other new function declarations here
};



#endif // MLIR_DIALECT_SCF_TRANSFORMS_TILESIZEOPTIMIZER_H_