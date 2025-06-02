#ifndef KERNEL_EXECUTION_OPTIMIZER_H
#define KERNEL_EXECUTION_OPTIMIZER_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include <vector>
#include <memory>

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Enums and Basic Types
//===----------------------------------------------------------------------===//

// 操作类型枚举
enum class OperationType {
  GPU_KERNEL,           // gpu.launch_func
  CUDNN_CONV,          // 卷积操作
  CUDNN_POOL,          // 池化操作  
  CUDNN_ACTIVATION,    // 激活函数
  CUDNN_ELEMENTWISE,   // 元素级操作 (add, mul, etc.)
  CUBLAS_GEMM,         // 矩阵乘法
  CUBLAS_FC,           // 全连接层
  MEMORY_OP,           // 内存操作
  UNKNOWN
};

// 计算特性枚举
enum class ComputeCharacteristic {
  COMPUTE_INTENSIVE,   // 计算密集型
  MEMORY_INTENSIVE,    // 存储密集型
  BALANCED            // 平衡型
};

//===----------------------------------------------------------------------===//
// Data Structures
//===----------------------------------------------------------------------===//
// 操作执行成本模型
struct OperationCost {
  double computeOps;        // 计算操作数 (FLOPs)
  double memoryAccess;      // 内存访问量 (bytes)
  double bandwidth;         // 所需带宽 (bytes/s)
  double estimatedTime;     // 预估执行时间 (ms)
  OperationType type;
  ComputeCharacteristic characteristic;
  double efficiency;        // 执行效率因子 (库函数 > 自动生成)
  
  OperationCost() : computeOps(0), memoryAccess(0), bandwidth(0), 
                   estimatedTime(0), type(OperationType::UNKNOWN),
                   characteristic(ComputeCharacteristic::BALANCED), efficiency(1.0) {}
};

// 并行组中的操作单元
struct ParallelGroupOperation {
  mlir::Operation* op;           // MLIR操作
  mlir::Value stream;            // 分配的stream
  OperationCost cost;            // 成本模型
  std::string functionName;      // 函数名
  std::vector<int64_t> shape;    // 操作的张量形状信息
  
  ParallelGroupOperation(mlir::Operation* operation) : op(operation) {}
};

// 并行组
struct ParallelGroup {
  std::vector<std::unique_ptr<ParallelGroupOperation>> operations;
  std::vector<mlir::Value> streams;  // 该组使用的所有stream
  double totalCost;
  double maxCost;  // 该组中最大的操作成本
  
  ParallelGroup() : totalCost(0), maxCost(0) {}
};

// 调度单元 - 用于重新组织并行组内的操作
struct ScheduleUnit {
  std::vector<ParallelGroupOperation*> operations;
  OperationType dominantType;
  ComputeCharacteristic characteristic;
  double totalCost;
  double maxCost;
  bool canBeSerial;  // 是否可以串行执行
  
  ScheduleUnit() : dominantType(OperationType::UNKNOWN), 
                  characteristic(ComputeCharacteristic::BALANCED),
                  totalCost(0), maxCost(0), canBeSerial(false) {}
};

struct KernelInfo {
  ParallelGroupOperation* operation;
  double executionTime;
  ComputeCharacteristic characteristic;
  double priority;
};

//===----------------------------------------------------------------------===//
// Core Classes
//===----------------------------------------------------------------------===//

// 成本估算器
class OperationCostEstimator {
public:
  // 主要接口
  static OperationCost estimateOperationCost(mlir::Operation* op, mlir::ModuleOp moduleOp);
  
  // GPU kernel分析
  static OperationCost estimateGPUKernelCost(mlir::gpu::LaunchFuncOp launchOp, mlir::ModuleOp moduleOp);
  
  // cuDNN操作分析
  static OperationCost estimateCuDNNConvCost(mlir::LLVM::CallOp callOp);
  static OperationCost estimateCuDNNPoolCost(mlir::LLVM::CallOp callOp);
  static OperationCost estimateCuDNNElementwiseCost(mlir::LLVM::CallOp callOp);
  
  // cuBLAS操作分析
  static OperationCost estimateCuBLASCost(mlir::LLVM::CallOp callOp);
  
  // 辅助函数
  static OperationType getOperationType(mlir::Operation* op);
  static std::vector<int64_t> extractTensorShape(mlir::LLVM::CallOp callOp);
  static mlir::gpu::GPUFuncOp findGPUFunction(mlir::gpu::LaunchFuncOp launchOp, mlir::ModuleOp moduleOp);
  static OperationCost analyzeGPUFunction(mlir::gpu::GPUFuncOp funcOp, 
                                         const std::vector<int64_t>& gridDim,
                                         const std::vector<int64_t>& blockDim);
};

// 并行组识别器
class ParallelGroupIdentifier {
public:
  static std::vector<ParallelGroup> identifyParallelGroups(mlir::LLVM::LLVMFuncOp funcOp);
  
private:
  static bool isStreamAcquire(mlir::Operation* op);
  static bool isStreamRelease(mlir::Operation* op);
  static bool isStreamSync(mlir::Operation* op);
  static bool isHandleAcquire(mlir::Operation* op);
  static mlir::Value getStreamFromOperation(mlir::Operation* op);
};

// 调度优化器
// class ScheduleOptimizer {
// public:
//   static std::vector<ScheduleUnit> optimizeParallelGroup(
//     ParallelGroup& group, 
//     double toleranceFactor = 1.1,
//     bool enableTypeGrouping = true);
  
// private:
//   static std::vector<ScheduleUnit> groupByCharacteristic(
//     const std::vector<ParallelGroupOperation*>& operations);
  
//   static bool canMergeUnits(const ScheduleUnit& unit1, const ScheduleUnit& unit2, 
//                            double maxCost, double toleranceFactor);
  
//   static ScheduleUnit mergeUnits(const ScheduleUnit& unit1, const ScheduleUnit& unit2);
  
//   static double calculateSerialOverhead(const std::vector<ParallelGroupOperation*>& operations);
//   static double calculateParallelOverhead(const std::vector<ParallelGroupOperation*>& operations);
// };

class ScheduleOptimizer {
public:
  static std::vector<ScheduleUnit> optimizeParallelGroup(
    ParallelGroup& group, 
    double toleranceFactor = 1.05, // 降低默认容忍因子，更保守
    bool enableTypeGrouping = true);
  
private:
  // 新的智能装箱算法
  static std::vector<ScheduleUnit> intelligentBinPacking(
    const std::vector<ParallelGroupOperation*>& operations,
    double maxExecutionTime, 
    double toleranceFactor);
  
  // 优先级和兼容性检查
  static double calculateKernelPriority(ParallelGroupOperation* op);
  static bool canAddToUnit(const ScheduleUnit& unit, 
                          const KernelInfo& kernelInfo, 
                          double timeLimit);
  static bool areCharacteristicsCompatible(ComputeCharacteristic unitChar, 
                                         ComputeCharacteristic kernelChar);
  static bool areMemoryPatternsCompatible(const ScheduleUnit& unit, 
                                        const KernelInfo& kernelInfo);
  
  // Unit 操作
  static void addKernelToUnit(ScheduleUnit& unit, const KernelInfo& kernelInfo);
  static ScheduleUnit createUnitFromKernel(const KernelInfo& kernelInfo);
  static std::vector<ScheduleUnit> postProcessOptimization(
    std::vector<ScheduleUnit> units, double timeLimit);
  
  // 保留旧的函数作为备用（可选）
  static std::vector<ScheduleUnit> groupByCharacteristic(
    const std::vector<ParallelGroupOperation*>& operations);
  static bool canMergeUnits(const ScheduleUnit& unit1, const ScheduleUnit& unit2, 
                           double maxCost, double toleranceFactor);
  static ScheduleUnit mergeUnits(const ScheduleUnit& unit1, const ScheduleUnit& unit2);
  static double calculateSerialOverhead(const std::vector<ParallelGroupOperation*>& operations);
  static double calculateParallelOverhead(const std::vector<ParallelGroupOperation*>& operations);
};

// IR重写器
class ScheduleRewriter {
public:
  static void rewriteParallelGroup(ParallelGroup& group, 
                                  const std::vector<ScheduleUnit>& optimizedSchedule,
                                  mlir::OpBuilder& builder);
  
private:
  static mlir::Value createOrReuseStream(mlir::OpBuilder& builder, mlir::Location loc);
  static void insertStreamAcquire(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value stream);
  static void insertHandleAcquire(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value stream);
  static void insertStreamSync(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value stream);
  static void insertStreamRelease(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value stream);
  static void updateOperationStream(mlir::Operation* op, mlir::Value newStream);
  static void removeRedundantStreamOps(const std::vector<ParallelGroupOperation*>& operations,
                                     mlir::Value mergedStream, mlir::OpBuilder& builder);
};

//===----------------------------------------------------------------------===//
// Core Optimizer Class (Business Logic)
//===----------------------------------------------------------------------===//

// 核心优化器类 - 包含所有业务逻辑，不涉及 Pass 框架
class KernelExecutionOptimizer {
public:
  // 主要优化入口点
  void optimize(mlir::ModuleOp moduleOp, bool verbose = false);

private:
  bool verboseMode = false;  // 内部存储 verbose 模式状态
  
  // 函数优化统计结构
  struct FunctionOptimizationStats {
    bool hasOptimizations = false;
    size_t originalOperations = 0;
    size_t optimizedOperations = 0;
    size_t originalStreams = 0;
    size_t optimizedStreams = 0;
    size_t parallelGroups = 0;
    size_t optimizedGroups = 0;
  };

  // 优化模块中所有函数的kernel执行
  void optimizeKernelExecution(mlir::ModuleOp moduleOp);
  
  // 优化单个函数
  FunctionOptimizationStats optimizeFunction(mlir::LLVM::LLVMFuncOp funcOp, mlir::ModuleOp moduleOp);
  
  // 计算并行组中所有操作的成本
  void calculateOperationCosts(ParallelGroup& group, mlir::ModuleOp moduleOp);
  
  // 统计和报告函数
  void printFunctionStats(llvm::StringRef funcName, const FunctionOptimizationStats& stats);
  void printGlobalStats(size_t totalFunctions, size_t optimizedFunctions,
                       size_t totalOriginalOps, size_t totalOptimizedOps,
                       size_t totalOriginalStreams, size_t totalOptimizedStreams);
  
  // 估算加速比
  double calculateEstimatedSpeedup(size_t originalStreams, size_t optimizedStreams);
};

//===----------------------------------------------------------------------===//
// Utility Functions and Factory
//===----------------------------------------------------------------------===//

// 工厂函数
std::unique_ptr<mlir::Pass> createKernelExecutionOptimizerPass();

// 辅助函数
bool isComputeIntensive(const OperationCost& cost);
bool isMemoryIntensive(const OperationCost& cost);
ComputeCharacteristic determineCharacteristic(const OperationCost& cost);
OperationType classifyCuDNNOperation(llvm::StringRef funcName);
OperationType classifyCuBLASOperation(llvm::StringRef funcName);

// 调试辅助函数
const char* getOperationTypeName(OperationType type);
const char* getCharacteristicName(ComputeCharacteristic characteristic);

} // namespace onnx_mlir

#endif // KERNEL_EXECUTION_OPTIMIZER_H