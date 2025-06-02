#include "KernelExecutionOptimizer.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>

using namespace onnx_mlir;

#define DEBUG_TYPE "kernel-execution-optimizer"

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// 调试和日志函数
//===----------------------------------------------------------------------===//

const char* getOperationTypeName(OperationType type) {
  switch (type) {
    case OperationType::GPU_KERNEL: return "GPU_KERNEL";
    case OperationType::CUDNN_CONV: return "CUDNN_CONV";
    case OperationType::CUDNN_POOL: return "CUDNN_POOL";
    case OperationType::CUDNN_ACTIVATION: return "CUDNN_ACTIVATION";
    case OperationType::CUDNN_ELEMENTWISE: return "CUDNN_ELEMENTWISE";
    case OperationType::CUBLAS_GEMM: return "CUBLAS_GEMM";
    case OperationType::CUBLAS_FC: return "CUBLAS_FC";
    case OperationType::MEMORY_OP: return "MEMORY_OP";
    case OperationType::UNKNOWN: return "UNKNOWN";
    default: return "UNDEFINED";
  }
}

const char* getCharacteristicName(ComputeCharacteristic characteristic) {
  switch (characteristic) {
    case ComputeCharacteristic::COMPUTE_INTENSIVE: return "COMPUTE_INTENSIVE";
    case ComputeCharacteristic::MEMORY_INTENSIVE: return "MEMORY_INTENSIVE";
    case ComputeCharacteristic::BALANCED: return "BALANCED";
    default: return "UNDEFINED";
  }
}

//===----------------------------------------------------------------------===//
// 常量提取辅助函数
//===----------------------------------------------------------------------===//

// 从Value中提取整型常量值
static std::optional<int64_t> extractConstantInt(mlir::Value value) {
  if (auto defOp = value.getDefiningOp()) {
    if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantOp>(defOp)) {
      if (auto intAttr = constOp.getValue().dyn_cast<mlir::IntegerAttr>()) {
        return intAttr.getInt();
      }
    }
    // 尝试从LLVM常量中提取
    if (auto llvmConstOp = mlir::dyn_cast<mlir::LLVM::ConstantOp>(defOp)) {
      if (auto intAttr = llvmConstOp.getValue().dyn_cast<mlir::IntegerAttr>()) {
        return intAttr.getInt();
      }
    }
  }
  return std::nullopt;
}

// 从操作数列表中提取维度信息
static std::vector<int64_t> extractDimensionsFromOperands(mlir::OperandRange operands, 
                                                          size_t startIdx, size_t count) {
  std::vector<int64_t> dims;
  dims.reserve(count);
  
  for (size_t i = 0; i < count && (startIdx + i) < operands.size(); i++) {
    auto constValue = extractConstantInt(operands[startIdx + i]);
    dims.push_back(constValue.value_or(1)); // 默认值为1
  }
  
  return dims;
}

//===----------------------------------------------------------------------===//
// OperationCostEstimator Implementation
//===----------------------------------------------------------------------===//

OperationCost OperationCostEstimator::estimateOperationCost(mlir::Operation* op, mlir::ModuleOp moduleOp) {
  OperationCost cost;
  
  if (auto launchOp = mlir::dyn_cast<mlir::gpu::LaunchFuncOp>(op)) {
    cost = estimateGPUKernelCost(launchOp, moduleOp);
  } else if (auto callOp = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
    auto funcName = callOp.getCallee().value_or("");
    
    if (funcName.starts_with("mgpuCudnn")) {
      if (funcName.contains("Conv2d")) {
        cost = estimateCuDNNConvCost(callOp);
      } else if (funcName.contains("Pool")) {
        cost = estimateCuDNNPoolCost(callOp);
      } else if (funcName.contains("Add") || funcName.contains("Mul") || 
                 funcName.contains("Sub") || funcName.contains("Neg")) {
        cost = estimateCuDNNElementwiseCost(callOp);
      }
    } else if (funcName.starts_with("mgpuCulibs") || funcName.contains("FullyConnected")) {
      cost = estimateCuBLASCost(callOp);
    }
    
    cost.type = getOperationType(op);
  }
  
  // 设置执行效率因子
  if (cost.type == OperationType::GPU_KERNEL) {
    cost.efficiency = 0.7;  // 自动生成的kernel效率较低
  } else {
    cost.efficiency = 1.0;  // 库函数效率高
  }
  
  // 确定计算特性
  cost.characteristic = determineCharacteristic(cost);
  
  return cost;
}

OperationCost OperationCostEstimator::estimateGPUKernelCost(mlir::gpu::LaunchFuncOp launchOp, mlir::ModuleOp moduleOp) {
  OperationCost cost;
  cost.type = OperationType::GPU_KERNEL;
  cost.efficiency = 0.7; // 自动生成的kernel效率较低
  
  // 获取启动参数
  std::vector<int64_t> gridDim(3, 1);
  std::vector<int64_t> blockDim(3, 1);
  
  // 提取grid维度
  if (auto constVal = extractConstantInt(launchOp.getGridSizeX())) {
    gridDim[0] = *constVal;
  }
  if (auto constVal = extractConstantInt(launchOp.getGridSizeY())) {
    gridDim[1] = *constVal;
  }
  if (auto constVal = extractConstantInt(launchOp.getGridSizeZ())) {
    gridDim[2] = *constVal;
  }
  
  // 提取block维度
  if (auto constVal = extractConstantInt(launchOp.getBlockSizeX())) {
    blockDim[0] = *constVal;
  }
  if (auto constVal = extractConstantInt(launchOp.getBlockSizeY())) {
    blockDim[1] = *constVal;
  }
  if (auto constVal = extractConstantInt(launchOp.getBlockSizeZ())) {
    blockDim[2] = *constVal;
  }
  
  LLVM_DEBUG(llvm::dbgs() << "GPU Kernel launch config: "
             << "grid=(" << gridDim[0] << "," << gridDim[1] << "," << gridDim[2] << "), "
             << "block=(" << blockDim[0] << "," << blockDim[1] << "," << blockDim[2] << ")\n");
  
  // 查找GPU函数并分析
  auto gpuFunc = findGPUFunction(launchOp, moduleOp);
  if (gpuFunc) {
    cost = analyzeGPUFunction(gpuFunc, gridDim, blockDim);
  } else {
    // 如果找不到函数，使用基于参数和启动配置的启发式估算
    int64_t totalThreads = gridDim[0] * gridDim[1] * gridDim[2] * 
                          blockDim[0] * blockDim[1] * blockDim[2];
    int numArgs = launchOp.getKernelOperands().size();
    
    // 基于线程数和参数数量的简单估算
    cost.computeOps = totalThreads * numArgs * 5; // 假设每线程每参数5次运算
    cost.memoryAccess = totalThreads * numArgs * 4; // 每线程每参数一次4字节访问
    cost.bandwidth = cost.memoryAccess * 0.6; // 中等带宽需求
    cost.characteristic = ComputeCharacteristic::BALANCED;
    
    // 简单的时间估算
    double computeTime = cost.computeOps / (5e10) * 1000; // 50 GFLOPS
    double memoryTime = cost.memoryAccess / (2e11) * 1000; // 200 GB/s
    cost.estimatedTime = std::max(computeTime, memoryTime);
    
    LLVM_DEBUG(llvm::dbgs() << "GPU Kernel heuristic estimation: "
               << "threads=" << totalThreads
               << ", args=" << numArgs << "\n");
  }
  
  return cost;
}

OperationCost OperationCostEstimator::estimateCuDNNConvCost(mlir::LLVM::CallOp callOp) {
  OperationCost cost;
  cost.type = OperationType::CUDNN_CONV;
  cost.characteristic = ComputeCharacteristic::COMPUTE_INTENSIVE;
  cost.efficiency = 1.0; // cuDNN库函数效率高
  
  auto operands = callOp.getOperands();
  
  // mgpuCudnnConv2dForward的参数顺序:
  // (n, c, h, w, k, r, s, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
  //  x_data, w_data, bias_data, y_data, stream)
  
  if (operands.size() >= 13) {
    // 提取卷积参数
    auto dims = extractDimensionsFromOperands(operands, 0, 13);
    
    int64_t n = dims[0];        // batch size
    int64_t c = dims[1];        // input channels
    int64_t h = dims[2];        // input height
    int64_t w = dims[3];        // input width
    int64_t k = dims[4];        // output channels
    int64_t r = dims[5];        // kernel height
    int64_t s = dims[6];        // kernel width
    int64_t pad_h = dims[7];    // padding height
    int64_t pad_w = dims[8];    // padding width
    int64_t stride_h = dims[9]; // stride height
    int64_t stride_w = dims[10]; // stride width
    
    // 计算输出尺寸
    int64_t out_h = (h + 2 * pad_h - r) / stride_h + 1;
    int64_t out_w = (w + 2 * pad_w - s) / stride_w + 1;
    
    // 计算FLOPs: 每个输出像素需要 c * r * s 次乘加运算
    cost.computeOps = n * k * out_h * out_w * c * r * s * 2.0; // MAC = 2 ops
    
    // 内存访问量 (bytes)
    cost.memoryAccess = (n * c * h * w +           // input
                        k * c * r * s +           // weights  
                        k +                       // bias
                        n * k * out_h * out_w     // output
                       ) * 4; // float32
    
    // 卷积是计算密集型，内存带宽需求相对较低
    cost.bandwidth = cost.memoryAccess * 0.2;
    
    // 基于经验的执行时间估算 (ms)
    // 考虑cuDNN的优化，假设可以达到较高的计算效率
    double computeTime = cost.computeOps / (2e12) * 1000; // 2 TFLOPS
    double memoryTime = cost.memoryAccess / (5e11) * 1000; // 500 GB/s
    cost.estimatedTime = std::max(computeTime, memoryTime);
    
    LLVM_DEBUG(llvm::dbgs() << "Conv2D cost estimation: "
               << "FLOPs=" << cost.computeOps 
               << ", Memory=" << cost.memoryAccess 
               << ", Time=" << cost.estimatedTime << "ms\n");
  } else {
    // 参数不足时使用默认估算
    cost.computeOps = 1e9;  // 1 GFLOP
    cost.memoryAccess = 1e8; // 100 MB
    cost.bandwidth = cost.memoryAccess * 0.2;
    cost.estimatedTime = 1.0; // 1ms
  }
  
  return cost;
}

OperationCost OperationCostEstimator::estimateCuDNNPoolCost(mlir::LLVM::CallOp callOp) {
  OperationCost cost;
  cost.type = OperationType::CUDNN_POOL;
  cost.characteristic = ComputeCharacteristic::MEMORY_INTENSIVE;
  cost.efficiency = 1.0;
  
  auto operands = callOp.getOperands();
  
  // mgpuCudnnMaxPoolForward参数:
  // (n, c, h, w, kernel_h, kernel_w, pad_h_begin, pad_w_begin, pad_h_end, pad_w_end, 
  //  stride_h, stride_w, dilation_h, dilation_w, input_data, output_data, stream)
  
  if (operands.size() >= 14) {
    auto dims = extractDimensionsFromOperands(operands, 0, 14);
    
    int64_t n = dims[0];
    int64_t c = dims[1];
    int64_t h = dims[2];
    int64_t w = dims[3];
    int64_t kernel_h = dims[4];
    int64_t kernel_w = dims[5];
    int64_t stride_h = dims[10];
    int64_t stride_w = dims[11];
    
    // 计算输出尺寸（简化）
    int64_t out_h = (h - kernel_h) / stride_h + 1;
    int64_t out_w = (w - kernel_w) / stride_w + 1;
    
    // 池化操作：每个输出像素需要比较kernel_h * kernel_w个输入
    cost.computeOps = n * c * out_h * out_w * kernel_h * kernel_w;
    
    // 内存访问量
    cost.memoryAccess = (n * c * h * w +              // input
                        n * c * out_h * out_w         // output
                       ) * 4; // float32
    
    // 池化操作内存密集，带宽需求高
    cost.bandwidth = cost.memoryAccess * 0.8;
    
    // 主要受内存带宽限制
    cost.estimatedTime = cost.memoryAccess / (7e11) * 1000; // 700 GB/s
    
  } else {
    // 默认估算
    cost.computeOps = 5e7;
    cost.memoryAccess = 1e8;
    cost.bandwidth = cost.memoryAccess * 0.8;
    cost.estimatedTime = 0.2;
  }
  
  return cost;
}

OperationCost OperationCostEstimator::estimateCuDNNElementwiseCost(mlir::LLVM::CallOp callOp) {
  OperationCost cost;
  cost.type = OperationType::CUDNN_ELEMENTWISE;
  cost.characteristic = ComputeCharacteristic::MEMORY_INTENSIVE;
  cost.efficiency = 1.0;
  
  auto funcName = callOp.getCallee().value_or("");
  auto operands = callOp.getOperands();
  
  // 大多数cuDNN元素级操作的参数格式:
  // (inputA, inputB, output, n, c, h, w, stream) 或类似
  if (operands.size() >= 7) {
    // 从参数中提取张量维度 - 通常在后面几个参数中
    size_t dimStart = operands.size() - 5; // 倒数第5个开始通常是维度参数
    auto dims = extractDimensionsFromOperands(operands, dimStart, 4);
    
    int64_t n = dims[0];
    int64_t c = dims[1]; 
    int64_t h = dims[2];
    int64_t w = dims[3];
    
    int64_t numElements = n * c * h * w;
    
    // 元素级操作: 每个元素1-2次运算
    if (funcName.contains("Add") || funcName.contains("Sub") || funcName.contains("Mul")) {
      cost.computeOps = numElements; // 每元素1次运算
      cost.memoryAccess = numElements * 3 * 4; // 2输入+1输出
    } else if (funcName.contains("Neg")) {
      cost.computeOps = numElements;
      cost.memoryAccess = numElements * 2 * 4; // 1输入+1输出
    } else if (funcName.contains("AddScalar") || funcName.contains("MulScalar")) {
      cost.computeOps = numElements;
      cost.memoryAccess = numElements * 2 * 4 + 4; // 1输入+1输出+1标量
    }
    
    // 元素级操作是内存密集型，带宽需求很高
    cost.bandwidth = cost.memoryAccess * 0.9;
    
    // 主要受内存带宽限制
    cost.estimatedTime = cost.memoryAccess / (8e11) * 1000; // 800 GB/s
    
    LLVM_DEBUG(llvm::dbgs() << "Elementwise cost estimation: "
               << "Elements=" << numElements
               << ", Memory=" << cost.memoryAccess 
               << ", Time=" << cost.estimatedTime << "ms\n");
  } else {
    // 默认估算
    cost.computeOps = 1e6;
    cost.memoryAccess = 1e7;
    cost.bandwidth = cost.memoryAccess * 0.9;
    cost.estimatedTime = 0.1;
  }
  
  return cost;
}

OperationCost OperationCostEstimator::estimateCuBLASCost(mlir::LLVM::CallOp callOp) {
  OperationCost cost;
  cost.efficiency = 1.0; // cuBLAS库函数效率高
  
  auto funcName = callOp.getCallee().value_or("");
  auto operands = callOp.getOperands();
  
  if (funcName.contains("FullyConnected")) {
    cost.type = OperationType::CUBLAS_FC;
    cost.characteristic = ComputeCharacteristic::COMPUTE_INTENSIVE;
    
    // mgpuCulibsFullyConnectedForward参数:
    // (batch_size, input_features, output_features, input_data, weight_data, bias_data, output_data, stream)
    if (operands.size() >= 3) {
      auto dims = extractDimensionsFromOperands(operands, 0, 3);
      
      int64_t batch_size = dims[0];
      int64_t input_features = dims[1]; 
      int64_t output_features = dims[2];
      
      // GEMM操作: C = A * B^T, 其中A是输入，B^T是权重转置
      cost.computeOps = batch_size * input_features * output_features * 2.0; // MAC
      
      cost.memoryAccess = (batch_size * input_features +          // input
                          input_features * output_features +     // weights
                          output_features +                      // bias (optional)
                          batch_size * output_features           // output
                         ) * 4; // float32
      
      // GEMM有很好的数据重用性，内存带宽需求相对较低
      cost.bandwidth = cost.memoryAccess * 0.3;
      
      // cuBLAS GEMM效率很高
      double computeTime = cost.computeOps / (3e12) * 1000; // 3 TFLOPS
      double memoryTime = cost.memoryAccess / (6e11) * 1000; // 600 GB/s
      cost.estimatedTime = std::max(computeTime, memoryTime);
      
      LLVM_DEBUG(llvm::dbgs() << "FC cost estimation: "
                 << "FLOPs=" << cost.computeOps 
                 << ", Memory=" << cost.memoryAccess 
                 << ", Time=" << cost.estimatedTime << "ms\n");
    }
  } else if (funcName.contains("FlattenFullyConnected")) {
    cost.type = OperationType::CUBLAS_FC;
    cost.characteristic = ComputeCharacteristic::COMPUTE_INTENSIVE;
    
    // mgpuCulibsFlattenFullyConnectedForward参数:
    // (batch_size, input_channels, input_height, input_width, output_features, ...)
    if (operands.size() >= 5) {
      auto dims = extractDimensionsFromOperands(operands, 0, 5);
      
      int64_t batch_size = dims[0];
      int64_t input_channels = dims[1];
      int64_t input_height = dims[2];
      int64_t input_width = dims[3];
      int64_t output_features = dims[4];
      
      int64_t flattened_features = input_channels * input_height * input_width;
      
      cost.computeOps = batch_size * flattened_features * output_features * 2.0;
      cost.memoryAccess = (batch_size * flattened_features +
                          flattened_features * output_features +
                          output_features +
                          batch_size * output_features) * 4;
      
      cost.bandwidth = cost.memoryAccess * 0.3;
      
      double computeTime = cost.computeOps / (3e12) * 1000;
      double memoryTime = cost.memoryAccess / (6e11) * 1000;
      cost.estimatedTime = std::max(computeTime, memoryTime);
    }
  }
  
  return cost;
}

OperationType OperationCostEstimator::getOperationType(mlir::Operation* op) {
  if (mlir::isa<mlir::gpu::LaunchFuncOp>(op)) {
    return OperationType::GPU_KERNEL;
  }
  
  if (auto callOp = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
    auto funcName = callOp.getCallee().value_or("");
    
    if (funcName.starts_with("mgpuCudnn")) {
      return classifyCuDNNOperation(funcName);
    } else if (funcName.starts_with("mgpuCulibs")) {
      return classifyCuBLASOperation(funcName);
    }
  }
  
  return OperationType::UNKNOWN;
}

std::vector<int64_t> OperationCostEstimator::extractTensorShape(mlir::LLVM::CallOp callOp) {
  std::vector<int64_t> shape;
  auto operands = callOp.getOperands();
  auto funcName = callOp.getCallee().value_or("");
  
  // 根据函数名确定张量形状参数的位置
  if (funcName.contains("Conv2d")) {
    // Conv2D: (n, c, h, w, ...)
    if (operands.size() >= 4) {
      auto dims = extractDimensionsFromOperands(operands, 0, 4);
      shape = {dims[0], dims[1], dims[2], dims[3]};
    }
  } else if (funcName.contains("FullyConnected")) {
    // FC: (batch_size, input_features, output_features, ...)
    if (operands.size() >= 3) {
      auto dims = extractDimensionsFromOperands(operands, 0, 3);
      shape = {dims[0], dims[1], dims[2]};
    }
  } else if (funcName.contains("Add") || funcName.contains("Mul") || funcName.contains("Sub")) {
    // Elementwise: (..., n, c, h, w, stream)
    if (operands.size() >= 5) {
      size_t dimStart = operands.size() - 5;
      auto dims = extractDimensionsFromOperands(operands, dimStart, 4);
      shape = {dims[0], dims[1], dims[2], dims[3]};
    }
  }
  
  return shape;
}

mlir::gpu::GPUFuncOp OperationCostEstimator::findGPUFunction(mlir::gpu::LaunchFuncOp launchOp, mlir::ModuleOp moduleOp) {
  auto kernelModuleName = launchOp.getKernelModuleName();
  auto kernelName = launchOp.getKernelName();
  
  // 查找GPU模块
  mlir::gpu::GPUModuleOp gpuModule = nullptr;
  moduleOp.walk([&](mlir::gpu::GPUModuleOp module) {
    if (module.getName() == kernelModuleName) {
      gpuModule = module;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  
  if (!gpuModule) {
    LLVM_DEBUG(llvm::dbgs() << "GPU module not found: " << kernelModuleName << "\n");
    return nullptr;
  }
  
  // 查找GPU函数
  for (auto func : gpuModule.getOps<mlir::gpu::GPUFuncOp>()) {
    if (func.getName() == kernelName) {
      LLVM_DEBUG(llvm::dbgs() << "Found GPU function: " << kernelName << "\n");
      return func;
    }
  }
  
  LLVM_DEBUG(llvm::dbgs() << "GPU function not found: " << kernelName << "\n");
  return nullptr;
}

OperationCost OperationCostEstimator::analyzeGPUFunction(mlir::gpu::GPUFuncOp funcOp, 
                                                         const std::vector<int64_t>& gridDim,
                                                         const std::vector<int64_t>& blockDim) {
  OperationCost cost;
  cost.type = OperationType::GPU_KERNEL;
  
  // 计算总线程数
  int64_t totalThreads = 1;
  for (int i = 0; i < 3; i++) {
    totalThreads *= gridDim[i] * blockDim[i];
  }
  
  // 分析函数体
  unsigned numLoads = 0, numStores = 0, numArithOps = 0, numControlOps = 0;
  bool hasLoops = false;
  unsigned loopDepth = 0;
  unsigned maxLoopDepth = 0;
  unsigned currentDepth = 0;
  
  // 估算循环迭代次数
  std::vector<int64_t> loopBounds;
  
  funcOp.walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::memref::LoadOp>(op)) {
      numLoads++;
    } else if (mlir::isa<mlir::memref::StoreOp>(op)) {
      numStores++;
    } else if (op->getDialect() && 
               op->getDialect()->getNamespace() == "arith") {
      numArithOps++;
    } else if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      hasLoops = true;
      currentDepth++;
      loopDepth++;
      maxLoopDepth = std::max(maxLoopDepth, currentDepth);
      numControlOps++;
      
      // 尝试估算循环边界
      auto lowerBound = extractConstantInt(forOp.getLowerBound());
      auto upperBound = extractConstantInt(forOp.getUpperBound());
      auto step = extractConstantInt(forOp.getStep());
      
      if (lowerBound && upperBound && step && *step > 0) {
        int64_t iterations = (*upperBound - *lowerBound + *step - 1) / *step;
        loopBounds.push_back(iterations);
      } else {
        loopBounds.push_back(10); // 默认估算
      }
    }
    
    // 检查循环结束
    if (mlir::isa<mlir::scf::YieldOp>(op) && currentDepth > 0) {
      currentDepth--;
    }
  });
  
  // 估算每个线程的操作数
  double opsPerThread = numArithOps;
  if (hasLoops) {
    // 计算循环复杂度
    double loopComplexity = 1.0;
    for (auto bound : loopBounds) {
      loopComplexity *= bound;
    }
    opsPerThread *= loopComplexity;
  }
  
  cost.computeOps = totalThreads * opsPerThread;
  cost.memoryAccess = totalThreads * (numLoads + numStores) * 4; // 假设float32
  
  // 判断是否为内存密集型
  double computeToMemoryRatio = cost.computeOps / (cost.memoryAccess / 4.0 + 1e-9);
  if (computeToMemoryRatio < 1.0) {
    cost.characteristic = ComputeCharacteristic::MEMORY_INTENSIVE;
    cost.bandwidth = cost.memoryAccess * 0.8;
  } else if (computeToMemoryRatio > 10.0) {
    cost.characteristic = ComputeCharacteristic::COMPUTE_INTENSIVE;
    cost.bandwidth = cost.memoryAccess * 0.2;
  } else {
    cost.characteristic = ComputeCharacteristic::BALANCED;
    cost.bandwidth = cost.memoryAccess * 0.5;
  }
  
  // 估算执行时间
  double computeTime = cost.computeOps / (1e11) * 1000; // 100 GFLOPS
  double memoryTime = cost.memoryAccess / (1e9) * 1000; // 1 GB/s
  cost.estimatedTime = std::max(computeTime, memoryTime);
  
  LLVM_DEBUG(llvm::dbgs() << "GPU Function analysis: "
             << "threads=" << totalThreads
             << ", loads=" << numLoads
             << ", stores=" << numStores
             << ", arithOps=" << numArithOps
             << ", loops=" << loopDepth
             << ", computeOps=" << cost.computeOps
             << ", memoryAccess=" << cost.memoryAccess
             << ", estimatedTime=" << cost.estimatedTime << "ms\n");
  
  return cost;
}

//===----------------------------------------------------------------------===//
// ParallelGroupIdentifier Implementation  
//===----------------------------------------------------------------------===//

std::vector<ParallelGroup> ParallelGroupIdentifier::identifyParallelGroups(mlir::LLVM::LLVMFuncOp funcOp) {
  std::vector<ParallelGroup> groups;
  ParallelGroup currentGroup;
  bool inGroup = false;
  
  // 用于跟踪stream状态
  llvm::DenseMap<mlir::Value, bool> streamInUse;
  
  funcOp.walk([&](mlir::Operation* op) {
    LLVM_DEBUG(llvm::dbgs() << "Processing operation: " << op->getName().getStringRef() << "\n");
    
    // 检查是否是stream同步和释放（上一组结束）
    if (isStreamSync(op) || isStreamRelease(op)) {
      auto stream = getStreamFromOperation(op);
      if (stream && streamInUse[stream]) {
        streamInUse[stream] = false;
        
        // 检查是否所有stream都已完成
        bool allStreamsDone = true;
        for (auto& pair : streamInUse) {
          if (pair.second) {
            allStreamsDone = false;
            break;
          }
        }
        
        if (allStreamsDone && inGroup && !currentGroup.operations.empty()) {
          // 结束当前组
          LLVM_DEBUG(llvm::dbgs() << "Ending parallel group with " 
                    << currentGroup.operations.size() << " operations\n");
          groups.push_back(std::move(currentGroup));
          currentGroup = ParallelGroup();
          inGroup = false;
          streamInUse.clear();
        }
      }
      return mlir::WalkResult::advance();
    }
    
    // 检查是否是stream获取（新组开始或添加stream）
    if (isStreamAcquire(op)) {
      auto stream = getStreamFromOperation(op);
      if (stream) {
        if (!inGroup) {
          inGroup = true;
          LLVM_DEBUG(llvm::dbgs() << "Starting new parallel group\n");
        }
        currentGroup.streams.push_back(stream);
        streamInUse[stream] = true;
        LLVM_DEBUG(llvm::dbgs() << "Added stream to group\n");
      }
      return mlir::WalkResult::advance();
    }
    
    // 检查是否是handle获取
    if (isHandleAcquire(op)) {
      LLVM_DEBUG(llvm::dbgs() << "Handle acquire operation\n");
      return mlir::WalkResult::advance();
    }
    
    // 如果在组内，检查是否是计算操作
    if (inGroup) {
      bool isComputeOp = false;
      mlir::Value opStream = nullptr;
      
      if (auto launchOp = mlir::dyn_cast<mlir::gpu::LaunchFuncOp>(op)) {
        isComputeOp = true;
        opStream = getStreamFromOperation(op);
        LLVM_DEBUG(llvm::dbgs() << "Found GPU launch operation\n");
      } else if (auto callOp = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
        auto funcName = callOp.getCallee().value_or("");
        if (funcName.starts_with("mgpuCudnn") || funcName.starts_with("mgpuCulibs")) {
          isComputeOp = true;
          opStream = getStreamFromOperation(op);
          LLVM_DEBUG(llvm::dbgs() << "Found library function call: " << funcName << "\n");
        }
      }
      
      if (isComputeOp) {
        auto operation = std::make_unique<ParallelGroupOperation>(op);
        operation->stream = opStream;
        
        // 设置函数名
        if (auto callOp = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
          operation->functionName = callOp.getCallee().value_or("").str();
        } else if (auto launchOp = mlir::dyn_cast<mlir::gpu::LaunchFuncOp>(op)) {
          operation->functionName = launchOp.getKernelName().str();
        }
        
        // 提取张量形状
        if (auto callOp = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
          operation->shape = OperationCostEstimator::extractTensorShape(callOp);
        }
        
        currentGroup.operations.push_back(std::move(operation));
        LLVM_DEBUG(llvm::dbgs() << "Added compute operation to group\n");
      }
    }
    
    return mlir::WalkResult::advance();
  });
  
  // 添加最后一个组（如果存在）
  if (inGroup && !currentGroup.operations.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Adding final parallel group with " 
              << currentGroup.operations.size() << " operations\n");
    groups.push_back(std::move(currentGroup));
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Identified " << groups.size() << " parallel groups\n");
  return groups;
}

bool ParallelGroupIdentifier::isStreamAcquire(mlir::Operation* op) {
  if (auto callOp = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
    auto funcName = callOp.getCallee().value_or("");
    return funcName == "mgpuAcquirePooledStream";
  }
  return false;
}

bool ParallelGroupIdentifier::isStreamRelease(mlir::Operation* op) {
  if (auto callOp = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
    auto funcName = callOp.getCallee().value_or("");
    return funcName == "mgpuReleasePooledStream";
  }
  return false;
}

bool ParallelGroupIdentifier::isStreamSync(mlir::Operation* op) {
  if (auto callOp = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
    auto funcName = callOp.getCallee().value_or("");
    return funcName == "mgpuStreamSynchronize";
  }
  return false;
}

bool ParallelGroupIdentifier::isHandleAcquire(mlir::Operation* op) {
  if (auto callOp = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
    auto funcName = callOp.getCallee().value_or("");
    return funcName == "mgpuAcquirePooledHandles";
  }
  return false;
}

mlir::Value ParallelGroupIdentifier::getStreamFromOperation(mlir::Operation* op) {
  if (auto callOp = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
    auto funcName = callOp.getCallee().value_or("");
    
    if (funcName == "mgpuAcquirePooledStream") {
      // 返回值就是stream
      return callOp.getResult();
    } else if (funcName == "mgpuAcquirePooledHandles" || 
               funcName == "mgpuStreamSynchronize" || 
               funcName == "mgpuReleasePooledStream") {
      // 第一个参数是stream
      if (!callOp.getOperands().empty()) {
        return callOp.getOperands()[0];
      }
    } else if (funcName.starts_with("mgpuCudnn") || funcName.starts_with("mgpuCulibs")) {
      // 最后一个参数是stream
      if (!callOp.getOperands().empty()) {
        return callOp.getOperands().back();
      }
    }
  } else if (auto launchOp = mlir::dyn_cast<mlir::gpu::LaunchFuncOp>(op)) {
    // gpu.launch_func的异步token参数
    if (launchOp.getAsyncToken()) {
      return launchOp.getAsyncToken();
    }
  }
  
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ScheduleOptimizer Implementation
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ScheduleOptimizer Implementation - 修复版本
//===----------------------------------------------------------------------===//

std::vector<ScheduleUnit> ScheduleOptimizer::optimizeParallelGroup(
    ParallelGroup& group, double toleranceFactor, bool enableTypeGrouping) {
  
  // 输入验证
  if (group.operations.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Empty group, returning empty schedule\n");
    return {};
  }
  
  // 首先计算每个操作的成本
  for (auto& operation : group.operations) {
    if (!operation || !operation->op) {
      LLVM_DEBUG(llvm::dbgs() << "Warning: Invalid operation in group\n");
      continue;
    }
    
    operation->cost = OperationCostEstimator::estimateOperationCost(
        operation->op, operation->op->getParentOfType<mlir::ModuleOp>());
    
    group.totalCost += operation->cost.estimatedTime;
    group.maxCost = std::max(group.maxCost, operation->cost.estimatedTime);
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Group cost analysis: total=" << group.totalCost 
            << ", max=" << group.maxCost << "\n");
  
  // 将操作转换为指针向量以便处理，同时验证有效性
  std::vector<ParallelGroupOperation*> operations;
  for (auto& op : group.operations) {
    if (op && op->op && op->stream) {  // 确保操作和stream都有效
      operations.push_back(op.get());
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Warning: Skipping invalid operation (op=" 
                << (op ? "valid" : "null") << ", stream=" 
                << (op && op->stream ? "valid" : "invalid") << ")\n");
    }
  }
  
  if (operations.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No valid operations after filtering\n");
    return {};
  }
  
  // 使用新的智能装箱算法
  std::vector<ScheduleUnit> units = intelligentBinPacking(operations, group.maxCost, toleranceFactor);
  
  LLVM_DEBUG(llvm::dbgs() << "Intelligent packing result: " << units.size() 
            << " units from " << operations.size() << " original operations\n");
  
  return units;
}

std::vector<ScheduleUnit> ScheduleOptimizer::intelligentBinPacking(
    const std::vector<ParallelGroupOperation*>& operations,
    double maxExecutionTime, double toleranceFactor) {
  
  if (operations.empty() || maxExecutionTime <= 0) {
    LLVM_DEBUG(llvm::dbgs() << "Invalid parameters for bin packing\n");
    return {};
  }
  
  // 计算时间上限（考虑一定的容忍因子，但要保守）
  double timeLimit = maxExecutionTime * std::min(toleranceFactor, 1.05); // 最多5%的延迟
  
  LLVM_DEBUG(llvm::dbgs() << "Bin packing with time limit: " << timeLimit << "ms\n");
  
  // 创建带优先级的操作列表，添加严格的验证
  std::vector<KernelInfo> kernelInfos;
  for (auto* op : operations) {
    if (!op || !op->op || !op->stream) {  // 严格验证
      LLVM_DEBUG(llvm::dbgs() << "Warning: Skipping invalid operation in bin packing\n");
      continue;
    }
    
    KernelInfo info;
    info.operation = op;
    info.executionTime = op->cost.estimatedTime;
    info.characteristic = op->cost.characteristic;
    info.priority = calculateKernelPriority(op);
    kernelInfos.push_back(info);
  }
  
  if (kernelInfos.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No valid kernels for packing\n");
    return {};
  }
  
  // 按执行时间降序排序（First Fit Decreasing策略）
  std::sort(kernelInfos.begin(), kernelInfos.end(), 
           [](const KernelInfo& a, const KernelInfo& b) {
             return a.executionTime > b.executionTime;
           });
  
  std::vector<ScheduleUnit> units;
  
  for (const auto& kernelInfo : kernelInfos) {
    bool placed = false;
    
    // 尝试放入现有的compatible unit
    for (auto& unit : units) {
      if (canAddToUnit(unit, kernelInfo, timeLimit)) {
        addKernelToUnit(unit, kernelInfo);
        placed = true;
        LLVM_DEBUG(llvm::dbgs() << "Added kernel (" << kernelInfo.executionTime 
                  << "ms) to existing unit, new total: " << unit.totalCost << "ms\n");
        break;
      }
    }
    
    // 如果无法放入现有unit，创建新unit
    if (!placed) {
      ScheduleUnit newUnit = createUnitFromKernel(kernelInfo);
      units.push_back(newUnit);
      LLVM_DEBUG(llvm::dbgs() << "Created new unit for kernel (" 
                << kernelInfo.executionTime << "ms)\n");
    }
  }
  
  // 后处理：尝试进一步优化
  units = postProcessOptimization(units, timeLimit);
  
  // 验证最终结果的有效性
  for (size_t i = 0; i < units.size(); i++) {
    auto& unit = units[i];
    for (size_t j = 0; j < unit.operations.size(); j++) {
      if (!unit.operations[j] || !unit.operations[j]->op || !unit.operations[j]->stream) {
        LLVM_DEBUG(llvm::dbgs() << "Error: Invalid operation in final unit " << i 
                  << " at position " << j << "\n");
        // 移除无效操作
        unit.operations.erase(unit.operations.begin() + j);
        j--; // 调整索引
      }
    }
    
    // 如果unit变空了，标记为删除
    if (unit.operations.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "Warning: Unit " << i << " became empty after validation\n");
    }
  }
  
  // 移除空的units
  units.erase(std::remove_if(units.begin(), units.end(), 
                            [](const ScheduleUnit& unit) { 
                              return unit.operations.empty(); 
                            }), 
              units.end());
  
  return units;
}

double ScheduleOptimizer::calculateKernelPriority(ParallelGroupOperation* op) {
  if (!op) return 0.0;  // 添加空值检查
  
  double priority = 0.0;
  
  // 基础优先级基于执行时间（短的kernel优先合并）
  priority += (1.0 / (op->cost.estimatedTime + 0.001)) * 100;
  
  // 计算特性加权
  switch (op->cost.characteristic) {
    case ComputeCharacteristic::MEMORY_INTENSIVE:
      priority += 50; // 内存密集型优先合并（减少内存竞争）
      break;
    case ComputeCharacteristic::BALANCED:
      priority += 30;
      break;
    case ComputeCharacteristic::COMPUTE_INTENSIVE:
      priority += 20; // 计算密集型较少合并（保持并行计算能力）
      break;
  }
  
  // 操作类型加权
  switch (op->cost.type) {
    case OperationType::CUDNN_ELEMENTWISE:
      priority += 40; // 元素级操作优先合并
      break;
    case OperationType::MEMORY_OP:
      priority += 35;
      break;
    case OperationType::CUDNN_POOL:
      priority += 25;
      break;
    case OperationType::CUDNN_CONV:
    case OperationType::CUBLAS_GEMM:
    case OperationType::CUBLAS_FC:
      priority += 10; // 重计算操作较少合并
      break;
    default:
      priority += 15;
      break;
  }
  
  return priority;
}

bool ScheduleOptimizer::canAddToUnit(const ScheduleUnit& unit, 
                                    const KernelInfo& kernelInfo, 
                                    double timeLimit) {
  // 输入验证
  if (!kernelInfo.operation || !kernelInfo.operation->op || !kernelInfo.operation->stream) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot add invalid kernel to unit\n");
    return false;
  }
  
  // 1. 检查时间约束
  double newTotalTime = unit.totalCost + kernelInfo.executionTime;
  if (newTotalTime > timeLimit) {
    return false;
  }
  
  // 2. 检查特性兼容性
  if (!areCharacteristicsCompatible(unit.characteristic, kernelInfo.characteristic)) {
    return false;
  }
  
  // 3. 检查unit大小限制（避免单个stream负载过重）
  if (unit.operations.size() >= 4) { // 限制每个stream最多4个kernel
    return false;
  }
  
  // 4. 检查内存访问模式兼容性
  if (!areMemoryPatternsCompatible(unit, kernelInfo)) {
    return false;
  }
  
  return true;
}

bool ScheduleOptimizer::areCharacteristicsCompatible(ComputeCharacteristic unitChar, 
                                                    ComputeCharacteristic kernelChar) {
  // 完全相同的特性最兼容
  if (unitChar == kernelChar) {
    return true;
  }
  
  // BALANCED 可以与任何特性兼容
  if (unitChar == ComputeCharacteristic::BALANCED || 
      kernelChar == ComputeCharacteristic::BALANCED) {
    return true;
  }
  
  // COMPUTE_INTENSIVE 和 MEMORY_INTENSIVE 不兼容
  // 因为它们会竞争不同的资源
  return false;
}

bool ScheduleOptimizer::areMemoryPatternsCompatible(const ScheduleUnit& unit, 
                                                   const KernelInfo& kernelInfo) {
  // 简单的启发式规则：
  // 1. 如果unit中已有大内存操作，避免添加更多大内存操作
  double unitMemoryIntensity = 0.0;
  for (auto* op : unit.operations) {
    if (op && op->cost.memoryAccess > 100 * 1024 * 1024) { // 100MB
      unitMemoryIntensity += 1.0;
    }
  }
  
  if (unitMemoryIntensity >= 2.0 && 
      kernelInfo.operation->cost.memoryAccess > 50 * 1024 * 1024) {
    return false;
  }
  
  return true;
}

void ScheduleOptimizer::addKernelToUnit(ScheduleUnit& unit, const KernelInfo& kernelInfo) {
  // 验证输入
  if (!kernelInfo.operation || !kernelInfo.operation->op || !kernelInfo.operation->stream) {
    LLVM_DEBUG(llvm::dbgs() << "Error: Cannot add invalid kernel to unit\n");
    return;
  }
  
  unit.operations.push_back(kernelInfo.operation);
  unit.totalCost += kernelInfo.executionTime;
  unit.maxCost = std::max(unit.maxCost, kernelInfo.executionTime);
  unit.canBeSerial = true; // 多个操作意味着可以串行
  
  // 更新主导特性（选择更严格的特性）
  if (kernelInfo.characteristic == ComputeCharacteristic::COMPUTE_INTENSIVE ||
      (unit.characteristic == ComputeCharacteristic::BALANCED && 
       kernelInfo.characteristic != ComputeCharacteristic::BALANCED)) {
    unit.characteristic = kernelInfo.characteristic;
  }
}

ScheduleUnit ScheduleOptimizer::createUnitFromKernel(const KernelInfo& kernelInfo) {
  // 验证输入
  if (!kernelInfo.operation || !kernelInfo.operation->op || !kernelInfo.operation->stream) {
    LLVM_DEBUG(llvm::dbgs() << "Error: Cannot create unit from invalid kernel\n");
    return ScheduleUnit(); // 返回空单元
  }
  
  ScheduleUnit unit;
  unit.operations.push_back(kernelInfo.operation);
  unit.dominantType = kernelInfo.operation->cost.type;
  unit.characteristic = kernelInfo.characteristic;
  unit.totalCost = kernelInfo.executionTime;
  unit.maxCost = kernelInfo.executionTime;
  unit.canBeSerial = false; // 单个操作
  return unit;
}

std::vector<ScheduleUnit> ScheduleOptimizer::postProcessOptimization(
    std::vector<ScheduleUnit> units, double timeLimit) {
  
  // 尝试合并小的units
  std::vector<ScheduleUnit> optimizedUnits;
  std::vector<bool> merged(units.size(), false);
  
  for (size_t i = 0; i < units.size(); i++) {
    if (merged[i] || units[i].operations.empty()) continue;
    
    ScheduleUnit currentUnit = units[i];
    
    // 如果当前unit比较小，尝试与其他小unit合并
    if (currentUnit.totalCost < timeLimit * 0.5) {
      for (size_t j = i + 1; j < units.size(); j++) {
        if (merged[j] || units[j].operations.empty()) continue;
        
        double combinedCost = currentUnit.totalCost + units[j].totalCost;
        if (combinedCost <= timeLimit && 
            areCharacteristicsCompatible(currentUnit.characteristic, units[j].characteristic) &&
            (currentUnit.operations.size() + units[j].operations.size()) <= 4) {
          
          LLVM_DEBUG(llvm::dbgs() << "Post-optimization: merging units with costs " 
                    << currentUnit.totalCost << " + " << units[j].totalCost 
                    << " = " << combinedCost << "\n");
          
          // 验证要合并的操作都是有效的
          bool validMerge = true;
          for (auto* op : units[j].operations) {
            if (!op || !op->op || !op->stream) {
              validMerge = false;
              break;
            }
          }
          
          if (validMerge) {
            // 合并units
            currentUnit.operations.insert(currentUnit.operations.end(),
                                        units[j].operations.begin(), units[j].operations.end());
            currentUnit.totalCost = combinedCost;
            currentUnit.maxCost = std::max(currentUnit.maxCost, units[j].maxCost);
            currentUnit.canBeSerial = true;
            merged[j] = true;
          }
        }
      }
    }
    
    optimizedUnits.push_back(currentUnit);
    merged[i] = true;
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Post-optimization: " << optimizedUnits.size() 
            << " units from " << units.size() << " units\n");
  
  return optimizedUnits;
}

// std::vector<ScheduleUnit> ScheduleOptimizer::optimizeParallelGroup(
//     ParallelGroup& group, double toleranceFactor, bool enableTypeGrouping) {
  
//   // 首先计算每个操作的成本
//   for (auto& operation : group.operations) {
//     operation->cost = OperationCostEstimator::estimateOperationCost(
//         operation->op, operation->op->getParentOfType<mlir::ModuleOp>());
    
//     group.totalCost += operation->cost.estimatedTime;
//     group.maxCost = std::max(group.maxCost, operation->cost.estimatedTime);
//   }
  
//   LLVM_DEBUG(llvm::dbgs() << "Group cost analysis: total=" << group.totalCost 
//             << ", max=" << group.maxCost << "\n");
  
//   // 将操作转换为指针向量以便处理
//   std::vector<ParallelGroupOperation*> operations;
//   for (auto& op : group.operations) {
//     operations.push_back(op.get());
//   }
  
//   // 按特性分组
//   std::vector<ScheduleUnit> units;
//   if (enableTypeGrouping) {
//     units = groupByCharacteristic(operations);
//     LLVM_DEBUG(llvm::dbgs() << "Grouped by characteristic into " << units.size() << " units\n");
//   } else {
//     // 每个操作独立成一个单元
//     for (auto* op : operations) {
//       ScheduleUnit unit;
//       unit.operations.push_back(op);
//       unit.dominantType = op->cost.type;
//       unit.characteristic = op->cost.characteristic;
//       unit.totalCost = op->cost.estimatedTime;
//       unit.maxCost = op->cost.estimatedTime;
//       units.push_back(unit);
//     }
//     LLVM_DEBUG(llvm::dbgs() << "Created " << units.size() << " individual units\n");
//   }
  
//   // 尝试合并可以串行执行的单元
//   std::vector<ScheduleUnit> optimizedUnits;
//   std::vector<bool> merged(units.size(), false);
  
//   for (size_t i = 0; i < units.size(); i++) {
//     if (merged[i]) continue;
    
//     ScheduleUnit currentUnit = units[i];
    
//     // 尝试与后续单元合并
//     for (size_t j = i + 1; j < units.size(); j++) {
//       if (merged[j]) continue;
      
//       if (canMergeUnits(currentUnit, units[j], group.maxCost, toleranceFactor)) {
//         LLVM_DEBUG(llvm::dbgs() << "Merging unit " << i << " with unit " << j << "\n");
//         currentUnit = mergeUnits(currentUnit, units[j]);
//         merged[j] = true;
//       }
//     }
    
//     optimizedUnits.push_back(currentUnit);
//     merged[i] = true;
//   }
  
//   LLVM_DEBUG(llvm::dbgs() << "Optimization result: " << optimizedUnits.size() 
//             << " units from " << units.size() << " original units\n");
  
//   return optimizedUnits;
// }

// std::vector<ScheduleUnit> ScheduleOptimizer::groupByCharacteristic(
//     const std::vector<ParallelGroupOperation*>& operations) {
  
//   std::map<ComputeCharacteristic, ScheduleUnit> groups;
  
//   for (auto* op : operations) {
//     auto characteristic = op->cost.characteristic;
    
//     if (groups.find(characteristic) == groups.end()) {
//       groups[characteristic] = ScheduleUnit();
//       groups[characteristic].characteristic = characteristic;
//     }
    
//     groups[characteristic].operations.push_back(op);
//     groups[characteristic].totalCost += op->cost.estimatedTime;
//     groups[characteristic].maxCost = std::max(groups[characteristic].maxCost, 
//                                              op->cost.estimatedTime);
    
//     // 设置主导类型
//     if (groups[characteristic].operations.size() == 1) {
//       groups[characteristic].dominantType = op->cost.type;
//     }
//   }
  
//   std::vector<ScheduleUnit> result;
//   for (auto& pair : groups) {
//     LLVM_DEBUG(llvm::dbgs() << "Characteristic group " << static_cast<int>(pair.first)
//               << " has " << pair.second.operations.size() << " operations\n");
//     result.push_back(pair.second);
//   }
  
//   return result;
// }

// bool ScheduleOptimizer::canMergeUnits(const ScheduleUnit& unit1, const ScheduleUnit& unit2, 
//                                      double maxCost, double toleranceFactor) {
//   // 检查特性兼容性
//   if (unit1.characteristic != unit2.characteristic) {
//     LLVM_DEBUG(llvm::dbgs() << "Cannot merge: different characteristics\n");
//     return false;
//   }
  
//   // 检查合并后的成本是否在容忍范围内
//   double mergedCost = unit1.totalCost + unit2.totalCost;
//   double parallelOverhead = calculateParallelOverhead(unit1.operations) + 
//                            calculateParallelOverhead(unit2.operations);
//   double serialOverhead = calculateSerialOverhead(unit1.operations) + 
//                          calculateSerialOverhead(unit2.operations);
  
//   double thresholdCost = maxCost * toleranceFactor + parallelOverhead;
//   double actualCost = mergedCost + serialOverhead;
  
//   LLVM_DEBUG(llvm::dbgs() << "Merge check: mergedCost=" << mergedCost
//             << ", parallelOverhead=" << parallelOverhead
//             << ", serialOverhead=" << serialOverhead  
//             << ", thresholdCost=" << thresholdCost
//             << ", actualCost=" << actualCost << "\n");
  
//   // 如果串行执行时间 + 串行开销 <= 最大并行时间 * 容忍因子，则可以合并
//   return actualCost <= thresholdCost;
// }

// ScheduleUnit ScheduleOptimizer::mergeUnits(const ScheduleUnit& unit1, const ScheduleUnit& unit2) {
//   ScheduleUnit merged;
  
//   merged.operations = unit1.operations;
//   merged.operations.insert(merged.operations.end(), 
//                           unit2.operations.begin(), unit2.operations.end());
  
//   merged.characteristic = unit1.characteristic;
//   merged.dominantType = unit1.dominantType;
//   merged.totalCost = unit1.totalCost + unit2.totalCost;
//   merged.maxCost = std::max(unit1.maxCost, unit2.maxCost);
//   merged.canBeSerial = true;
  
//   return merged;
// }

// double ScheduleOptimizer::calculateSerialOverhead(const std::vector<ParallelGroupOperation*>& operations) {
//   // 串行执行的开销主要是减少的stream创建/销毁开销
//   // 每个减少的stream节省大约0.1ms
//   return -(static_cast<double>(operations.size()) - 1.0) * 0.1; // 负数表示节省的开销
// }

// double ScheduleOptimizer::calculateParallelOverhead(const std::vector<ParallelGroupOperation*>& operations) {
//   // 并行执行的开销：stream创建、handle获取、同步等
//   // 每个stream大约0.1ms开销
//   return static_cast<double>(operations.size()) * 0.1;
// }

//===----------------------------------------------------------------------===//
// ScheduleRewriter Implementation
//===----------------------------------------------------------------------===//

void ScheduleRewriter::rewriteParallelGroup(ParallelGroup& group, 
                                            const std::vector<ScheduleUnit>& optimizedSchedule,
                                            mlir::OpBuilder& builder) {
  
  LLVM_DEBUG(llvm::dbgs() << "Rewriting parallel group with " << optimizedSchedule.size() 
            << " schedule units\n");
  
  // 为每个调度单元分配stream
  std::vector<mlir::Value> unitStreams;
  
  for (size_t unitIdx = 0; unitIdx < optimizedSchedule.size(); unitIdx++) {
    const auto& unit = optimizedSchedule[unitIdx];
    mlir::Value stream;
    
    if (unit.operations.size() == 1) {
      // 单个操作，使用原始stream
      stream = unit.operations[0]->stream;
      LLVM_DEBUG(llvm::dbgs() << "Unit " << unitIdx << ": using original stream\n");
    } else {
      // 多个操作合并，需要新的stream
      auto loc = unit.operations[0]->op->getLoc();
      
      // 使用第一个操作的stream作为合并后的stream
      stream = unit.operations[0]->stream;
      
      LLVM_DEBUG(llvm::dbgs() << "Unit " << unitIdx << ": merging " 
                << unit.operations.size() << " operations to same stream\n");
      
      // 更新所有操作的stream
      for (size_t opIdx = 1; opIdx < unit.operations.size(); opIdx++) {
        auto* op = unit.operations[opIdx];
        updateOperationStream(op->op, stream);
        LLVM_DEBUG(llvm::dbgs() << "  Updated operation " << opIdx 
                  << " to use merged stream\n");
      }
      
      // 移除多余的stream获取和释放操作
      removeRedundantStreamOps(unit.operations, stream, builder);
    }
    
    unitStreams.push_back(stream);
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Rewriting completed\n");
}

mlir::Value ScheduleRewriter::createOrReuseStream(mlir::OpBuilder& builder, mlir::Location loc) {
  // 创建新的stream获取调用
  auto streamType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto funcName = builder.getStringAttr("mgpuAcquirePooledStream");
  auto funcType = mlir::LLVM::LLVMFunctionType::get(streamType, {});
  
  return builder.create<mlir::LLVM::CallOp>(loc, funcType, funcName, mlir::ValueRange{}).getResult();
}

void ScheduleRewriter::insertStreamAcquire(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value stream) {
  // 注意：这里stream应该是已经通过createOrReuseStream创建的
}

void ScheduleRewriter::insertHandleAcquire(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value stream) {
  auto voidType = mlir::LLVM::LLVMVoidType::get(builder.getContext());
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto funcName = builder.getStringAttr("mgpuAcquirePooledHandles");
  auto funcType = mlir::LLVM::LLVMFunctionType::get(voidType, {ptrType});
  
  builder.create<mlir::LLVM::CallOp>(loc, funcType, funcName, mlir::ValueRange{stream});
}

void ScheduleRewriter::insertStreamSync(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value stream) {
  auto voidType = mlir::LLVM::LLVMVoidType::get(builder.getContext());
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto funcName = builder.getStringAttr("mgpuStreamSynchronize");
  auto funcType = mlir::LLVM::LLVMFunctionType::get(voidType, {ptrType});
  
  builder.create<mlir::LLVM::CallOp>(loc, funcType, funcName, mlir::ValueRange{stream});
}

void ScheduleRewriter::insertStreamRelease(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value stream) {
  auto voidType = mlir::LLVM::LLVMVoidType::get(builder.getContext());
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  auto funcName = builder.getStringAttr("mgpuReleasePooledStream");
  auto funcType = mlir::LLVM::LLVMFunctionType::get(voidType, {ptrType});
  
  builder.create<mlir::LLVM::CallOp>(loc, funcType, funcName, mlir::ValueRange{stream});
}

void ScheduleRewriter::updateOperationStream(mlir::Operation* op, mlir::Value newStream) {
  LLVM_DEBUG(llvm::dbgs() << "Updating operation stream\n");
  
  if (auto callOp = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
    auto funcName = callOp.getCallee().value_or("");
    if (funcName.starts_with("mgpuCudnn") || funcName.starts_with("mgpuCulibs")) {
      // 更新最后一个操作数（stream参数）
      auto operands = llvm::to_vector(callOp.getOperands());
      if (!operands.empty()) {
        operands.back() = newStream;
        callOp.getArgOperandsMutable().assign(operands);
        LLVM_DEBUG(llvm::dbgs() << "Updated library call stream parameter\n");
      }
    }
  } else if (auto launchOp = mlir::dyn_cast<mlir::gpu::LaunchFuncOp>(op)) {
    // 更新gpu.launch_func的stream参数
    if (launchOp.getAsyncToken()) {
      launchOp.getAsyncObjectMutable().assign(newStream);
      LLVM_DEBUG(llvm::dbgs() << "Updated GPU launch stream parameter\n");
    }
  }
}

// void ScheduleRewriter::removeRedundantStreamOps(const std::vector<ParallelGroupOperation*>& operations,
//                                                mlir::Value mergedStream, mlir::OpBuilder& builder) {
//   // 收集需要移除的stream操作
//   std::vector<mlir::Operation*> toRemove;
  
//   for (size_t i = 1; i < operations.size(); i++) {
//     auto* op = operations[i];
//     mlir::Value originalStream = op->stream;
    
//     // 查找并标记要删除的stream相关操作
//     op->op->getParentOfType<mlir::LLVM::LLVMFuncOp>().walk([&](mlir::Operation* walkOp) {
//       if (auto callOp = mlir::dyn_cast<mlir::LLVM::CallOp>(walkOp)) {
//         auto funcName = callOp.getCallee().value_or("");
        
//         // 检查是否是与该stream相关的操作
//         if (funcName == "mgpuAcquirePooledStream" && callOp.getResult() == originalStream) {
//           toRemove.push_back(walkOp);
//         } else if (funcName == "mgpuAcquirePooledHandles" || 
//                    funcName == "mgpuStreamSynchronize" || 
//                    funcName == "mgpuReleasePooledStream") {
//           if (!callOp.getOperands().empty() && callOp.getOperands()[0] == originalStream) {
//             toRemove.push_back(walkOp);
//           }
//         }
//       }
//     });
//   }
  
//   // 删除冗余操作
//   for (auto* op : toRemove) {
//     LLVM_DEBUG(llvm::dbgs() << "Removing redundant stream operation\n");
//     op->erase();
//   }
// }
void ScheduleRewriter::removeRedundantStreamOps(const std::vector<ParallelGroupOperation*>& operations,
                                               mlir::Value mergedStream, mlir::OpBuilder& builder) {
  // 收集需要删除的stream相关操作
  // 关键：我们要删除除第一个操作外所有操作对应的完整stream生命周期
  
  for (size_t i = 1; i < operations.size(); i++) {
    auto* op = operations[i];
    mlir::Value redundantStream = op->stream;
    
    // 收集与这个stream相关的所有操作
    std::vector<mlir::Operation*> streamAcquire;      // mgpuAcquirePooledStream
    std::vector<mlir::Operation*> handleAcquire;      // mgpuAcquirePooledHandles  
    std::vector<mlir::Operation*> streamSync;         // mgpuStreamSynchronize
    std::vector<mlir::Operation*> streamRelease;      // mgpuReleasePooledStream
    
    // 遍历函数中的所有操作，找到与该stream相关的操作
    op->op->getParentOfType<mlir::LLVM::LLVMFuncOp>().walk([&](mlir::Operation* walkOp) {
      if (auto callOp = mlir::dyn_cast<mlir::LLVM::CallOp>(walkOp)) {
        auto funcName = callOp.getCallee().value_or("");
        
        if (funcName == "mgpuAcquirePooledStream") {
          // 检查是否是我们要删除的stream的分配操作
          if (callOp.getResult() == redundantStream) {
            streamAcquire.push_back(walkOp);
            LLVM_DEBUG(llvm::dbgs() << "Found stream acquire operation for redundant stream\n");
          }
        } else if (funcName == "mgpuAcquirePooledHandles") {
          // 检查是否使用了要删除的stream
          if (!callOp.getOperands().empty() && callOp.getOperands()[0] == redundantStream) {
            handleAcquire.push_back(walkOp);
            LLVM_DEBUG(llvm::dbgs() << "Found handle acquire operation for redundant stream\n");
          }
        } else if (funcName == "mgpuStreamSynchronize") {
          // 检查是否同步了要删除的stream
          if (!callOp.getOperands().empty() && callOp.getOperands()[0] == redundantStream) {
            streamSync.push_back(walkOp);
            LLVM_DEBUG(llvm::dbgs() << "Found stream synchronize operation for redundant stream\n");
          }
        } else if (funcName == "mgpuReleasePooledStream") {
          // 检查是否释放了要删除的stream
          if (!callOp.getOperands().empty() && callOp.getOperands()[0] == redundantStream) {
            streamRelease.push_back(walkOp);
            LLVM_DEBUG(llvm::dbgs() << "Found stream release operation for redundant stream\n");
          }
        }
      }
    });
    
    // 按正确顺序删除操作：
    // 1. 先删除使用stream的操作（同步和释放）
    // 2. 再删除handle获取操作（如果有的话）
    // 3. 最后删除stream获取操作
    
    // 删除stream同步操作
    for (auto* syncOp : streamSync) {
      LLVM_DEBUG(llvm::dbgs() << "Removing redundant stream synchronize operation\n");
      syncOp->erase();
    }
    
    // 删除stream释放操作
    for (auto* releaseOp : streamRelease) {
      LLVM_DEBUG(llvm::dbgs() << "Removing redundant stream release operation\n");
      releaseOp->erase();
    }
    
    // 删除handle获取操作
    for (auto* handleOp : handleAcquire) {
      LLVM_DEBUG(llvm::dbgs() << "Removing redundant handle acquire operation\n");
      handleOp->erase();
    }
    
    // 最后删除stream获取操作
    for (auto* acquireOp : streamAcquire) {
      LLVM_DEBUG(llvm::dbgs() << "Removing redundant stream acquire operation\n");
      acquireOp->erase();
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Completed removal of all operations for redundant stream " << i << "\n");
  }
  
  LLVM_DEBUG(llvm::dbgs() << "Removed " << (operations.size() - 1) 
            << " redundant stream lifecycles\n");
}

//===----------------------------------------------------------------------===//
// KernelExecutionOptimizer Implementation (Business Logic)
//===----------------------------------------------------------------------===//

void KernelExecutionOptimizer::optimize(mlir::ModuleOp moduleOp, bool verbose) {
  verboseMode = verbose;
  
  LLVM_DEBUG(llvm::dbgs() << "=== Starting Kernel Execution Optimization ===\n");
  
  // Optimize kernel execution for all functions
  optimizeKernelExecution(moduleOp);
  
  LLVM_DEBUG(llvm::dbgs() << "=== Kernel Execution Optimization Completed ===\n");
}

void KernelExecutionOptimizer::optimizeKernelExecution(mlir::ModuleOp moduleOp) {
  size_t totalFunctions = 0;
  size_t optimizedFunctions = 0;
  size_t totalOriginalOps = 0;
  size_t totalOptimizedOps = 0;
  size_t totalOriginalStreams = 0;
  size_t totalOptimizedStreams = 0;
  
  // 遍历所有 LLVM 函数 - 修正为 LLVM dialect
  for (auto funcOp : moduleOp.getOps<mlir::LLVM::LLVMFuncOp>()) {
    totalFunctions++;
    
    if (funcOp.getBody().empty()) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping empty function: " << funcOp.getName() << "\n");
      continue;
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Optimizing function: " << funcOp.getName() << "\n");
    
    // 优化单个函数
    auto stats = optimizeFunction(funcOp, moduleOp);
    
    if (stats.hasOptimizations) {
      optimizedFunctions++;
      totalOriginalOps += stats.originalOperations;
      totalOptimizedOps += stats.optimizedOperations;
      totalOriginalStreams += stats.originalStreams;
      totalOptimizedStreams += stats.optimizedStreams;
    }
  }
  
  // 打印全局统计 - 这些信息总是显示
  printGlobalStats(totalFunctions, optimizedFunctions, 
                  totalOriginalOps, totalOptimizedOps,
                  totalOriginalStreams, totalOptimizedStreams);
}

KernelExecutionOptimizer::FunctionOptimizationStats 
KernelExecutionOptimizer::optimizeFunction(mlir::LLVM::LLVMFuncOp funcOp, mlir::ModuleOp moduleOp) {
  FunctionOptimizationStats stats;
  
  // 1. 识别并行组
  auto parallelGroups = ParallelGroupIdentifier::identifyParallelGroups(funcOp);
  stats.parallelGroups = parallelGroups.size();
  
  LLVM_DEBUG(llvm::dbgs() << "  Found " << parallelGroups.size() << " parallel groups\n");
  
  // Verbose 模式下显示函数处理信息
  if (verboseMode && !parallelGroups.empty()) {
    llvm::errs() << "  Processing function '" << funcOp.getName() 
                << "' with " << parallelGroups.size() << " parallel groups\n";
  }
  
  if (parallelGroups.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "  No parallel groups found, skipping function\n");
    return stats;
  }
  
  // 2. 为每个并行组计算成本和优化
  mlir::OpBuilder builder(funcOp.getContext());
  bool hasAnyOptimization = false;
  
  for (size_t groupIdx = 0; groupIdx < parallelGroups.size(); groupIdx++) {
    auto& group = parallelGroups[groupIdx];
    
    LLVM_DEBUG(llvm::dbgs() << "  Processing group " << groupIdx 
              << " with " << group.operations.size() << " operations\n");
    
    // 统计原始数据
    stats.originalOperations += group.operations.size();
    stats.originalStreams += group.streams.size();
    
    // 计算每个操作的成本
    calculateOperationCosts(group, moduleOp);
    
    // 优化调度
    auto optimizedSchedule = ScheduleOptimizer::optimizeParallelGroup(group);
    
    LLVM_DEBUG(llvm::dbgs() << "    Optimized to " << optimizedSchedule.size() 
              << " schedule units\n");
    
    // Verbose 模式下显示组优化信息
    if (verboseMode && optimizedSchedule.size() < group.operations.size()) {
      llvm::errs() << "    Group " << groupIdx << ": " 
                  << group.operations.size() << " -> " << optimizedSchedule.size() 
                  << " parallel units\n";
    }
    
    // 如果有优化空间，执行重写
    if (optimizedSchedule.size() < group.operations.size()) {
      LLVM_DEBUG(llvm::dbgs() << "    Applying optimization to group " << groupIdx << "\n");
      
      try {
        ScheduleRewriter::rewriteParallelGroup(group, optimizedSchedule, builder);
        hasAnyOptimization = true;
        stats.optimizedGroups++;
        
        // 统计优化后的数据
        stats.optimizedOperations += group.operations.size(); // 操作数不变，但调度改变
        stats.optimizedStreams += optimizedSchedule.size();   // stream数减少
        
        LLVM_DEBUG(llvm::dbgs() << "    Successfully optimized group " << groupIdx << "\n");
      } catch (const std::exception& e) {
        llvm::errs() << "    Error optimizing group " << groupIdx 
                    << ": " << e.what() << "\n";
        // 继续处理其他组
        stats.optimizedOperations += group.operations.size();
        stats.optimizedStreams += group.streams.size();
      }
    } else {
      LLVM_DEBUG(llvm::dbgs() << "    No optimization benefit for group " << groupIdx << "\n");
      stats.optimizedOperations += group.operations.size();
      stats.optimizedStreams += group.streams.size();
    }
  }
  
  stats.hasOptimizations = hasAnyOptimization;
  
  // 打印函数级统计
  printFunctionStats(funcOp.getName(), stats);
  
  return stats;
}

void KernelExecutionOptimizer::calculateOperationCosts(ParallelGroup& group, mlir::ModuleOp moduleOp) {
  for (auto& operation : group.operations) {
    operation->cost = OperationCostEstimator::estimateOperationCost(
        operation->op, moduleOp);
    
    group.totalCost += operation->cost.estimatedTime;
    group.maxCost = std::max(group.maxCost, operation->cost.estimatedTime);
    
    LLVM_DEBUG(llvm::dbgs() << "      Operation: " << operation->functionName 
              << ", Cost: " << operation->cost.estimatedTime << "ms"
              << ", Type: " << getOperationTypeName(operation->cost.type)
              << ", Characteristic: " << getCharacteristicName(operation->cost.characteristic)
              << ", Efficiency: " << operation->cost.efficiency << "\n");
  }
  
  LLVM_DEBUG(llvm::dbgs() << "    Group total cost: " << group.totalCost 
            << "ms, max cost: " << group.maxCost << "ms\n");
}

void KernelExecutionOptimizer::printFunctionStats(llvm::StringRef funcName, 
                                                 const FunctionOptimizationStats& stats) {
  if (!stats.hasOptimizations) {
    LLVM_DEBUG(llvm::dbgs() << "  Function " << funcName << ": No optimizations applied\n");
    return;
  }
  
  // 简要信息 - 显示在命令行（stderr）
  llvm::errs() << "  Function '" << funcName << "': ";
  if (stats.originalStreams > 0) {
    double streamReduction = (double)(stats.originalStreams - stats.optimizedStreams) / 
                            stats.originalStreams * 100.0;
    llvm::errs() << stats.originalStreams << " -> " << stats.optimizedStreams 
                << " streams (" << llvm::format("%.1f", streamReduction) << "% reduction)\n";
  } else {
    llvm::errs() << "optimized\n";
  }
  
  // 详细信息 - 仅在调试模式下显示
  LLVM_DEBUG(llvm::dbgs() << "  Function " << funcName << " optimization summary:\n");
  LLVM_DEBUG(llvm::dbgs() << "    Parallel groups: " << stats.parallelGroups 
            << " (optimized: " << stats.optimizedGroups << ")\n");
  LLVM_DEBUG(llvm::dbgs() << "    Operations: " << stats.originalOperations 
            << " -> " << stats.optimizedOperations << "\n");
  LLVM_DEBUG(llvm::dbgs() << "    Streams: " << stats.originalStreams 
            << " -> " << stats.optimizedStreams);
  
  if (stats.originalStreams > 0) {
    double streamReduction = (double)(stats.originalStreams - stats.optimizedStreams) / 
                            stats.originalStreams * 100.0;
    LLVM_DEBUG(llvm::dbgs() << " (" << streamReduction << "% reduction)");
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");
}

void KernelExecutionOptimizer::printGlobalStats(size_t totalFunctions, size_t optimizedFunctions,
                                               size_t totalOriginalOps, size_t totalOptimizedOps,
                                               size_t totalOriginalStreams, size_t totalOptimizedStreams) {
  // 总是显示简要统计信息 - 使用 stderr 显示在命令行
  llvm::errs() << "Kernel Optimization Summary:\n";
  llvm::errs() << "  Functions processed: " << totalFunctions << "\n";
  llvm::errs() << "  Functions optimized: " << optimizedFunctions << "\n";
  
  if (optimizedFunctions > 0) {
    llvm::errs() << "  Stream reduction: " << totalOriginalStreams 
                << " -> " << totalOptimizedStreams;
    
    if (totalOriginalStreams > 0) {
      double globalStreamReduction = (double)(totalOriginalStreams - totalOptimizedStreams) / 
                                    totalOriginalStreams * 100.0;
      llvm::errs() << " (" << llvm::format("%.1f", globalStreamReduction) << "% reduction)";
    }
    llvm::errs() << "\n";
    
    // 估算性能收益
    if (totalOriginalStreams > totalOptimizedStreams) {
      double estimatedSpeedup = calculateEstimatedSpeedup(totalOriginalStreams, totalOptimizedStreams);
      llvm::errs() << "  Estimated speedup: " << llvm::format("%.2f", estimatedSpeedup) << "x\n";
    }
  } else {
    llvm::errs() << "  No optimizations applied.\n";
  }
  
  // 详细的调试信息
  LLVM_DEBUG(llvm::dbgs() << "\n=== Detailed Global Optimization Statistics ===\n");
  LLVM_DEBUG(llvm::dbgs() << "Functions processed: " << totalFunctions << "\n");
  LLVM_DEBUG(llvm::dbgs() << "Functions optimized: " << optimizedFunctions << "\n");
  
  if (optimizedFunctions > 0) {
    LLVM_DEBUG(llvm::dbgs() << "Total operations: " << totalOriginalOps 
              << " -> " << totalOptimizedOps << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Total streams: " << totalOriginalStreams 
              << " -> " << totalOptimizedStreams);
    
    if (totalOriginalStreams > 0) {
      double globalStreamReduction = (double)(totalOriginalStreams - totalOptimizedStreams) / 
                                    totalOriginalStreams * 100.0;
      LLVM_DEBUG(llvm::dbgs() << " (" << globalStreamReduction << "% reduction)");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");
    
    // 估算性能收益
    if (totalOriginalStreams > totalOptimizedStreams) {
      double estimatedSpeedup = calculateEstimatedSpeedup(totalOriginalStreams, totalOptimizedStreams);
      LLVM_DEBUG(llvm::dbgs() << "Estimated speedup: " << estimatedSpeedup << "x\n");
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "======================================\n");
}

double KernelExecutionOptimizer::calculateEstimatedSpeedup(size_t originalStreams, size_t optimizedStreams) {
  if (optimizedStreams == 0) return 1.0;
  
  // 简化模型：假设stream管理开销为每个stream 0.1ms
  double originalOverhead = originalStreams * 0.1;
  double optimizedOverhead = optimizedStreams * 0.1;
  double savedOverhead = originalOverhead - optimizedOverhead;
  
  // 假设平均执行时间为5ms
  double averageExecutionTime = 5.0;
  double speedup = (averageExecutionTime + originalOverhead) / 
                  (averageExecutionTime + optimizedOverhead);
  
  return speedup;
}

//===----------------------------------------------------------------------===//
// Utility Functions Implementation
//===----------------------------------------------------------------------===//

bool isComputeIntensive(const OperationCost& cost) {
  return cost.characteristic == ComputeCharacteristic::COMPUTE_INTENSIVE;
}

bool isMemoryIntensive(const OperationCost& cost) {
  return cost.characteristic == ComputeCharacteristic::MEMORY_INTENSIVE;
}

ComputeCharacteristic determineCharacteristic(const OperationCost& cost) {
  if (cost.computeOps <= 0 && cost.memoryAccess <= 0) {
    return ComputeCharacteristic::BALANCED;
  }
  
  // 操作强度 = 计算操作数 / 内存访问字节数 * 4
  // 高操作强度表示计算密集型，低操作强度表示内存密集型
  double operationalIntensity = cost.computeOps / (cost.memoryAccess / 4.0 + 1e-9);
  
  // 根据经验设定阈值
  if (operationalIntensity > 8.0) {
    return ComputeCharacteristic::COMPUTE_INTENSIVE;
  } else if (operationalIntensity < 2.0) {
    return ComputeCharacteristic::MEMORY_INTENSIVE;
  } else {
    return ComputeCharacteristic::BALANCED;
  }
}

OperationType classifyCuDNNOperation(llvm::StringRef funcName) {
  // 更详细的cuDNN操作分类
  static const std::unordered_map<std::string, OperationType> cudnnOps = {
    {"mgpuCudnnConv2dForward", OperationType::CUDNN_CONV},
    {"mgpuCudnnMaxPoolForward", OperationType::CUDNN_POOL},
    {"mgpuCudnnAdd", OperationType::CUDNN_ELEMENTWISE},
    {"mgpuCudnnMul", OperationType::CUDNN_ELEMENTWISE},
    {"mgpuCudnnSub", OperationType::CUDNN_ELEMENTWISE}, 
    {"mgpuCudnnNeg", OperationType::CUDNN_ELEMENTWISE},
    {"mgpuCudnnAddScalar", OperationType::CUDNN_ELEMENTWISE},
    {"mgpuCudnnMulScalar", OperationType::CUDNN_ELEMENTWISE},
    {"mgpuCudnnSubScalar", OperationType::CUDNN_ELEMENTWISE},
    {"mgpuCudnnRSubScalar", OperationType::CUDNN_ELEMENTWISE}
  };
  
  auto it = cudnnOps.find(funcName.str());
  if (it != cudnnOps.end()) {
    return it->second;
  }
  
  // 模式匹配作为fallback
  if (funcName.contains("Conv")) {
    return OperationType::CUDNN_CONV;
  } else if (funcName.contains("Pool")) {
    return OperationType::CUDNN_POOL;
  } else if (funcName.contains("Add") || funcName.contains("Mul") || 
             funcName.contains("Sub") || funcName.contains("Neg")) {
    return OperationType::CUDNN_ELEMENTWISE;
  } else {
    return OperationType::CUDNN_ACTIVATION;
  }
}

OperationType classifyCuBLASOperation(llvm::StringRef funcName) {
  static const std::unordered_map<std::string, OperationType> cublasOps = {
    {"mgpuCulibsFullyConnectedForward", OperationType::CUBLAS_FC},
    {"mgpuCulibsFlattenFullyConnectedForward", OperationType::CUBLAS_FC}
  };
  
  auto it = cublasOps.find(funcName.str());
  if (it != cublasOps.end()) {
    return it->second;
  }
  
  if (funcName.contains("FullyConnected") || funcName.contains("FC")) {
    return OperationType::CUBLAS_FC;
  } else {
    return OperationType::CUBLAS_GEMM;
  }
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// PassWrapper Implementation (在匿名命名空间中)
//===----------------------------------------------------------------------===//

namespace {

// 命令行选项：控制是否显示详细优化信息
static llvm::cl::opt<bool> verboseOptimization(
    "kernel-opt-verbose", 
    llvm::cl::desc("Print detailed kernel optimization information"),
    llvm::cl::init(false));

// 完整的 PassWrapper 实现，只负责 Pass 框架相关的职责
struct KernelExecutionOptimizerPass
    : public mlir::PassWrapper<KernelExecutionOptimizerPass, mlir::OperationPass<mlir::ModuleOp>> {
  
  llvm::StringRef getArgument() const final {
    return "kernel-execution-optimizer";
  }
  
  llvm::StringRef getDescription() const final {
    return "Optimize GPU kernel execution by intelligently merging operations with similar characteristics";
  }
  
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    
    // 输出到 stderr，这样会显示在命令行而不是混入 IR
    llvm::errs() << "Running Kernel Execution Optimizer...\n";
    
    // 创建业务逻辑实例并执行优化，传递 verbose 参数
    onnx_mlir::KernelExecutionOptimizer optimizer;
    optimizer.optimize(moduleOp, verboseOptimization);
    
    // 输出到 stderr  
    llvm::errs() << "Kernel Execution Optimization completed.\n";
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Factory Function
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
  namespace krnl {
    std::unique_ptr<mlir::Pass> createKernelExecutionOptimizerPass() {
      return std::make_unique<KernelExecutionOptimizerPass>();
    }
  }
} // namespace onnx_mlir

// Pass registration
static mlir::PassRegistration<KernelExecutionOptimizerPass> pass;