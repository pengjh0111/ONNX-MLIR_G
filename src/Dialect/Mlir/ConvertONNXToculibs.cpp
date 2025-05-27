#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h" 
#include "llvm/Support/Debug.h"

// Include ONNX dialect headers
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Support/KrnlSupport.hpp"

using namespace mlir;
using namespace onnx_mlir;

#define DEBUG_TYPE "onnx-to-culibs"

namespace {

class ONNXToCuLibsPatternBase {
protected:
  // 辅助函数：确保Handle Pool已初始化
void ensureHandlePoolInitialization(ModuleOp moduleOp, PatternRewriter &rewriter, Location loc, Type ptrType, func::FuncOp currentFunc) const {
  // 检查当前函数中是否已有初始化调用
  bool hasInitCall = false;
  bool hasDestroyCall = false;
  
  currentFunc.walk([&](func::CallOp callOp) {
    StringRef calleeName = callOp.getCallee();
    if (calleeName == "mgpuInitHandlePool") {
      hasInitCall = true;
    }
    if (calleeName == "mgpuDestroyHandlePool") {
      hasDestroyCall = true;
    }
    return WalkResult::advance();
  });
  
  // 如果已经有初始化和销毁调用，就不需要再添加
  if (hasInitCall && hasDestroyCall) {
    return;
  }
  
  auto i32Type = rewriter.getI32Type();
  
  // 创建初始化函数声明
  func::FuncOp initFunc = getOrCreateFunction(moduleOp, rewriter, loc, 
      "mgpuInitHandlePool", rewriter.getFunctionType({i32Type}, {}));
  
  // 创建销毁函数声明
  func::FuncOp destroyFunc = getOrCreateFunction(moduleOp, rewriter, loc, 
      "mgpuDestroyHandlePool", rewriter.getFunctionType({}, {}));
  
  // 确保当前函数有函数体
  if (currentFunc.getBody().empty()) {
    // 如果函数体为空，创建一个基本的函数体
    OpBuilder::InsertionGuard guard(rewriter);
    Block *entryBlock = rewriter.createBlock(&currentFunc.getBody());
    
    // 添加一个返回语句
    rewriter.setInsertionPointToEnd(entryBlock);
    rewriter.create<func::ReturnOp>(loc);
  }
  
  // 插入初始化和销毁调用
  if (!hasInitCall) {
    // 在函数开头插入初始化调用
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&currentFunc.getBody().front());
    
    // 创建池大小常量（默认15个handle）
    auto poolSizeValue = rewriter.create<arith::ConstantOp>(
        loc, i32Type, rewriter.getI32IntegerAttr(15));
    rewriter.create<func::CallOp>(
        loc, TypeRange{}, initFunc.getName(), ValueRange{poolSizeValue});
    
    LLVM_DEBUG(llvm::dbgs() << "Inserted mgpuInitHandlePool call in current function: " 
               << currentFunc.getName() << "\n");
  }
  
  if (!hasDestroyCall) {
    // 在函数结尾插入销毁调用
    OpBuilder::InsertionGuard guard(rewriter);
    
    // 找到所有的返回语句，在每个返回语句前插入销毁调用
    SmallVector<func::ReturnOp> returnOps;
    currentFunc.walk([&](func::ReturnOp returnOp) {
      returnOps.push_back(returnOp);
    });
    
    for (auto returnOp : returnOps) {
      rewriter.setInsertionPoint(returnOp);
      rewriter.create<func::CallOp>(
          loc, TypeRange{}, destroyFunc.getName(), ValueRange{});
    }
    
    LLVM_DEBUG(llvm::dbgs() << "Inserted mgpuDestroyHandlePool call(s) in current function: " 
               << currentFunc.getName() << "\n");
  }
}
  // 辅助函数：获取或创建函数声明
  func::FuncOp getOrCreateFunction(ModuleOp moduleOp, PatternRewriter &rewriter, 
      Location loc, StringRef name, FunctionType funcType) const {
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>(name);
    if (!funcOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      funcOp = rewriter.create<func::FuncOp>(loc, name, funcType);
      funcOp.setPrivate();
    }
    return funcOp;
  }
};

// Pattern to convert onnx.Conv to a call to mgpuCudnnConv2dForward
class ConvOpLowering : public OpRewritePattern<mlir::ONNXConvOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXConvOp convOp, PatternRewriter &rewriter) const override {
    // Get the location for error reporting
    Location loc = convOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Conv at " << loc << "\n");

    // Get the input, weight, and bias tensors
    Value input = convOp.getX();
    Value weights = convOp.getW();
    Value bias = convOp.getB();
    
    // Get the input type
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(convOp, "Input must have static shape");
    }
    
    // Get the weight type
    auto weightType = mlir::dyn_cast<RankedTensorType>(weights.getType());
    if (!weightType || !weightType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(convOp, "Weights must have static shape");
    }
    
    // Extract input dimensions (N, C, H, W)
    auto inputShape = inputType.getShape();
    if (inputShape.size() != 4) {
      return rewriter.notifyMatchFailure(convOp, "Input must be 4D tensor (NCHW)");
    }
    int64_t n = inputShape[0];
    int64_t c = inputShape[1];
    int64_t h = inputShape[2];
    int64_t w = inputShape[3];
    
    // Extract kernel dimensions (K, C, R, S)
    auto weightShape = weightType.getShape();
    if (weightShape.size() != 4) {
      return rewriter.notifyMatchFailure(convOp, "Weights must be 4D tensor (KCHW)");
    }
    int64_t k = weightShape[0]; // Output channels
    int64_t r = weightShape[2]; // Kernel height
    int64_t s = weightShape[3]; // Kernel width
    
    // Extract convolution parameters
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> pads = {0, 0, 0, 0};
    std::vector<int64_t> strides = {1, 1};
    
    // Extract from attributes if available
    if (auto dilationsAttr = convOp.getDilations()) {
      dilations.clear();
      for (auto attr : dilationsAttr.value()) { 
        dilations.push_back(attr.cast<IntegerAttr>().getInt());
      }
    }
    
    if (auto padsAttr = convOp.getPads()) {
      pads.clear();
      for (auto attr : padsAttr.value()) {
        pads.push_back(attr.cast<IntegerAttr>().getInt());
      }
    }
      
    if (auto stridesAttr = convOp.getStrides()) {
      strides.clear();
      for (auto attr : stridesAttr.value()) {
        strides.push_back(attr.cast<IntegerAttr>().getInt());
      }
    }
    
    // Validate parameter dimensions
    if (dilations.size() < 2 || pads.size() < 4 || strides.size() < 2) {
      return rewriter.notifyMatchFailure(convOp, "Invalid convolution parameters");
    }
    
    // Extract specific parameter values
    int64_t dilation_h = dilations[0];
    int64_t dilation_w = dilations[1];
    int64_t pad_h = pads[0];
    int64_t pad_w = pads[1];
    int64_t stride_h = strides[0];
    int64_t stride_w = strides[1];
    
    LLVM_DEBUG(llvm::dbgs() << "Conv params: dilation=" << dilation_h << "," << dilation_w 
               << " pad=" << pad_h << "," << pad_w 
               << " stride=" << stride_h << "," << stride_w << "\n");
    
    // Create constants for integer parameters
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    auto nValue = createI32Const(n);
    auto cValue = createI32Const(c);
    auto hValue = createI32Const(h);
    auto wValue = createI32Const(w);
    auto kValue = createI32Const(k);
    auto rValue = createI32Const(r);
    auto sValue = createI32Const(s);
    auto padHValue = createI32Const(pad_h);
    auto padWValue = createI32Const(pad_w);
    auto strideHValue = createI32Const(stride_h);
    auto strideWValue = createI32Const(stride_w);
    auto dilationHValue = createI32Const(dilation_h);
    auto dilationWValue = createI32Const(dilation_w);
    
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemref = markForBufferization(input);
    auto weightMemref = markForBufferization(weights);
    Value biasMemref;
    if (bias)
      biasMemref = markForBufferization(bias);
    
    // Convert memrefs to void pointers
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    // auto getPtr = [&](Value memref) -> Value {
    //   // Extract the aligned pointer as index
    //   auto indexType = rewriter.getIndexType();
    //   auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
    //   // Convert index to pointer
    //   return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrIndex);
    // };

    auto getPtr = [&](Value memref) -> Value {
        // Extract the aligned pointer as index
        auto indexType = rewriter.getIndexType();
        auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
        
        auto i64Type = rewriter.getIntegerType(64);
        auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
        
        return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
      };
    
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    auto inputPtr = getPtr(inputMemref);
    auto weightPtr = getPtr(weightMemref);
    Value biasPtr;
    if (bias)
      biasPtr = getPtr(biasMemref);
    else
      biasPtr = create.llvm.null(ptrType);
    
    // Allocate output memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(convOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // Create a null CUDA stream (or get from context if available)
  auto moduleOp = convOp->getParentOfType<ModuleOp>();
  auto currentFunc = convOp->getParentOfType<func::FuncOp>();

  ensureHandlePoolInitialization(moduleOp, rewriter, loc, ptrType, currentFunc);

  func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
  
  if (!streamCreateFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    
    auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
    streamCreateFunc = rewriter.create<func::FuncOp>(
      loc, "mgpuStreamCreate", streamCreateType);
    streamCreateFunc.setPrivate();
  }
  
  func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuAcquirePooledHandles");

  if (!handleCreateFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    
    auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
    handleCreateFunc = rewriter.create<func::FuncOp>(
      loc, "mgpuAcquirePooledHandles", handleCreateType);
    handleCreateFunc.setPrivate();
  }

  auto streamCallOp = rewriter.create<func::CallOp>(
    loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
  auto streamPtr = streamCallOp.getResult(0);

  rewriter.create<func::CallOp>(
    loc, TypeRange{}, "mgpuAcquirePooledHandles", ValueRange{streamPtr});
  
  func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>("mgpuCudnnConv2dForward");
  
  if (!funcOp) {
    LLVM_DEBUG(llvm::dbgs() << "Creating mgpuCudnnConv2dForward declaration\n");
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    
    auto funcType = rewriter.getFunctionType({
      i32Type, i32Type, i32Type, i32Type,  // n, c, h, w_in
      i32Type, i32Type, i32Type,           // k, r, s
      i32Type, i32Type,                    // pad_h, pad_w
      i32Type, i32Type,                    // stride_h, stride_w
      i32Type, i32Type,                    // dilation_h, dilation_w
      ptrType, ptrType, ptrType,           // x_data, w_data, bias_data
      ptrType,                             // y_data
      ptrType                              // stream
    }, {});
    
    funcOp = rewriter.create<func::FuncOp>(
      loc, "mgpuCudnnConv2dForward", funcType);
    funcOp.setPrivate();
  }
  
  std::vector<Value> args = {
    nValue, cValue, hValue, wValue,
    kValue, rValue, sValue,
    padHValue, padWValue,
    strideHValue, strideWValue,
    dilationHValue, dilationWValue,
    inputPtr, weightPtr, biasPtr,
    outputPtr, streamPtr
  };
  
  rewriter.create<func::CallOp>(
    loc, TypeRange(), funcOp.getName(), ValueRange(args));
  
  func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
  
  if (!streamSyncFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    
    auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
    streamSyncFunc = rewriter.create<func::FuncOp>(
      loc, "mgpuStreamSynchronize", streamSyncType);
    streamSyncFunc.setPrivate();
  }
  
  rewriter.create<func::CallOp>(
    loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
  
  func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
  
  if (!streamDestroyFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    
    auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
    streamDestroyFunc = rewriter.create<func::FuncOp>(
      loc, "mgpuStreamDestroy", streamDestroyType);
    streamDestroyFunc.setPrivate();
  }
  
  rewriter.create<func::CallOp>(
    loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
  
  auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
      loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
  
  rewriter.replaceOp(convOp, resultTensor);
  
  LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Conv to cuDNN call\n");
  return success();
}
};

// Pattern to convert onnx.Add to a call to mgpuCudnnAdd
class AddOpLowering : public OpRewritePattern<mlir::ONNXAddOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXAddOp>::OpRewritePattern;

  // LogicalResult matchAndRewrite(mlir::ONNXAddOp addOp, PatternRewriter &rewriter) const override {
  //   // Get the location for error reporting
  //   Location loc = addOp.getLoc();
  //   LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Add at " << loc << "\n");

  //   // Get the input tensors
  //   Value inputA = addOp.getA();
  //   Value inputB = addOp.getB();
    
  //   // Get the input types
  //   auto inputTypeA = mlir::dyn_cast<RankedTensorType>(inputA.getType());
  //   auto inputTypeB = mlir::dyn_cast<RankedTensorType>(inputB.getType());
    
  //   if (!inputTypeA || !inputTypeA.hasStaticShape() || !inputTypeB || !inputTypeB.hasStaticShape()) {
  //     return rewriter.notifyMatchFailure(addOp, "Inputs must have static shapes");
  //   }
    
  //   // Extract input dimensions
  //   auto inputShapeA = inputTypeA.getShape();
  //   if (inputShapeA.size() < 1 || inputShapeA.size() > 4) {
  //     return rewriter.notifyMatchFailure(addOp, "Input must be 1D to 4D tensor");
  //   }
    
  //   // Pad shape to 4D (NCHW) if needed
  //   std::vector<int64_t> paddedShapeA(4, 1);
  //   int offset = 4 - inputShapeA.size();
  //   for (size_t i = 0; i < inputShapeA.size(); ++i) {
  //     paddedShapeA[i + offset] = inputShapeA[i];
  //   }
    
  //   int64_t n = paddedShapeA[0];
  //   int64_t c = paddedShapeA[1];
  //   int64_t h = paddedShapeA[2];
  //   int64_t w = paddedShapeA[3];
    
  //   // Create constants for integer parameters
  //   auto i32Type = rewriter.getI32Type();
  //   auto createI32Const = [&](int64_t value) -> Value {
  //     return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
  //   };
    
  //   auto nValue = createI32Const(n);
  //   auto cValue = createI32Const(c);
  //   auto hValue = createI32Const(h);
  //   auto wValue = createI32Const(w);
    
  //   // Prepare input and output buffers
  //   auto markForBufferization = [&](Value tensor) -> Value {
  //     auto tensorType = tensor.getType().cast<RankedTensorType>();
  //     auto memrefType = MemRefType::get(
  //       tensorType.getShape(),
  //       tensorType.getElementType());
  //     return rewriter.create<UnrealizedConversionCastOp>(
  //       loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
  //   };
    
  //   auto inputMemrefA = markForBufferization(inputA);
  //   auto inputMemrefB = markForBufferization(inputB);
    
  //   // Convert memrefs to void pointers
  //   auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
  //   auto getPtr = [&](Value memref) -> Value {
  //     // Extract the aligned pointer as index
  //     auto indexType = rewriter.getIndexType();
  //     auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
  //     auto i64Type = rewriter.getIntegerType(64);
  //     auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
  //     return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
  //   };
    
  //   auto inputPtrA = getPtr(inputMemrefA);
  //   auto inputPtrB = getPtr(inputMemrefB);
    
  //   // Allocate output memref
  //   auto outputType = mlir::dyn_cast<RankedTensorType>(addOp.getResult().getType());
  //   auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
  //   auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
  //   auto outputPtr = getPtr(outputMemref);
    
  //   // Create a CUDA stream
  //   auto moduleOp = addOp->getParentOfType<ModuleOp>();
  //   func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
  //   if (!streamCreateFunc) {
  //     OpBuilder::InsertionGuard guard(rewriter);
  //     rewriter.setInsertionPointToStart(moduleOp.getBody());
      
  //     auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
  //     streamCreateFunc = rewriter.create<func::FuncOp>(
  //       loc, "mgpuStreamCreate", streamCreateType);
  //     streamCreateFunc.setPrivate();
  //   }
    
  //   auto streamCallOp = rewriter.create<func::CallOp>(
  //     loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
  //   auto streamPtr = streamCallOp.getResult(0);
    
  //   // Look up or create the mgpuCudnnAdd function
  //   func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>("mgpuCudnnAdd");
    
  //   if (!funcOp) {
  //     LLVM_DEBUG(llvm::dbgs() << "Creating mgpuCudnnAdd declaration\n");
  //     OpBuilder::InsertionGuard guard(rewriter);
  //     rewriter.setInsertionPointToStart(moduleOp.getBody());
      
  //     auto funcType = rewriter.getFunctionType({
  //       ptrType, ptrType, ptrType,  // inputA, inputB, output
  //       i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
  //       ptrType  // stream
  //     }, {});
      
  //     funcOp = rewriter.create<func::FuncOp>(
  //       loc, "mgpuCudnnAdd", funcType);
  //     funcOp.setPrivate();
  //   }
    
  //   // Call the function
  //   std::vector<Value> args = {
  //     inputPtrA, inputPtrB, outputPtr,
  //     nValue, cValue, hValue, wValue,
  //     streamPtr
  //   };
    
  //   rewriter.create<func::CallOp>(
  //     loc, TypeRange(), funcOp.getName(), ValueRange(args));
    
  //   // Synchronize and destroy the stream
  //   func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
  //   if (!streamSyncFunc) {
  //     OpBuilder::InsertionGuard guard(rewriter);
  //     rewriter.setInsertionPointToStart(moduleOp.getBody());
      
  //     auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
  //     streamSyncFunc = rewriter.create<func::FuncOp>(
  //       loc, "mgpuStreamSynchronize", streamSyncType);
  //     streamSyncFunc.setPrivate();
  //   }
    
  //   rewriter.create<func::CallOp>(
  //     loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
  //   func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
  //   if (!streamDestroyFunc) {
  //     OpBuilder::InsertionGuard guard(rewriter);
  //     rewriter.setInsertionPointToStart(moduleOp.getBody());
      
  //     auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
  //     streamDestroyFunc = rewriter.create<func::FuncOp>(
  //       loc, "mgpuStreamDestroy", streamDestroyType);
  //     streamDestroyFunc.setPrivate();
  //   }
    
  //   rewriter.create<func::CallOp>(
  //     loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
  //   // Convert memref back to tensor
  //   auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
  //       loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
  //   rewriter.replaceOp(addOp, resultTensor);
    
  //   LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Add to cuDNN call\n");
  //   return success();
  // }

  LogicalResult matchAndRewrite(mlir::ONNXAddOp addOp, PatternRewriter &rewriter) const override {
    // 获取位置信息用于错误报告
    Location loc = addOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Add at " << loc << "\n");
  
    // 获取输入张量
    Value inputA = addOp.getA();
    Value inputB = addOp.getB();
    
    // 获取输入类型
    auto inputTypeA = mlir::dyn_cast<RankedTensorType>(inputA.getType());
    auto inputTypeB = mlir::dyn_cast<RankedTensorType>(inputB.getType());
    
    if (!inputTypeA || !inputTypeA.hasStaticShape() || !inputTypeB || !inputTypeB.hasStaticShape()) {
      return rewriter.notifyMatchFailure(addOp, "Inputs must have static shapes");
    }
    
    // 检查是否为标量操作 (inputB 是标量)
    bool isScalarOperation = false;
    auto inputShapeB = inputTypeB.getShape();
    
    // inputB 是标量的条件: 形状为空 [] 或 [1] 或 全1形状 [1,1,...,1]
    if (inputShapeB.empty() || 
        (inputShapeB.size() == 1 && inputShapeB[0] == 1) ||
        (llvm::all_of(inputShapeB, [](int64_t dim) { return dim == 1; }))) {
      isScalarOperation = true;
      LLVM_DEBUG(llvm::dbgs() << "Detected scalar addition\n");
    }
    
    // 提取输入维度
    auto inputShapeA = inputTypeA.getShape();
    if (inputShapeA.size() < 1 || inputShapeA.size() > 4) {
      return rewriter.notifyMatchFailure(addOp, "Input must be 1D to 4D tensor");
    }
    
    // 填充形状到4D (NCHW)
    std::vector<int64_t> paddedShapeA(4, 1);
    int offset = 4 - inputShapeA.size();
    for (size_t i = 0; i < inputShapeA.size(); ++i) {
      paddedShapeA[i + offset] = inputShapeA[i];
    }
    
    int64_t n = paddedShapeA[0];
    int64_t c = paddedShapeA[1];
    int64_t h = paddedShapeA[2];
    int64_t w = paddedShapeA[3];
    
    // 创建整数参数常量
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    auto nValue = createI32Const(n);
    auto cValue = createI32Const(c);
    auto hValue = createI32Const(h);
    auto wValue = createI32Const(w);
    
    // 准备输入和输出缓冲区
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemrefA = markForBufferization(inputA);
    auto inputMemrefB = markForBufferization(inputB);
    
    // 转换 memrefs 为 void pointers
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      // 提取对齐的指针为索引
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtrA = getPtr(inputMemrefA);
    auto inputPtrB = getPtr(inputMemrefB);
    
    // 分配输出 memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(addOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // 创建 CUDA 流
    auto moduleOp = addOp->getParentOfType<ModuleOp>();
    auto currentFunc = addOp->getParentOfType<func::FuncOp>();

    ensureHandlePoolInitialization(moduleOp, rewriter, loc, ptrType, currentFunc);

    func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
    if (!streamCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
      streamCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamCreate", streamCreateType);
      streamCreateFunc.setPrivate();
    }

    func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuAcquirePooledHandles");

    if (!handleCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
      handleCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuAcquirePooledHandles", handleCreateType);
      handleCreateFunc.setPrivate();
    }
    
    auto streamCallOp = rewriter.create<func::CallOp>(
      loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
    auto streamPtr = streamCallOp.getResult(0);
    
    rewriter.create<func::CallOp>(
      loc, TypeRange{}, "mgpuAcquirePooledHandles", ValueRange{streamPtr});

    // 根据是否为标量操作选择合适的函数
    func::FuncOp funcOp;
    
    if (isScalarOperation) {
      // 使用标量加法函数
      funcOp = moduleOp.lookupSymbol<func::FuncOp>("mgpuCudnnAddScalar");
      
      if (!funcOp) {
        LLVM_DEBUG(llvm::dbgs() << "Creating mgpuCudnnAddScalar declaration\n");
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        
        auto funcType = rewriter.getFunctionType({
          ptrType, ptrType, ptrType,  // input, scalar, output
          i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
          ptrType  // stream
        }, {});
        
        funcOp = rewriter.create<func::FuncOp>(
          loc, "mgpuCudnnAddScalar", funcType);
        funcOp.setPrivate();
      }
    } else {
      // 使用普通张量加法函数
      funcOp = moduleOp.lookupSymbol<func::FuncOp>("mgpuCudnnAdd");
      
      if (!funcOp) {
        LLVM_DEBUG(llvm::dbgs() << "Creating mgpuCudnnAdd declaration\n");
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        
        auto funcType = rewriter.getFunctionType({
          ptrType, ptrType, ptrType,  // inputA, inputB, output
          i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
          ptrType  // stream
        }, {});
        
        funcOp = rewriter.create<func::FuncOp>(
          loc, "mgpuCudnnAdd", funcType);
        funcOp.setPrivate();
      }
    }
    
    // 调用函数
    std::vector<Value> args = {
      inputPtrA, inputPtrB, outputPtr,
      nValue, cValue, hValue, wValue,
      streamPtr
    };
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), funcOp.getName(), ValueRange(args));
    
    // 同步并销毁流
    func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
    if (!streamSyncFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
      streamSyncFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamSynchronize", streamSyncType);
      streamSyncFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
    func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
    if (!streamDestroyFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
      streamDestroyFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamDestroy", streamDestroyType);
      streamDestroyFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
    // 将 memref 转换回 tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(addOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Add to cuDNN call\n");
    return success();
  }

};

// Pattern to convert onnx.Sub to a call to mgpuCudnnSub
class SubOpLowering : public OpRewritePattern<mlir::ONNXSubOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXSubOp>::OpRewritePattern;

  // LogicalResult matchAndRewrite(mlir::ONNXSubOp subOp, PatternRewriter &rewriter) const override {
  //   // Get the location for error reporting
  //   Location loc = subOp.getLoc();
  //   LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Sub at " << loc << "\n");

  //   // Get the input tensors
  //   Value inputA = subOp.getA();
  //   Value inputB = subOp.getB();
    
  //   // Get the input types
  //   auto inputTypeA = mlir::dyn_cast<RankedTensorType>(inputA.getType());
  //   auto inputTypeB = mlir::dyn_cast<RankedTensorType>(inputB.getType());
    
  //   if (!inputTypeA || !inputTypeA.hasStaticShape() || !inputTypeB || !inputTypeB.hasStaticShape()) {
  //     return rewriter.notifyMatchFailure(subOp, "Inputs must have static shapes");
  //   }
    
  //   // Extract input dimensions
  //   auto inputShapeA = inputTypeA.getShape();
  //   if (inputShapeA.size() < 1 || inputShapeA.size() > 4) {
  //     return rewriter.notifyMatchFailure(subOp, "Input must be 1D to 4D tensor");
  //   }
    
  //   // Pad shape to 4D (NCHW) if needed
  //   std::vector<int64_t> paddedShapeA(4, 1);
  //   int offset = 4 - inputShapeA.size();
  //   for (size_t i = 0; i < inputShapeA.size(); ++i) {
  //     paddedShapeA[i + offset] = inputShapeA[i];
  //   }
    
  //   int64_t n = paddedShapeA[0];
  //   int64_t c = paddedShapeA[1];
  //   int64_t h = paddedShapeA[2];
  //   int64_t w = paddedShapeA[3];
    
  //   // Create constants for integer parameters
  //   auto i32Type = rewriter.getI32Type();
  //   auto createI32Const = [&](int64_t value) -> Value {
  //     return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
  //   };
    
  //   auto nValue = createI32Const(n);
  //   auto cValue = createI32Const(c);
  //   auto hValue = createI32Const(h);
  //   auto wValue = createI32Const(w);
    
  //   // Prepare input and output buffers
  //   auto markForBufferization = [&](Value tensor) -> Value {
  //     auto tensorType = tensor.getType().cast<RankedTensorType>();
  //     auto memrefType = MemRefType::get(
  //       tensorType.getShape(),
  //       tensorType.getElementType());
  //     return rewriter.create<UnrealizedConversionCastOp>(
  //       loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
  //   };
    
  //   auto inputMemrefA = markForBufferization(inputA);
  //   auto inputMemrefB = markForBufferization(inputB);
    
  //   // Convert memrefs to void pointers
  //   auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
  //   auto getPtr = [&](Value memref) -> Value {
  //     // Extract the aligned pointer as index
  //     auto indexType = rewriter.getIndexType();
  //     auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
  //     auto i64Type = rewriter.getIntegerType(64);
  //     auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
  //     return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
  //   };
    
  //   auto inputPtrA = getPtr(inputMemrefA);
  //   auto inputPtrB = getPtr(inputMemrefB);
    
  //   // Allocate output memref
  //   auto outputType = mlir::dyn_cast<RankedTensorType>(subOp.getResult().getType());
  //   auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
  //   auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
  //   auto outputPtr = getPtr(outputMemref);
    
  //   // Create a CUDA stream
  //   auto moduleOp = subOp->getParentOfType<ModuleOp>();
  //   func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
  //   if (!streamCreateFunc) {
  //     OpBuilder::InsertionGuard guard(rewriter);
  //     rewriter.setInsertionPointToStart(moduleOp.getBody());
      
  //     auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
  //     streamCreateFunc = rewriter.create<func::FuncOp>(
  //       loc, "mgpuStreamCreate", streamCreateType);
  //     streamCreateFunc.setPrivate();
  //   }
    
  //   auto streamCallOp = rewriter.create<func::CallOp>(
  //     loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
  //   auto streamPtr = streamCallOp.getResult(0);
    
  //   // Look up or create the mgpuCudnnSub function
  //   func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>("mgpuCudnnSub");
    
  //   if (!funcOp) {
  //     LLVM_DEBUG(llvm::dbgs() << "Creating mgpuCudnnSub declaration\n");
  //     OpBuilder::InsertionGuard guard(rewriter);
  //     rewriter.setInsertionPointToStart(moduleOp.getBody());
      
  //     auto funcType = rewriter.getFunctionType({
  //       ptrType, ptrType, ptrType,  // inputA, inputB, output
  //       i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
  //       ptrType  // stream
  //     }, {});
      
  //     funcOp = rewriter.create<func::FuncOp>(
  //       loc, "mgpuCudnnSub", funcType);
  //     funcOp.setPrivate();
  //   }
    
  //   // Call the function
  //   std::vector<Value> args = {
  //     inputPtrA, inputPtrB, outputPtr,
  //     nValue, cValue, hValue, wValue,
  //     streamPtr
  //   };
    
  //   rewriter.create<func::CallOp>(
  //     loc, TypeRange(), funcOp.getName(), ValueRange(args));
    
  //   // Synchronize and destroy the stream
  //   func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
  //   if (!streamSyncFunc) {
  //     OpBuilder::InsertionGuard guard(rewriter);
  //     rewriter.setInsertionPointToStart(moduleOp.getBody());
      
  //     auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
  //     streamSyncFunc = rewriter.create<func::FuncOp>(
  //       loc, "mgpuStreamSynchronize", streamSyncType);
  //     streamSyncFunc.setPrivate();
  //   }
    
  //   rewriter.create<func::CallOp>(
  //     loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
  //   func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
  //   if (!streamDestroyFunc) {
  //     OpBuilder::InsertionGuard guard(rewriter);
  //     rewriter.setInsertionPointToStart(moduleOp.getBody());
      
  //     auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
  //     streamDestroyFunc = rewriter.create<func::FuncOp>(
  //       loc, "mgpuStreamDestroy", streamDestroyType);
  //     streamDestroyFunc.setPrivate();
  //   }
    
  //   rewriter.create<func::CallOp>(
  //     loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
  //   // Convert memref back to tensor
  //   auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
  //       loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
  //   rewriter.replaceOp(subOp, resultTensor);
    
  //   LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Sub to cuDNN call\n");
  //   return success();
  // }

  LogicalResult matchAndRewrite(mlir::ONNXSubOp subOp, PatternRewriter &rewriter) const override {
    // 获取位置信息用于错误报告
    Location loc = subOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Sub at " << loc << "\n");
  
    // 获取输入张量
    Value inputA = subOp.getA();
    Value inputB = subOp.getB();
    
    // 获取输入类型
    auto inputTypeA = mlir::dyn_cast<RankedTensorType>(inputA.getType());
    auto inputTypeB = mlir::dyn_cast<RankedTensorType>(inputB.getType());
    
    if (!inputTypeA || !inputTypeA.hasStaticShape() || !inputTypeB || !inputTypeB.hasStaticShape()) {
      return rewriter.notifyMatchFailure(subOp, "Inputs must have static shapes");
    }
    
    // 检查是否为标量操作 (inputB 是标量)
    bool isScalarOperation = false;
    auto inputShapeB = inputTypeB.getShape();
    
    // inputB 是标量的条件: 形状为空 [] 或 [1] 或 全1形状 [1,1,...,1]
    if (inputShapeB.empty() || 
        (inputShapeB.size() == 1 && inputShapeB[0] == 1) ||
        (llvm::all_of(inputShapeB, [](int64_t dim) { return dim == 1; }))) {
      isScalarOperation = true;
      LLVM_DEBUG(llvm::dbgs() << "Detected scalar subtraction\n");
    }
    
    // 提取输入维度
    auto inputShapeA = inputTypeA.getShape();
    if (inputShapeA.size() < 1 || inputShapeA.size() > 4) {
      return rewriter.notifyMatchFailure(subOp, "Input must be 1D to 4D tensor");
    }
    
    // 填充形状到4D (NCHW)
    std::vector<int64_t> paddedShapeA(4, 1);
    int offset = 4 - inputShapeA.size();
    for (size_t i = 0; i < inputShapeA.size(); ++i) {
      paddedShapeA[i + offset] = inputShapeA[i];
    }
    
    int64_t n = paddedShapeA[0];
    int64_t c = paddedShapeA[1];
    int64_t h = paddedShapeA[2];
    int64_t w = paddedShapeA[3];
    
    // 创建整数参数常量
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    auto nValue = createI32Const(n);
    auto cValue = createI32Const(c);
    auto hValue = createI32Const(h);
    auto wValue = createI32Const(w);
    
    // 准备输入和输出缓冲区
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemrefA = markForBufferization(inputA);
    auto inputMemrefB = markForBufferization(inputB);
    
    // 转换 memrefs 为 void pointers
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      // 提取对齐的指针为索引
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtrA = getPtr(inputMemrefA);
    auto inputPtrB = getPtr(inputMemrefB);
    
    // 分配输出 memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(subOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // 创建 CUDA 流
    auto moduleOp = subOp->getParentOfType<ModuleOp>();
    auto currentFunc = subOp->getParentOfType<func::FuncOp>();

    ensureHandlePoolInitialization(moduleOp, rewriter, loc, ptrType, currentFunc);

    func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
    if (!streamCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
      streamCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamCreate", streamCreateType);
      streamCreateFunc.setPrivate();
    }

    func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuAcquirePooledHandles");

    if (!handleCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
      handleCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuAcquirePooledHandles", handleCreateType);
      handleCreateFunc.setPrivate();
    }
    
    auto streamCallOp = rewriter.create<func::CallOp>(
      loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
    auto streamPtr = streamCallOp.getResult(0);
    
    rewriter.create<func::CallOp>(
      loc, TypeRange{}, "mgpuAcquirePooledHandles", ValueRange{streamPtr});

    // 根据是否为标量操作选择合适的函数
    func::FuncOp funcOp;
    
    if (isScalarOperation) {
      // 使用标量减法函数
      funcOp = moduleOp.lookupSymbol<func::FuncOp>("mgpuCudnnSubScalar");
      
      if (!funcOp) {
        LLVM_DEBUG(llvm::dbgs() << "Creating mgpuCudnnSubScalar declaration\n");
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        
        auto funcType = rewriter.getFunctionType({
          ptrType, ptrType, ptrType,  // input, scalar, output
          i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
          ptrType  // stream
        }, {});
        
        funcOp = rewriter.create<func::FuncOp>(
          loc, "mgpuCudnnSubScalar", funcType);
        funcOp.setPrivate();
      }
    } else {
      // 使用普通张量减法函数
      funcOp = moduleOp.lookupSymbol<func::FuncOp>("mgpuCudnnSub");
      
      if (!funcOp) {
        LLVM_DEBUG(llvm::dbgs() << "Creating mgpuCudnnSub declaration\n");
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        
        auto funcType = rewriter.getFunctionType({
          ptrType, ptrType, ptrType,  // inputA, inputB, output
          i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
          ptrType  // stream
        }, {});
        
        funcOp = rewriter.create<func::FuncOp>(
          loc, "mgpuCudnnSub", funcType);
        funcOp.setPrivate();
      }
    }
    
    // 调用函数
    std::vector<Value> args = {
      inputPtrA, inputPtrB, outputPtr,
      nValue, cValue, hValue, wValue,
      streamPtr
    };
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), funcOp.getName(), ValueRange(args));
    
    // 同步并销毁流
    func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
    if (!streamSyncFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
      streamSyncFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamSynchronize", streamSyncType);
      streamSyncFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
    func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
    if (!streamDestroyFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
      streamDestroyFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamDestroy", streamDestroyType);
      streamDestroyFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
    // 将 memref 转换回 tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(subOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Sub to cuDNN call\n");
    return success();
  }

};

// Pattern to convert onnx.Mul to a call to mgpuCudnnMul
class MulOpLowering : public OpRewritePattern<mlir::ONNXMulOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXMulOp>::OpRewritePattern;

  // LogicalResult matchAndRewrite(mlir::ONNXMulOp mulOp, PatternRewriter &rewriter) const override {
  //   // Get the location for error reporting
  //   Location loc = mulOp.getLoc();
  //   LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Mul at " << loc << "\n");

  //   // Get the input tensors
  //   Value inputA = mulOp.getA();
  //   Value inputB = mulOp.getB();
    
  //   // Get the input types
  //   auto inputTypeA = mlir::dyn_cast<RankedTensorType>(inputA.getType());
  //   auto inputTypeB = mlir::dyn_cast<RankedTensorType>(inputB.getType());
    
  //   if (!inputTypeA || !inputTypeA.hasStaticShape() || !inputTypeB || !inputTypeB.hasStaticShape()) {
  //     return rewriter.notifyMatchFailure(mulOp, "Inputs must have static shapes");
  //   }
    
  //   // Extract input dimensions
  //   auto inputShapeA = inputTypeA.getShape();
  //   if (inputShapeA.size() < 1 || inputShapeA.size() > 4) {
  //     return rewriter.notifyMatchFailure(mulOp, "Input must be 1D to 4D tensor");
  //   }
    
  //   // Pad shape to 4D (NCHW) if needed
  //   std::vector<int64_t> paddedShapeA(4, 1);
  //   int offset = 4 - inputShapeA.size();
  //   for (size_t i = 0; i < inputShapeA.size(); ++i) {
  //     paddedShapeA[i + offset] = inputShapeA[i];
  //   }
    
  //   int64_t n = paddedShapeA[0];
  //   int64_t c = paddedShapeA[1];
  //   int64_t h = paddedShapeA[2];
  //   int64_t w = paddedShapeA[3];
    
  //   // Create constants for integer parameters
  //   auto i32Type = rewriter.getI32Type();
  //   auto createI32Const = [&](int64_t value) -> Value {
  //     return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
  //   };
    
  //   auto nValue = createI32Const(n);
  //   auto cValue = createI32Const(c);
  //   auto hValue = createI32Const(h);
  //   auto wValue = createI32Const(w);
    
  //   // Prepare input and output buffers
  //   auto markForBufferization = [&](Value tensor) -> Value {
  //     auto tensorType = tensor.getType().cast<RankedTensorType>();
  //     auto memrefType = MemRefType::get(
  //       tensorType.getShape(),
  //       tensorType.getElementType());
  //     return rewriter.create<UnrealizedConversionCastOp>(
  //       loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
  //   };
    
  //   auto inputMemrefA = markForBufferization(inputA);
  //   auto inputMemrefB = markForBufferization(inputB);
    
  //   // Convert memrefs to void pointers
  //   auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
  //   auto getPtr = [&](Value memref) -> Value {
  //     // Extract the aligned pointer as index
  //     auto indexType = rewriter.getIndexType();
  //     auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
  //     auto i64Type = rewriter.getIntegerType(64);
  //     auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
  //     return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
  //   };
    
  //   auto inputPtrA = getPtr(inputMemrefA);
  //   auto inputPtrB = getPtr(inputMemrefB);
    
  //   // Allocate output memref
  //   auto outputType = mlir::dyn_cast<RankedTensorType>(mulOp.getResult().getType());
  //   auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
  //   auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
  //   auto outputPtr = getPtr(outputMemref);
    
  //   // Create a CUDA stream
  //   auto moduleOp = mulOp->getParentOfType<ModuleOp>();
  //   func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
  //   if (!streamCreateFunc) {
  //     OpBuilder::InsertionGuard guard(rewriter);
  //     rewriter.setInsertionPointToStart(moduleOp.getBody());
      
  //     auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
  //     streamCreateFunc = rewriter.create<func::FuncOp>(
  //       loc, "mgpuStreamCreate", streamCreateType);
  //     streamCreateFunc.setPrivate();
  //   }
    
  //   auto streamCallOp = rewriter.create<func::CallOp>(
  //     loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
  //   auto streamPtr = streamCallOp.getResult(0);
    
  //   // Look up or create the mgpuCudnnMul function
  //   func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>("mgpuCudnnMul");
    
  //   if (!funcOp) {
  //     LLVM_DEBUG(llvm::dbgs() << "Creating mgpuCudnnMul declaration\n");
  //     OpBuilder::InsertionGuard guard(rewriter);
  //     rewriter.setInsertionPointToStart(moduleOp.getBody());
      
  //     auto funcType = rewriter.getFunctionType({
  //       ptrType, ptrType, ptrType,  // inputA, inputB, output
  //       i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
  //       ptrType  // stream
  //     }, {});
      
  //     funcOp = rewriter.create<func::FuncOp>(
  //       loc, "mgpuCudnnMul", funcType);
  //     funcOp.setPrivate();
  //   }
    
  //   // Call the function
  //   std::vector<Value> args = {
  //     inputPtrA, inputPtrB, outputPtr,
  //     nValue, cValue, hValue, wValue,
  //     streamPtr
  //   };
    
  //   rewriter.create<func::CallOp>(
  //     loc, TypeRange(), funcOp.getName(), ValueRange(args));
    
  //   // Synchronize and destroy the stream
  //   func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
  //   if (!streamSyncFunc) {
  //     OpBuilder::InsertionGuard guard(rewriter);
  //     rewriter.setInsertionPointToStart(moduleOp.getBody());
      
  //     auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
  //     streamSyncFunc = rewriter.create<func::FuncOp>(
  //       loc, "mgpuStreamSynchronize", streamSyncType);
  //     streamSyncFunc.setPrivate();
  //   }
    
  //   rewriter.create<func::CallOp>(
  //     loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
  //   func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
  //   if (!streamDestroyFunc) {
  //     OpBuilder::InsertionGuard guard(rewriter);
  //     rewriter.setInsertionPointToStart(moduleOp.getBody());
      
  //     auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
  //     streamDestroyFunc = rewriter.create<func::FuncOp>(
  //       loc, "mgpuStreamDestroy", streamDestroyType);
  //     streamDestroyFunc.setPrivate();
  //   }
    
  //   rewriter.create<func::CallOp>(
  //     loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
  //   // Convert memref back to tensor
  //   auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
  //       loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
  //   rewriter.replaceOp(mulOp, resultTensor);
    
  //   LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Mul to cuDNN call\n");
  //   return success();
  // }

  LogicalResult matchAndRewrite(mlir::ONNXMulOp mulOp, PatternRewriter &rewriter) const override {
    // 获取位置信息用于错误报告
    Location loc = mulOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Mul at " << loc << "\n");
  
    // 获取输入张量
    Value inputA = mulOp.getA();
    Value inputB = mulOp.getB();
    
    // 获取输入类型
    auto inputTypeA = mlir::dyn_cast<RankedTensorType>(inputA.getType());
    auto inputTypeB = mlir::dyn_cast<RankedTensorType>(inputB.getType());
    
    if (!inputTypeA || !inputTypeA.hasStaticShape() || !inputTypeB || !inputTypeB.hasStaticShape()) {
      return rewriter.notifyMatchFailure(mulOp, "Inputs must have static shapes");
    }
    
    // 检查是否为标量操作 (inputB 是标量)
    bool isScalarOperation = false;
    auto inputShapeB = inputTypeB.getShape();
    
    // inputB 是标量的条件: 形状为空 [] 或 [1] 或 全1形状 [1,1,...,1]
    if (inputShapeB.empty() || 
        (inputShapeB.size() == 1 && inputShapeB[0] == 1) ||
        (llvm::all_of(inputShapeB, [](int64_t dim) { return dim == 1; }))) {
      isScalarOperation = true;
      LLVM_DEBUG(llvm::dbgs() << "Detected scalar multiplication\n");
    }
    
    // 提取输入维度
    auto inputShapeA = inputTypeA.getShape();
    if (inputShapeA.size() < 1 || inputShapeA.size() > 4) {
      return rewriter.notifyMatchFailure(mulOp, "Input must be 1D to 4D tensor");
    }
    
    // 填充形状到4D (NCHW)
    std::vector<int64_t> paddedShapeA(4, 1);
    int offset = 4 - inputShapeA.size();
    for (size_t i = 0; i < inputShapeA.size(); ++i) {
      paddedShapeA[i + offset] = inputShapeA[i];
    }
    
    int64_t n = paddedShapeA[0];
    int64_t c = paddedShapeA[1];
    int64_t h = paddedShapeA[2];
    int64_t w = paddedShapeA[3];
    
    // 创建整数参数常量
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    auto nValue = createI32Const(n);
    auto cValue = createI32Const(c);
    auto hValue = createI32Const(h);
    auto wValue = createI32Const(w);
    
    // 准备输入和输出缓冲区
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemrefA = markForBufferization(inputA);
    auto inputMemrefB = markForBufferization(inputB);
    
    // 转换 memrefs 为 void pointers
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      // 提取对齐的指针为索引
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtrA = getPtr(inputMemrefA);
    auto inputPtrB = getPtr(inputMemrefB);
    
    // 分配输出 memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(mulOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // 创建 CUDA 流
    auto moduleOp = mulOp->getParentOfType<ModuleOp>();
    auto currentFunc = mulOp->getParentOfType<func::FuncOp>();

    ensureHandlePoolInitialization(moduleOp, rewriter, loc, ptrType, currentFunc);

    func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
    if (!streamCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
      streamCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamCreate", streamCreateType);
      streamCreateFunc.setPrivate();
    }

    func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuAcquirePooledHandles");

    if (!handleCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
      handleCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuAcquirePooledHandles", handleCreateType);
      handleCreateFunc.setPrivate();
    }
    
    auto streamCallOp = rewriter.create<func::CallOp>(
      loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
    auto streamPtr = streamCallOp.getResult(0);
    
    rewriter.create<func::CallOp>(
      loc, TypeRange{}, "mgpuAcquirePooledHandles", ValueRange{streamPtr});

    // 根据是否为标量操作选择合适的函数
    func::FuncOp funcOp;
    
    if (isScalarOperation) {
      // 使用标量乘法函数
      funcOp = moduleOp.lookupSymbol<func::FuncOp>("mgpuCudnnMulScalar");
      
      if (!funcOp) {
        LLVM_DEBUG(llvm::dbgs() << "Creating mgpuCudnnMulScalar declaration\n");
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        
        auto funcType = rewriter.getFunctionType({
          ptrType, ptrType, ptrType,  // input, scalar, output
          i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
          ptrType  // stream
        }, {});
        
        funcOp = rewriter.create<func::FuncOp>(
          loc, "mgpuCudnnMulScalar", funcType);
        funcOp.setPrivate();
      }
    } else {
      // 使用普通张量乘法函数
      funcOp = moduleOp.lookupSymbol<func::FuncOp>("mgpuCudnnMul");
      
      if (!funcOp) {
        LLVM_DEBUG(llvm::dbgs() << "Creating mgpuCudnnMul declaration\n");
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        
        auto funcType = rewriter.getFunctionType({
          ptrType, ptrType, ptrType,  // inputA, inputB, output
          i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
          ptrType  // stream
        }, {});
        
        funcOp = rewriter.create<func::FuncOp>(
          loc, "mgpuCudnnMul", funcType);
        funcOp.setPrivate();
      }
    }
    
    // 调用函数
    std::vector<Value> args = {
      inputPtrA, inputPtrB, outputPtr,
      nValue, cValue, hValue, wValue,
      streamPtr
    };
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), funcOp.getName(), ValueRange(args));
    
    // 同步并销毁流
    func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
    if (!streamSyncFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
      streamSyncFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamSynchronize", streamSyncType);
      streamSyncFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
    func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
    if (!streamDestroyFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
      streamDestroyFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamDestroy", streamDestroyType);
      streamDestroyFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
    // 将 memref 转换回 tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(mulOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Mul to cuDNN call\n");
    return success();
  }

};

// Pattern to convert onnx.Neg to a call to mgpuCudnnNeg
class NegOpLowering : public OpRewritePattern<mlir::ONNXNegOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXNegOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXNegOp negOp, PatternRewriter &rewriter) const override {
    // Get the location for error reporting
    Location loc = negOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Neg at " << loc << "\n");

    // Get the input tensor
    Value input = negOp.getX();
    
    // Get the input type
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    
    if (!inputType || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(negOp, "Input must have static shape");
    }
    
    // Extract input dimensions
    auto inputShape = inputType.getShape();
    if (inputShape.size() < 1 || inputShape.size() > 4) {
      return rewriter.notifyMatchFailure(negOp, "Input must be 1D to 4D tensor");
    }
    
    // Pad shape to 4D (NCHW) if needed
    std::vector<int64_t> paddedShape(4, 1);
    int offset = 4 - inputShape.size();
    for (size_t i = 0; i < inputShape.size(); ++i) {
      paddedShape[i + offset] = inputShape[i];
    }
    
    int64_t n = paddedShape[0];
    int64_t c = paddedShape[1];
    int64_t h = paddedShape[2];
    int64_t w = paddedShape[3];
    
    // Create constants for integer parameters
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    auto nValue = createI32Const(n);
    auto cValue = createI32Const(c);
    auto hValue = createI32Const(h);
    auto wValue = createI32Const(w);
    
    // Prepare input and output buffers
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemref = markForBufferization(input);
    
    // Convert memrefs to void pointers
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      // Extract the aligned pointer as index
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtr = getPtr(inputMemref);
    
    // Allocate output memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(negOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // Create a CUDA stream
    auto moduleOp = negOp->getParentOfType<ModuleOp>();
    auto currentFunc = negOp->getParentOfType<func::FuncOp>();

    ensureHandlePoolInitialization(moduleOp, rewriter, loc, ptrType, currentFunc);

    func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
    if (!streamCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
      streamCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamCreate", streamCreateType);
      streamCreateFunc.setPrivate();
    }

    func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuAcquirePooledHandles");

    if (!handleCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
      handleCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuAcquirePooledHandles", handleCreateType);
      handleCreateFunc.setPrivate();
    }
    
    auto streamCallOp = rewriter.create<func::CallOp>(
      loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
    auto streamPtr = streamCallOp.getResult(0);
    
    rewriter.create<func::CallOp>(
      loc, TypeRange{}, "mgpuAcquirePooledHandles", ValueRange{streamPtr});

    // Look up or create the mgpuCudnnNeg function
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>("mgpuCudnnNeg");
    
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Creating mgpuCudnnNeg declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto funcType = rewriter.getFunctionType({
        ptrType, ptrType,  // input, output
        i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
        ptrType  // stream
      }, {});
      
      funcOp = rewriter.create<func::FuncOp>(
        loc, "mgpuCudnnNeg", funcType);
      funcOp.setPrivate();
    }
    
    // Call the function
    std::vector<Value> args = {
      inputPtr, outputPtr,
      nValue, cValue, hValue, wValue,
      streamPtr
    };
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), funcOp.getName(), ValueRange(args));
    
    // Synchronize and destroy the stream
    func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
    if (!streamSyncFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
      streamSyncFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamSynchronize", streamSyncType);
      streamSyncFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
    func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
    if (!streamDestroyFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
      streamDestroyFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamDestroy", streamDestroyType);
      streamDestroyFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
    // Convert memref back to tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(negOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Neg to cuDNN call\n");
    return success();
  }
};

// Pattern to convert onnx.MatMul to a call to mgpuCulibsFullyConnectedForward
class MatMulOpLowering : public OpRewritePattern<mlir::ONNXMatMulOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXMatMulOp matMulOp, PatternRewriter &rewriter) const override {
    // 获取位置信息用于错误报告
    Location loc = matMulOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.MatMul at " << loc << "\n");

    // 获取输入张量
    Value inputA = matMulOp.getA();
    Value inputB = matMulOp.getB();
    
    // 获取输入类型
    auto inputTypeA = mlir::dyn_cast<RankedTensorType>(inputA.getType());
    auto inputTypeB = mlir::dyn_cast<RankedTensorType>(inputB.getType());
    
    if (!inputTypeA || !inputTypeA.hasStaticShape() || !inputTypeB || !inputTypeB.hasStaticShape()) {
      return rewriter.notifyMatchFailure(matMulOp, "Inputs must have static shapes");
    }
    
    // 提取输入维度
    auto inputShapeA = inputTypeA.getShape();
    auto inputShapeB = inputTypeB.getShape();
    
    // MatMul需要至少2D张量
    if (inputShapeA.size() < 2 || inputShapeB.size() < 2) {
      return rewriter.notifyMatchFailure(matMulOp, "Inputs must be at least 2D tensors");
    }
    
    // 我们只处理2D矩阵乘法（像全连接层那样）
    if (inputShapeA.size() != 2 || inputShapeB.size() != 2) {
      return rewriter.notifyMatchFailure(matMulOp, "Only 2D matrix multiplication is supported");
    }
    
    // 对于全连接，inputA形状为[batch_size, input_features]，inputB形状为[input_features, output_features]
    int64_t batch_size = inputShapeA[0];
    int64_t input_features = inputShapeA[1];
    int64_t output_features = inputShapeB[1];
    
    // 验证维度匹配
    if (input_features != inputShapeB[0]) {
      return rewriter.notifyMatchFailure(matMulOp, "Inner dimensions must match for matrix multiplication");
    }
    
    LLVM_DEBUG(llvm::dbgs() << "MatMul dimensions: batch_size=" << batch_size 
               << ", input_features=" << input_features 
               << ", output_features=" << output_features << "\n");
    
    // 创建常量用于整数参数
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    auto batchSizeValue = createI32Const(batch_size);
    auto inputFeaturesValue = createI32Const(input_features);
    auto outputFeaturesValue = createI32Const(output_features);
    
    // 将输入张量标记为缓冲区
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemrefA = markForBufferization(inputA);
    auto inputMemrefB = markForBufferization(inputB);
    
    // 将memref转为void指针
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      // 提取对齐的指针为索引
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtrA = getPtr(inputMemrefA);
    auto weightPtrB = getPtr(inputMemrefB);
    
    // 分配输出memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(matMulOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // 创建CUDA流
    auto moduleOp = matMulOp->getParentOfType<ModuleOp>();
    auto currentFunc = matMulOp->getParentOfType<func::FuncOp>();

    ensureHandlePoolInitialization(moduleOp, rewriter, loc, ptrType, currentFunc);

    func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
    if (!streamCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
      streamCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamCreate", streamCreateType);
      streamCreateFunc.setPrivate();
    }

    func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuAcquirePooledHandles");

    if (!handleCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
      handleCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuAcquirePooledHandles", handleCreateType);
      handleCreateFunc.setPrivate();
    }
    
    auto streamCallOp = rewriter.create<func::CallOp>(
      loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
    auto streamPtr = streamCallOp.getResult(0);
    
    rewriter.create<func::CallOp>(
      loc, TypeRange{}, "mgpuAcquirePooledHandles", ValueRange{streamPtr});
  
    // 创建或查找FC函数声明
    func::FuncOp fcFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuCulibsFullyConnectedForward");
    
    if (!fcFunc) {
      LLVM_DEBUG(llvm::dbgs() << "Creating mgpuCulibsFullyConnectedForward declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto fcFuncType = rewriter.getFunctionType({
        i32Type, i32Type, i32Type,  // batch_size, input_features, output_features
        ptrType, ptrType, ptrType,  // input_data, weight_data, bias_data
        ptrType,                    // output_data
        ptrType                     // stream
      }, {});
      
      fcFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuCulibsFullyConnectedForward", fcFuncType);
      fcFunc.setPrivate();
    }
    
    // 创建null指针用于偏置（MatMul没有偏置）
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    auto nullBiasPtr = create.llvm.null(ptrType);
    
    // 调用FC函数
    std::vector<Value> args = {
      batchSizeValue, inputFeaturesValue, outputFeaturesValue,
      inputPtrA, weightPtrB, nullBiasPtr,
      outputPtr, streamPtr
    };
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), fcFunc.getName(), ValueRange(args));
    
    // 同步流
    func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
    if (!streamSyncFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
      streamSyncFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamSynchronize", streamSyncType);
      streamSyncFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
    // 销毁流
    func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
    if (!streamDestroyFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
      streamDestroyFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamDestroy", streamDestroyType);
      streamDestroyFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
    // 将memref转回tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(matMulOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.MatMul to FC call\n");
    return success();
  }
};

// Pattern to convert onnx.Gemm to a call to mgpuCulibsFullyConnectedForward
class GemmOpLowering : public OpRewritePattern<mlir::ONNXGemmOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXGemmOp gemmOp, PatternRewriter &rewriter) const override {
    // 获取位置信息用于错误报告
    Location loc = gemmOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Gemm at " << loc << "\n");

    // 获取输入张量
    Value inputA = gemmOp.getA();
    Value inputB = gemmOp.getB();
    Value inputC = gemmOp.getC(); // 这是偏置
    
    // 获取属性
    // float alpha = 1.0f;
    // if (auto alphaAttr = gemmOp.getAlpha())
    //   alpha = alphaAttr.value().convertToFloat();
      
    // float beta = 1.0f;
    // if (auto betaAttr = gemmOp.getBeta())
    //   beta = betaAttr.value().convertToFloat();

    float alpha = 1.0f;
    if (auto alphaAttr = dyn_cast_or_null<FloatAttr>(gemmOp.getAlphaAttr()))
      alpha = alphaAttr.getValueAsDouble();
      
    float beta = 1.0f;
    if (auto betaAttr = dyn_cast_or_null<FloatAttr>(gemmOp.getBetaAttr()))
      beta = betaAttr.getValueAsDouble();
    
    // bool transA = false;
    // if (auto transAAttr = gemmOp.getTransA())
    //   transA = transAAttr.value() != 0;
      
    // bool transB = false;
    // if (auto transBAttr = gemmOp.getTransB())
    //   transB = transBAttr.value() != 0;

    bool transA = false;
    if (auto transAAttr = gemmOp.getTransAAttr()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(transAAttr)) {
        // 使用 getSExtValue() 安全地获取有符号整数值
        transA = intAttr.getValue().getSExtValue() != 0;
      }
    }

    bool transB = false;
    if (auto transBAttr = gemmOp.getTransBAttr()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(transBAttr)) {
        // 使用 getSExtValue() 安全地获取有符号整数值
        transB = intAttr.getValue().getSExtValue() != 0;
      }
    }
    
    // 检查是否有需要特殊处理的alpha和beta值
    if (alpha != 1.0f) {
      return rewriter.notifyMatchFailure(gemmOp, "Alpha != 1.0 not supported yet");
    }
    
    if (beta != 1.0f) {
      return rewriter.notifyMatchFailure(gemmOp, "Beta != 1.0 not supported yet");
    }
    
    // 获取输入类型
    auto inputTypeA = mlir::dyn_cast<RankedTensorType>(inputA.getType());
    auto inputTypeB = mlir::dyn_cast<RankedTensorType>(inputB.getType());
    
    if (!inputTypeA || !inputTypeA.hasStaticShape() || !inputTypeB || !inputTypeB.hasStaticShape()) {
      return rewriter.notifyMatchFailure(gemmOp, "Inputs must have static shapes");
    }
    
    // 提取输入维度
    auto inputShapeA = inputTypeA.getShape();
    auto inputShapeB = inputTypeB.getShape();
    
    // Gemm需要2D张量
    if (inputShapeA.size() != 2 || inputShapeB.size() != 2) {
      return rewriter.notifyMatchFailure(gemmOp, "Gemm inputs must be 2D tensors");
    }
    
    // 根据转置标志确定实际维度
    int64_t batch_size, input_features;
    if (transA) {
      batch_size = inputShapeA[1];
      input_features = inputShapeA[0];
    } else {
      batch_size = inputShapeA[0];
      input_features = inputShapeA[1];
    }
    
    int64_t weight_rows, weight_cols;
    if (transB) {
      weight_rows = inputShapeB[1];
      weight_cols = inputShapeB[0];
    } else {
      weight_rows = inputShapeB[0];
      weight_cols = inputShapeB[1];
    }
    
    // 验证内部维度匹配
    if (input_features != weight_rows) {
      return rewriter.notifyMatchFailure(gemmOp, "Inner dimensions must match for Gemm");
    }
    
    int64_t output_features = weight_cols;
    
    LLVM_DEBUG(llvm::dbgs() << "Gemm dimensions: batch_size=" << batch_size 
               << ", input_features=" << input_features 
               << ", output_features=" << output_features 
               << ", transA=" << transA << ", transB=" << transB << "\n");
    
    // 如果转置标志不是FC层支持的形式，报错
    if (transA) {
      return rewriter.notifyMatchFailure(gemmOp, "TransA=1 not supported for FC conversion");
    }
    
    // 创建常量用于整数参数
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    auto batchSizeValue = createI32Const(batch_size);
    auto inputFeaturesValue = createI32Const(input_features);
    auto outputFeaturesValue = createI32Const(output_features);
    
    // 将输入张量标记为缓冲区
    auto markForBufferization = [&](Value tensor) -> Value {
      if (!tensor)
        return nullptr;
      
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemrefA = markForBufferization(inputA);
    auto inputMemrefB = markForBufferization(inputB);
    auto biasMemref = markForBufferization(inputC);
    
    // 将memref转为void指针
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      if (!memref) {
        MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
        return create.llvm.null(ptrType);
      }
      
      // 提取对齐的指针为索引
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtrA = getPtr(inputMemrefA);
    auto weightPtrB = getPtr(inputMemrefB);
    // auto biasPtrC = biasMemref ? getPtr(biasMemref) : rewriter.create<LLVM::NullOp>(loc, ptrType).getResult();
    Value biasPtrC;
    if (biasMemref) {
        biasPtrC = getPtr(biasMemref);
    } else {
        MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
        biasPtrC = create.llvm.null(ptrType);
    }
    
    // 分配输出memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(gemmOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // 创建CUDA流
    auto moduleOp = gemmOp->getParentOfType<ModuleOp>();
    auto currentFunc = gemmOp->getParentOfType<func::FuncOp>();

    ensureHandlePoolInitialization(moduleOp, rewriter, loc, ptrType, currentFunc);

    func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
    if (!streamCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
      streamCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamCreate", streamCreateType);
      streamCreateFunc.setPrivate();
    }

    func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuAcquirePooledHandles");

    if (!handleCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
      handleCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuAcquirePooledHandles", handleCreateType);
      handleCreateFunc.setPrivate();
    }
    
    auto streamCallOp = rewriter.create<func::CallOp>(
      loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
    auto streamPtr = streamCallOp.getResult(0);
    
    rewriter.create<func::CallOp>(
      loc, TypeRange{}, "mgpuAcquirePooledHandles", ValueRange{streamPtr});

    // 创建或查找FC函数声明
    func::FuncOp fcFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuCulibsFullyConnectedForward");
    
    if (!fcFunc) {
      LLVM_DEBUG(llvm::dbgs() << "Creating mgpuCulibsFullyConnectedForward declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto fcFuncType = rewriter.getFunctionType({
        i32Type, i32Type, i32Type,  // batch_size, input_features, output_features
        ptrType, ptrType, ptrType,  // input_data, weight_data, bias_data
        ptrType,                    // output_data
        ptrType                     // stream
      }, {});
      
      fcFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuCulibsFullyConnectedForward", fcFuncType);
      fcFunc.setPrivate();
    }
    
    // 调用FC函数
    std::vector<Value> args = {
      batchSizeValue, inputFeaturesValue, outputFeaturesValue,
      inputPtrA, weightPtrB, biasPtrC,
      outputPtr, streamPtr
    };
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), fcFunc.getName(), ValueRange(args));
    
    // 同步流
    func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
    if (!streamSyncFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
      streamSyncFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamSynchronize", streamSyncType);
      streamSyncFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
    // 销毁流
    func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
    if (!streamDestroyFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
      streamDestroyFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamDestroy", streamDestroyType);
      streamDestroyFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
    // 将memref转回tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(gemmOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Gemm to FC call\n");
    return success();
  }
};

// Pattern to convert onnx.Flatten+onnx.Gemm to a call to mgpuCulibsFlattenFullyConnectedForward
class FlattenGemmOpLowering : public OpRewritePattern<mlir::ONNXGemmOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXGemmOp gemmOp, PatternRewriter &rewriter) const override {
    // First, check if the input to the Gemm is from a Flatten operation
    auto flattenOp = gemmOp.getA().getDefiningOp<mlir::ONNXFlattenOp>();
    if (!flattenOp) {
      // Not a Flatten->Gemm pattern, skip this operation
      return failure();
    }

    // Get the location for error reporting
    Location loc = gemmOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Flatten+onnx.Gemm at " << loc << "\n");

    // Get the original input (before flattening)
    Value originalInput = flattenOp.getInput();
    Value weights = gemmOp.getB();
    Value bias = gemmOp.getC();
    
    // Check that the Flatten has axis=1 (we only support that case)
    int64_t flattenAxis = 1;
    if (auto axisAttr = flattenOp.getAxisAttr()) {
      flattenAxis = axisAttr.getValue().getSExtValue();
    }
    
    if (flattenAxis != 1) {
      return rewriter.notifyMatchFailure(flattenOp, "Only support flattening with axis=1");
    }
    
    // Get the original input type
    auto inputType = mlir::dyn_cast<RankedTensorType>(originalInput.getType());
    if (!inputType || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(flattenOp, "Input must have static shape");
    }
    
    // Get input dimensions
    auto inputShape = inputType.getShape();
    if (inputShape.size() != 4) {
      return rewriter.notifyMatchFailure(flattenOp, "Only 4D NCHW inputs are supported for now");
    }
    
    // Extract NCHW dimensions
    int64_t batchSize = inputShape[0];
    int64_t channels = inputShape[1];
    int64_t height = inputShape[2];
    int64_t width = inputShape[3];
    
    // Calculate flattened features
    int64_t flattenedFeatures = channels * height * width;
    
    // Get the weights type
    auto weightType = mlir::dyn_cast<RankedTensorType>(weights.getType());
    if (!weightType || !weightType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(gemmOp, "Weights must have static shape");
    }
    
    // Extract weight dimensions
    auto weightShape = weightType.getShape();
    if (weightShape.size() != 2) {
      return rewriter.notifyMatchFailure(gemmOp, "Weights must be 2D");
    }
    
    // Check that the flattened dimension matches the weight input dimension
    if (flattenedFeatures != weightShape[0]) {
      return rewriter.notifyMatchFailure(gemmOp, "Flattened dimension doesn't match weight dimension");
    }
    
    int64_t outputFeatures = weightShape[1];
    
    // Get attributes from Gemm op
    float alpha = 1.0f;
    if (auto alphaAttr = dyn_cast_or_null<FloatAttr>(gemmOp.getAlphaAttr()))
      alpha = alphaAttr.getValueAsDouble();
      
    float beta = 1.0f;
    if (auto betaAttr = dyn_cast_or_null<FloatAttr>(gemmOp.getBetaAttr()))
      beta = betaAttr.getValueAsDouble();
    
    bool transA = false;
    if (auto transAAttr = gemmOp.getTransAAttr()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(transAAttr)) {
        transA = intAttr.getValue().getSExtValue() != 0;
      }
    }

    bool transB = false;
    if (auto transBAttr = gemmOp.getTransBAttr()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(transBAttr)) {
        transB = intAttr.getValue().getSExtValue() != 0;
      }
    }
    
    // Check for unsupported attributes
    if (alpha != 1.0f) {
      return rewriter.notifyMatchFailure(gemmOp, "Alpha != 1.0 not supported yet");
    }
    
    if (beta != 1.0f) {
      return rewriter.notifyMatchFailure(gemmOp, "Beta != 1.0 not supported yet");
    }
    
    if (transA) {
      return rewriter.notifyMatchFailure(gemmOp, "TransA=1 not supported for flattened FC");
    }
    
    if (transB) {
      return rewriter.notifyMatchFailure(gemmOp, "TransB=1 not supported for flattened FC");
    }
    
    // Create constants for integer parameters
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    auto batchSizeValue = createI32Const(batchSize);
    auto channelsValue = createI32Const(channels);
    auto heightValue = createI32Const(height);
    auto widthValue = createI32Const(width);
    auto outputFeaturesValue = createI32Const(outputFeatures);
    
    // Mark tensors for bufferization
    auto markForBufferization = [&](Value tensor) -> Value {
      if (!tensor)
        return nullptr;
      
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemref = markForBufferization(originalInput);
    auto weightMemref = markForBufferization(weights);
    auto biasMemref = markForBufferization(bias);
    
    // Convert memrefs to void pointers
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      if (!memref) {
        MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
        return create.llvm.null(ptrType);
      }
      
      // Extract the aligned pointer as index
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtr = getPtr(inputMemref);
    auto weightPtr = getPtr(weightMemref);
    Value biasPtr;
    if (biasMemref)
      biasPtr = getPtr(biasMemref);
    else {
      MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
      biasPtr = create.llvm.null(ptrType);
    }
    
    // Allocate output memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(gemmOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // Create a CUDA stream
    auto moduleOp = gemmOp->getParentOfType<ModuleOp>();
    auto currentFunc = gemmOp->getParentOfType<func::FuncOp>();

    ensureHandlePoolInitialization(moduleOp, rewriter, loc, ptrType, currentFunc);

    func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
    if (!streamCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
      streamCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamCreate", streamCreateType);
      streamCreateFunc.setPrivate();
    }

    func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuAcquirePooledHandles");

    if (!handleCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
      handleCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuAcquirePooledHandles", handleCreateType);
      handleCreateFunc.setPrivate();
    }
    
    auto streamCallOp = rewriter.create<func::CallOp>(
      loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
    auto streamPtr = streamCallOp.getResult(0);
    
    rewriter.create<func::CallOp>(
      loc, TypeRange{}, "mgpuAcquirePooledHandles", ValueRange{streamPtr});

    // Create or locate the flatten-FC function declaration
    func::FuncOp fcFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuCulibsFlattenFullyConnectedForward");
    
    if (!fcFunc) {
      LLVM_DEBUG(llvm::dbgs() << "Creating mgpuCulibsFlattenFullyConnectedForward declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto fcFuncType = rewriter.getFunctionType({
        i32Type, i32Type, i32Type, i32Type,  // batch_size, channels, height, width
        i32Type,                             // output_features
        ptrType, ptrType, ptrType,           // input_data, weight_data, bias_data
        ptrType,                             // output_data
        ptrType                              // stream
      }, {});
      
      fcFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuCulibsFlattenFullyConnectedForward", fcFuncType);
      fcFunc.setPrivate();
    }
    
    // Call the function
    std::vector<Value> args = {
      batchSizeValue, channelsValue, heightValue, widthValue,
      outputFeaturesValue,
      inputPtr, weightPtr, biasPtr,
      outputPtr, streamPtr
    };
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), fcFunc.getName(), ValueRange(args));
    
    // Synchronize the stream
    func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
    if (!streamSyncFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
      streamSyncFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamSynchronize", streamSyncType);
      streamSyncFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
    // Destroy the stream
    func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
    if (!streamDestroyFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
      streamDestroyFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamDestroy", streamDestroyType);
      streamDestroyFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
    // Convert memref back to tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(gemmOp, resultTensor);
    
    // The Flatten op will be removed automatically if it has no other uses
    if (flattenOp.use_empty())
      rewriter.eraseOp(flattenOp);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Flatten+onnx.Gemm to FC call\n");
    return success();
  }
};

// Pattern to convert onnx.Flatten+onnx.MatMul to a call to mgpuCulibsFlattenFullyConnectedForward
class FlattenMatMulOpLowering : public OpRewritePattern<mlir::ONNXMatMulOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXMatMulOp matMulOp, PatternRewriter &rewriter) const override {
    // First, check if the input to the MatMul is from a Flatten operation
    auto flattenOp = matMulOp.getA().getDefiningOp<mlir::ONNXFlattenOp>();
    if (!flattenOp) {
      // Not a Flatten->MatMul pattern, skip this operation
      return failure();
    }

    // Get the location for error reporting
    Location loc = matMulOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.Flatten+onnx.MatMul at " << loc << "\n");

    // Get the original input (before flattening)
    Value originalInput = flattenOp.getInput();
    Value weights = matMulOp.getB();
    
    // Check that the Flatten has axis=1 (we only support that case)
    int64_t flattenAxis = 1;
    if (auto axisAttr = flattenOp.getAxisAttr()) {
      flattenAxis = axisAttr.getValue().getSExtValue();
    }
    
    if (flattenAxis != 1) {
      return rewriter.notifyMatchFailure(flattenOp, "Only support flattening with axis=1");
    }
    
    // Get the original input type
    auto inputType = mlir::dyn_cast<RankedTensorType>(originalInput.getType());
    if (!inputType || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(flattenOp, "Input must have static shape");
    }
    
    // Get input dimensions
    auto inputShape = inputType.getShape();
    if (inputShape.size() != 4) {
      return rewriter.notifyMatchFailure(flattenOp, "Only 4D NCHW inputs are supported for now");
    }
    
    // Extract NCHW dimensions
    int64_t batchSize = inputShape[0];
    int64_t channels = inputShape[1];
    int64_t height = inputShape[2];
    int64_t width = inputShape[3];
    
    // Calculate flattened features
    int64_t flattenedFeatures = channels * height * width;
    
    // Get the weights type
    auto weightType = mlir::dyn_cast<RankedTensorType>(weights.getType());
    if (!weightType || !weightType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(matMulOp, "Weights must have static shape");
    }
    
    // Extract weight dimensions
    auto weightShape = weightType.getShape();
    if (weightShape.size() != 2) {
      return rewriter.notifyMatchFailure(matMulOp, "Weights must be 2D");
    }
    
    // Check that the flattened dimension matches the weight input dimension
    if (flattenedFeatures != weightShape[0]) {
      return rewriter.notifyMatchFailure(matMulOp, "Flattened dimension doesn't match weight dimension");
    }
    
    int64_t outputFeatures = weightShape[1];
    
    // Create constants for integer parameters
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    auto batchSizeValue = createI32Const(batchSize);
    auto channelsValue = createI32Const(channels);
    auto heightValue = createI32Const(height);
    auto widthValue = createI32Const(width);
    auto outputFeaturesValue = createI32Const(outputFeatures);
    
    // Mark tensors for bufferization
    auto markForBufferization = [&](Value tensor) -> Value {
      if (!tensor)
        return nullptr;
      
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemref = markForBufferization(originalInput);
    auto weightMemref = markForBufferization(weights);
    
    // Convert memrefs to void pointers
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      if (!memref) {
        MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
        return create.llvm.null(ptrType);
      }
      
      // Extract the aligned pointer as index
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtr = getPtr(inputMemref);
    auto weightPtr = getPtr(weightMemref);
    
    // Create null bias pointer (MatMul doesn't have bias)
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    auto nullBiasPtr = create.llvm.null(ptrType);
    
    // Allocate output memref
    auto outputType = mlir::dyn_cast<RankedTensorType>(matMulOp.getResult().getType());
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // Create a CUDA stream
    auto moduleOp = matMulOp->getParentOfType<ModuleOp>();
    auto currentFunc = matMulOp->getParentOfType<func::FuncOp>();

    ensureHandlePoolInitialization(moduleOp, rewriter, loc, ptrType, currentFunc);

    func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
    if (!streamCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
      streamCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamCreate", streamCreateType);
      streamCreateFunc.setPrivate();
    }

    func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuAcquirePooledHandles");

    if (!handleCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
      handleCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuAcquirePooledHandles", handleCreateType);
      handleCreateFunc.setPrivate();
    }
    
    auto streamCallOp = rewriter.create<func::CallOp>(
      loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
    auto streamPtr = streamCallOp.getResult(0);

    rewriter.create<func::CallOp>(
      loc, TypeRange{}, "mgpuAcquirePooledHandles", ValueRange{streamPtr});
    
    // Create or locate the flatten-FC function declaration
    func::FuncOp fcFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuCulibsFlattenFullyConnectedForward");
    
    if (!fcFunc) {
      LLVM_DEBUG(llvm::dbgs() << "Creating mgpuCulibsFlattenFullyConnectedForward declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto fcFuncType = rewriter.getFunctionType({
        i32Type, i32Type, i32Type, i32Type,  // batch_size, channels, height, width
        i32Type,                             // output_features
        ptrType, ptrType, ptrType,           // input_data, weight_data, bias_data
        ptrType,                             // output_data
        ptrType                              // stream
      }, {});
      
      fcFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuCulibsFlattenFullyConnectedForward", fcFuncType);
      fcFunc.setPrivate();
    }
    
    // Call the function
    std::vector<Value> args = {
      batchSizeValue, channelsValue, heightValue, widthValue,
      outputFeaturesValue,
      inputPtr, weightPtr, nullBiasPtr,
      outputPtr, streamPtr
    };
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), fcFunc.getName(), ValueRange(args));
    
    // Synchronize the stream
    func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
    if (!streamSyncFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
      streamSyncFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamSynchronize", streamSyncType);
      streamSyncFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
    // Destroy the stream
    func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
    if (!streamDestroyFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
      streamDestroyFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamDestroy", streamDestroyType);
      streamDestroyFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
    // Convert memref back to tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(matMulOp, resultTensor);
    
    // The Flatten op will be removed automatically if it has no other uses
    if (flattenOp.use_empty())
      rewriter.eraseOp(flattenOp);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.Flatten+onnx.MatMul to FC call\n");
    return success();
  }
};

// Pattern to convert onnx.MaxPoolSingleOut to a call to mgpuCudnnMaxPoolForward
class MaxPoolOpLowering : public OpRewritePattern<mlir::ONNXMaxPoolSingleOutOp>, public ONNXToCuLibsPatternBase {
public:
  using OpRewritePattern<mlir::ONNXMaxPoolSingleOutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::ONNXMaxPoolSingleOutOp maxPoolOp, PatternRewriter &rewriter) const override {
    // 获取位置用于错误报告
    Location loc = maxPoolOp.getLoc();
    LLVM_DEBUG(llvm::dbgs() << "Converting onnx.MaxPoolSingleOut at " << loc << "\n");

    // 获取输入张量
    Value input = maxPoolOp.getX();
    
    // 获取输入类型
    auto inputType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(maxPoolOp, "Input must have static shape");
    }
    
    // 获取输出类型
    auto outputType = mlir::dyn_cast<RankedTensorType>(maxPoolOp.getO_Y().getType());
    if (!outputType || !outputType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(maxPoolOp, "Output must have static shape");
    }
    
    // 提取属性
    std::vector<int64_t> kernelShape;
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    
    // 获取核形状（必需）
    if (auto kernelShapeAttr = maxPoolOp.getKernelShapeAttr()) {
      for (auto attr : kernelShapeAttr) {
        kernelShape.push_back(attr.cast<IntegerAttr>().getInt());
      }
    } else {
      return rewriter.notifyMatchFailure(maxPoolOp, "Kernel shape attribute is required");
    }
    
    // 获取填充、步长和膨胀（可选，有默认值）
    if (auto padsAttr = maxPoolOp.getPadsAttr()) {
      for (auto attr : padsAttr) {
        pads.push_back(attr.cast<IntegerAttr>().getInt());
      }
    } else {
      // 默认：无填充
      pads = std::vector<int64_t>(kernelShape.size() * 2, 0);
    }
    
    if (auto stridesAttr = maxPoolOp.getStridesAttr()) {
      for (auto attr : stridesAttr) {
        strides.push_back(attr.cast<IntegerAttr>().getInt());
      }
    } else {
      // 默认：步长为1
      strides = std::vector<int64_t>(kernelShape.size(), 1);
    }
    
    if (auto dilationsAttr = maxPoolOp.getDilationsAttr()) {
      for (auto attr : dilationsAttr) {
        dilations.push_back(attr.cast<IntegerAttr>().getInt());
      }
    } else {
      // 默认：无膨胀
      dilations = std::vector<int64_t>(kernelShape.size(), 1);
    }
    
    // 检查auto_pad属性
    std::string autoPad = "NOTSET";
    if (auto autoPadAttr = maxPoolOp.getAutoPadAttr()) {
      autoPad = autoPadAttr.getValue().str();
    }
    
    // 当前仅支持"NOTSET"自动填充模式
    if (autoPad != "NOTSET") {
      return rewriter.notifyMatchFailure(maxPoolOp, "Only NOTSET auto_pad mode is supported");
    }
    
    // 检查ceil_mode属性
    int64_t ceilMode = 0;
    if (auto ceilModeAttr = maxPoolOp.getCeilModeAttr()) {
      ceilMode = ceilModeAttr.getValue().getSExtValue();
    }
    
    // 当前仅支持ceil_mode == 0
    if (ceilMode != 0) {
      return rewriter.notifyMatchFailure(maxPoolOp, "Only ceil_mode=0 is supported");
    }
    
    // 提取输入维度（NCHW）
    auto inputShape = inputType.getShape();
    if (inputShape.size() != 4) {
      return rewriter.notifyMatchFailure(maxPoolOp, "Input must be 4D tensor (NCHW)");
    }
    int64_t n = inputShape[0];
    int64_t c = inputShape[1];
    int64_t h = inputShape[2];
    int64_t w = inputShape[3];
    
    // 提取池化参数
    if (kernelShape.size() != 2) {
      return rewriter.notifyMatchFailure(maxPoolOp, "Only 2D pooling is supported");
    }
    int64_t kernel_h = kernelShape[0];
    int64_t kernel_w = kernelShape[1];
    
    if (pads.size() != 4) {
      return rewriter.notifyMatchFailure(maxPoolOp, "Pads must have 4 values");
    }
    int64_t pad_h_begin = pads[0]; // 顶部填充
    int64_t pad_w_begin = pads[1]; // 左侧填充
    int64_t pad_h_end = pads[2];   // 底部填充
    int64_t pad_w_end = pads[3];   // 右侧填充
    
    if (strides.size() != 2) {
      return rewriter.notifyMatchFailure(maxPoolOp, "Strides must have 2 values");
    }
    int64_t stride_h = strides[0];
    int64_t stride_w = strides[1];
    
    if (dilations.size() != 2) {
      return rewriter.notifyMatchFailure(maxPoolOp, "Dilations must have 2 values");
    }
    int64_t dilation_h = dilations[0];
    int64_t dilation_w = dilations[1];
    
    // 创建整数参数常量
    auto i32Type = rewriter.getI32Type();
    auto createI32Const = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    
    auto nValue = createI32Const(n);
    auto cValue = createI32Const(c);
    auto hValue = createI32Const(h);
    auto wValue = createI32Const(w);
    auto kernelHValue = createI32Const(kernel_h);
    auto kernelWValue = createI32Const(kernel_w);
    auto padHBeginValue = createI32Const(pad_h_begin);
    auto padWBeginValue = createI32Const(pad_w_begin);
    auto padHEndValue = createI32Const(pad_h_end);
    auto padWEndValue = createI32Const(pad_w_end);
    auto strideHValue = createI32Const(stride_h);
    auto strideWValue = createI32Const(stride_w);
    auto dilationHValue = createI32Const(dilation_h);
    auto dilationWValue = createI32Const(dilation_w);
    
    // 准备bufferization
    auto markForBufferization = [&](Value tensor) -> Value {
      auto tensorType = tensor.getType().cast<RankedTensorType>();
      auto memrefType = MemRefType::get(
        tensorType.getShape(),
        tensorType.getElementType());
      return rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{memrefType}, ValueRange{tensor}).getResult(0);
    };
    
    auto inputMemref = markForBufferization(input);
    
    // 转换memrefs为void指针
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    
    auto getPtr = [&](Value memref) -> Value {
      // 提取对齐的指针作为索引
      auto indexType = rewriter.getIndexType();
      auto ptrIndex = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, indexType, memref);
      
      auto i64Type = rewriter.getIntegerType(64);
      auto ptrI64 = rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrIndex);
      
      return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrI64);
    };
    
    auto inputPtr = getPtr(inputMemref);
    
    // 分配输出memref
    auto outputMemrefType = MemRefType::get(outputType.getShape(), outputType.getElementType());
    auto outputMemref = rewriter.create<memref::AllocOp>(loc, outputMemrefType);
    auto outputPtr = getPtr(outputMemref);
    
    // 创建CUDA流
    auto moduleOp = maxPoolOp->getParentOfType<ModuleOp>();
    auto currentFunc = maxPoolOp->getParentOfType<func::FuncOp>();

    ensureHandlePoolInitialization(moduleOp, rewriter, loc, ptrType, currentFunc);

    func::FuncOp streamCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamCreate");
    
    if (!streamCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamCreateType = rewriter.getFunctionType({}, {ptrType});
      streamCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamCreate", streamCreateType);
      streamCreateFunc.setPrivate();
    }

    func::FuncOp handleCreateFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuAcquirePooledHandles");

    if (!handleCreateFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto handleCreateType = rewriter.getFunctionType({ptrType}, {});
      handleCreateFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuAcquirePooledHandles", handleCreateType);
      handleCreateFunc.setPrivate();
    }
    
    auto streamCallOp = rewriter.create<func::CallOp>(
      loc, TypeRange{ptrType}, streamCreateFunc.getName(), ValueRange{});
    auto streamPtr = streamCallOp.getResult(0);

    rewriter.create<func::CallOp>(
      loc, TypeRange{}, "mgpuAcquirePooledHandles", ValueRange{streamPtr});
    
    // 查找或创建mgpuCudnnMaxPoolForward函数
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>("mgpuCudnnMaxPoolForward");
    
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "Creating mgpuCudnnMaxPoolForward declaration\n");
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto funcType = rewriter.getFunctionType({
        i32Type, i32Type, i32Type, i32Type,  // n, c, h, w
        i32Type, i32Type,                    // kernel_h, kernel_w
        i32Type, i32Type,                    // pad_h_begin, pad_w_begin
        i32Type, i32Type,                    // pad_h_end, pad_w_end
        i32Type, i32Type,                    // stride_h, stride_w
        i32Type, i32Type,                    // dilation_h, dilation_w
        ptrType, ptrType,                    // input_data, output_data
        ptrType                              // stream
      }, {});
      
      funcOp = rewriter.create<func::FuncOp>(
        loc, "mgpuCudnnMaxPoolForward", funcType);
      funcOp.setPrivate();
    }
    
    // 调用函数
    std::vector<Value> args = {
      nValue, cValue, hValue, wValue,
      kernelHValue, kernelWValue,
      padHBeginValue, padWBeginValue,
      padHEndValue, padWEndValue,
      strideHValue, strideWValue,
      dilationHValue, dilationWValue,
      inputPtr, outputPtr,
      streamPtr
    };
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), funcOp.getName(), ValueRange(args));
    
    // 同步并销毁流
    func::FuncOp streamSyncFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamSynchronize");
    
    if (!streamSyncFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamSyncType = rewriter.getFunctionType({ptrType}, {});
      streamSyncFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamSynchronize", streamSyncType);
      streamSyncFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamSyncFunc.getName(), ValueRange{streamPtr});
    
    func::FuncOp streamDestroyFunc = moduleOp.lookupSymbol<func::FuncOp>("mgpuStreamDestroy");
    
    if (!streamDestroyFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      
      auto streamDestroyType = rewriter.getFunctionType({ptrType}, {});
      streamDestroyFunc = rewriter.create<func::FuncOp>(
        loc, "mgpuStreamDestroy", streamDestroyType);
      streamDestroyFunc.setPrivate();
    }
    
    rewriter.create<func::CallOp>(
      loc, TypeRange(), streamDestroyFunc.getName(), ValueRange{streamPtr});
    
    // 将memref转回tensor
    auto resultTensor = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{outputType}, ValueRange{outputMemref}).getResult(0);
    
    rewriter.replaceOp(maxPoolOp, resultTensor);
    
    LLVM_DEBUG(llvm::dbgs() << "Successfully converted onnx.MaxPoolSingleOut to cuDNN call\n");
    return success();
  }
};

// Pass to convert ONNX operations to cuDNN calls
struct ONNXToCuDNNPass
    : public PassWrapper<ONNXToCuDNNPass, OperationPass<ModuleOp>> {
  
  StringRef getArgument() const final { return "convert-onnx-to-culibs"; }
  StringRef getDescription() const final {
    return "Convert ONNX operations to cuDNN runtime calls";
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<arith::ArithDialect>();
  }
  
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();
    
    // Define the conversion patterns
    RewritePatternSet patterns(context);
    patterns.add<ConvOpLowering>(context);
    patterns.add<AddOpLowering>(context);
    patterns.add<SubOpLowering>(context);
    patterns.add<MulOpLowering>(context);
    patterns.add<NegOpLowering>(context);
    patterns.add<MatMulOpLowering>(context);
    patterns.add<GemmOpLowering>(context);
    patterns.add<FlattenGemmOpLowering>(context, /*benefit=*/2);
    patterns.add<FlattenMatMulOpLowering>(context, /*benefit=*/2);
    patterns.add<MaxPoolOpLowering>(context);

    // Apply patterns
    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect, arith::ArithDialect, 
                           memref::MemRefDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addLegalOp<arith::IndexCastOp>();                       
    target.addIllegalOp<mlir::ONNXConvOp>();
    target.addIllegalOp<mlir::ONNXAddOp>();
    target.addIllegalOp<mlir::ONNXSubOp>();
    target.addIllegalOp<mlir::ONNXMulOp>();
    target.addIllegalOp<mlir::ONNXNegOp>();
    target.addIllegalOp<mlir::ONNXMatMulOp>();
    target.addIllegalOp<mlir::ONNXGemmOp>();
    target.addIllegalOp<mlir::ONNXMaxPoolSingleOutOp>();
    
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace

// Pass registration
namespace onnx_mlir {
    std::unique_ptr<Pass> createONNXToCuDNNPass() {
      return std::make_unique<ONNXToCuDNNPass>();
    }
} // namespace onnx_mlir

static mlir::PassRegistration<ONNXToCuDNNPass> pass;