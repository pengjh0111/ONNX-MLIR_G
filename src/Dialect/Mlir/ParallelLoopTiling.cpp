//===- ParallelLoopTiling.cpp - Tiles scf.parallel ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop tiling on parallel loops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include <queue>
namespace mlir {
#define GEN_PASS_DEF_SCFPARALLELLOOPTILING
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;
using namespace llvm;

/// Tile a parallel loop of the form
///   scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
///                                            step (%arg4, %arg5)
///
/// into
///   scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
///                                            step (%arg4*tileSize[0],
///                                                  %arg5*tileSize[1])
///     scf.parallel (%j0, %j1) = (0, 0) to (min(%arg4*tileSize[0], %arg2-%i0)
///                                          min(%arg5*tileSize[1], %arg3-%i1))
///                                      step (%arg4, %arg5)
///
/// or, when no-min-max-bounds is true, into
///   scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
///                                            step (%arg4*tileSize[0],
///                                                  %arg5*tileSize[1])
///     scf.parallel (%j0, %j1) = (0, 0) to (%arg4*tileSize[0],
///                                          %arg5*tileSize[1])
///                                      step (%arg4, %arg5)
///        %inbound = (%j0 * %arg4 + %i0 < %arg2) &&
///                   (%j1 * %arg5 + %i1 < %arg3)
///        scf.if (%inbound)
///          ....
///
/// where the uses of %i0 and %i1 in the loop body are replaced by
/// %i0 + j0 and %i1 + %j1.
///
/// The old loop is replaced with the new one.
namespace {
struct LoopDim {
  int length;
  int max_axis;
};

struct TilingSolution {
  SmallVector<int> thread_tiles;
  SmallVector<int> block_tiles;
  float score = -1.0f;
};
struct TileCandidate {
  SmallVector<int> tiles;
  float score;
  
  TileCandidate(size_t numDims) : tiles(numDims, 1), score(0.0f) {}
  
  bool operator<(const TileCandidate &other) const {
      return score < other.score;
  }
};

constexpr int MAX_THREADS = 1024;
constexpr int WARP_SIZE = 32;
constexpr std::array<int, 3> MAX_BLOCK_DIM_ARRAY = {1024, 1024, 64};
const ArrayRef<int> MAX_BLOCK_DIM(MAX_BLOCK_DIM_ARRAY);
// const ArrayRef<int> MAX_BLOCK_DIM = {1024, 1024, 64};
} // namespace

void generateCandidates(int length, int maxTile, SmallVectorImpl<int> &candidates) {
  candidates.clear();
  maxTile = std::min(maxTile, length);

  // 生成Warp对齐的候选
  bool hasCandidate = false;
  for (int t = (maxTile / WARP_SIZE) * WARP_SIZE; t >= WARP_SIZE; t -= WARP_SIZE)
    if (length % t == 0) {
      candidates.push_back(t);
      hasCandidate = true;
    }

  // 生成非对齐但可整除的候选
  if (!hasCandidate) {
    for (int t = maxTile; t >= 1; --t) {
        if (length % t == 0) {
            candidates.push_back(t);
        }
    }
  }
}

float evaluateSolution(ArrayRef<LoopDim> dims, ArrayRef<int> threadTiles) {
  int threadProduct = 1;
  int blockProduct = 1;
  int warpAlignedCount = 0;
  
  for (int i = 0; i < dims.size(); ++i) {
      threadProduct *= threadTiles[i];
      blockProduct *= dims[i].length / threadTiles[i];
      
      // 奖励Warp对齐的维度
      if (threadTiles[i] % WARP_SIZE == 0) {
          warpAlignedCount++;
      }
  }
  
  float utilization = 1.0f - std::abs(1.0f - float(threadProduct)/MAX_THREADS);
  float blockScore = std::log(1 + blockProduct) * 2.0f;
  float alignmentBonus = warpAlignedCount * 0.5f;
  
  return utilization * 0.4 + blockScore * 0.5 + alignmentBonus * 0.1;
}

void optimizeTiling(ArrayRef<LoopDim> dims, TilingSolution &best) {
  best = TilingSolution{};
  if (dims.empty()) return;

  const int numDims = dims.size();
  SmallVector<SmallVector<int>> allCandidates(numDims);

  // 步骤1：为每个维度生成候选（降序排列）
  // 生成候选时确保非空
  for (int i = 0; i < numDims; ++i) {
    generateCandidates(dims[i].length, dims[i].max_axis, allCandidates[i]);
    assert(!allCandidates[i].empty() && "Candidates should never be empty");
    std::reverse(allCandidates[i].begin(), allCandidates[i].end());
  }

  // 步骤2：优先队列用于存储候选方案
  // 选择每个维度的第一个候选（最大可能值）
  TileCandidate initial(numDims);
  for (int i = 0; i < numDims; ++i) {
    initial.tiles[i] = allCandidates[i].front(); // 取最大候选
  }
  
  // 检查初始候选的有效性
  int totalThreads = 1;
  for (int t : initial.tiles) totalThreads *= t;
  if (totalThreads > MAX_THREADS) {
    // 如果初始候选无效，回退到全1
    initial.tiles.assign(numDims, 1);
  }
  initial.score = evaluateSolution(dims, initial.tiles);
  
  std::priority_queue<TileCandidate> pq;
  pq.push(initial);

  // 步骤3：分支限界搜索
  constexpr int MAX_CANDIDATES = 1000;
  int evaluated = 0;
  
  while (!pq.empty() && evaluated++ < MAX_CANDIDATES) {
      auto current = pq.top();
      pq.pop();

      // 检查是否完整方案
      bool isComplete = true;
      for (int i = 0; i < numDims; ++i) {
          if (current.tiles[i] == 1) {
              isComplete = false;
              break;
          }
      }

      if (isComplete) {
          if (current.score > best.score) {
              best.thread_tiles = current.tiles;
              best.score = current.score;
          }
          continue;
      }

      // 生成子候选
      for (int dim = 0; dim < numDims; ++dim) {
          if (current.tiles[dim] != 1) continue;

          for (int cand : allCandidates[dim]) {
              // 检查线程总数约束
              int total = 1;
              for (int d = 0; d < numDims; ++d) {
                  total *= (d == dim) ? cand : current.tiles[d];
                  if (total > MAX_THREADS) break;
              }
              if (total > MAX_THREADS) continue;

              // 创建新候选
              TileCandidate newCandidate = current;
              newCandidate.tiles[dim] = cand;
              newCandidate.score = evaluateSolution(dims, newCandidate.tiles);
              
              pq.push(newCandidate);
          }
          break; // 每次只扩展一个维度
      }
  }

  // 在计算block_tiles前添加回退
  if (best.thread_tiles.empty()) {
    best.thread_tiles.resize(numDims);
    
    // 第一遍：设置每个维度为最大可能值
    for (int i = 0; i < numDims; ++i) {
      best.thread_tiles[i] = std::min(dims[i].length, dims[i].max_axis);
      best.thread_tiles[i] = std::max(best.thread_tiles[i], 1); // 确保不小于1
    }

    // 第二遍：调整到满足线程总数限制
    int total = 1;
    for (int t : best.thread_tiles) total *= t;
    
    while (total > MAX_THREADS && total > 1) {
      bool reduced = false;
      // 从最后一个维度开始调整
      for (int i = numDims-1; i >= 0; --i) {
        if (best.thread_tiles[i] > 1) {
          const int original = best.thread_tiles[i];
          // 寻找下一个更小的候选
          for (int t : allCandidates[i]) {
            if (t < original) {
              best.thread_tiles[i] = t;
              total = total / original * t;
              reduced = true;
              break;
            }
          }
          if (reduced) break;
        }
      }
      
      // 如果无法进一步调整，直接设为全1
      if (!reduced) {
        best.thread_tiles.assign(numDims, 1);
        break;
      }
    }
    
    best.score = evaluateSolution(dims, best.thread_tiles);
  }
  
  // 步骤4：计算block_tiles
  best.block_tiles.resize(numDims);
  for (int i = 0; i < numDims; ++i) {
    assert(best.thread_tiles[i] != 0 && "Zero tile size detected");
    assert(dims[i].length % best.thread_tiles[i] == 0 && "Invalid tile size");
    best.block_tiles[i] = dims[i].length / best.thread_tiles[i];
  }
}

std::pair<ParallelOp, ParallelOp>
tileParallelLoopPlus(ParallelOp op, llvm::ArrayRef<int64_t> tileSizes,
                            bool noMinMaxBounds) {
  OpBuilder b(op);
  auto zero = b.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  SmallVector<Value, 2> tileSizeConstants;
  tileSizeConstants.reserve(op.getUpperBound().size()); //调用reserve方法为tileSizeConstants预分配足够的存储空间，空间大小与op的上界维度数量（op.getUpperBound().size()）相同。
  for (size_t i = 0, end = op.getUpperBound().size(); i != end; ++i) { //遍历从0到op.getUpperBound().size()-1的所有维度。每个维度都需要为tile操作设置一个尺寸常量
    if (i < tileSizes.size())
      tileSizeConstants.push_back(
          b.create<arith::ConstantIndexOp>(op.getLoc(), tileSizes[i]));
    else //如果当前维度在tileSizes数组中没有对应的tile尺寸（即i超出了tileSizes的范围），则默认创建一个值为1的常数索引
      // Just pick 1 for the remaining dimensions.
      tileSizeConstants.push_back(
          b.create<arith::ConstantIndexOp>(op.getLoc(), 1));
  }

  // Create the outer loop with adjusted steps.
  SmallVector<Value, 2> newSteps;
  newSteps.reserve(op.getStep().size());
  for (auto step : llvm::zip(op.getStep(), tileSizeConstants)) {
    newSteps.push_back(b.create<arith::MulIOp>(op.getLoc(), std::get<0>(step),
                                               std::get<1>(step)));
  }
  auto outerLoop = b.create<ParallelOp>(op.getLoc(), op.getLowerBound(),
                                        op.getUpperBound(), newSteps);
  b.setInsertionPointToStart(outerLoop.getBody());

  // Compute min(size, dim - offset) to avoid out-of-bounds accesses.
  auto minMap = AffineMap::get(
      /*dimCount=*/3, /*symbolCount=*/0,
      {getAffineDimExpr(/*position=*/0, b.getContext()),
       getAffineDimExpr(/*position=*/1, b.getContext()) -
           getAffineDimExpr(/*position=*/2, b.getContext())},
      b.getContext());

  // Create the inner loop with adjusted bounds.
  SmallVector<Value, 2> newBounds;
  newBounds.reserve(op.getUpperBound().size());
  bool needInboundCheck = false;
  for (auto [lowerBound, upperBound, newStep, iv, step, tileSizeConstant] :
       llvm::zip(outerLoop.getLowerBound(), outerLoop.getUpperBound(),
                 outerLoop.getStep(), outerLoop.getInductionVars(),
                 op.getStep(), tileSizeConstants)) {
    // Collect the statically known loop bounds
    auto lowerBoundConstant =
        dyn_cast_or_null<arith::ConstantIndexOp>(lowerBound.getDefiningOp());
    auto upperBoundConstant =
        dyn_cast_or_null<arith::ConstantIndexOp>(upperBound.getDefiningOp());
    auto stepConstant =
        dyn_cast_or_null<arith::ConstantIndexOp>(step.getDefiningOp());
    auto tileSize =
        cast<arith::ConstantIndexOp>(tileSizeConstant.getDefiningOp()).value();
    // If the loop bounds and the loop step are constant and if the number of
    // loop iterations is an integer multiple of the tile size, we use a static
    // bound for the inner loop.
    if (lowerBoundConstant && upperBoundConstant && stepConstant) {
      auto numIterations = llvm::divideCeil(upperBoundConstant.value() -
                                                lowerBoundConstant.value(),
                                            stepConstant.value());
      if (numIterations % tileSize == 0) {
        newBounds.push_back(newStep);
        continue;
      }
    }

    // For InboundCheck mode, just use the variable outer step
    if (noMinMaxBounds) {
      newBounds.push_back(newStep);
      needInboundCheck = true;
      continue;
    }

    // Otherwise, we dynamically compute the bound for
    // each iteration of the outer loop.
    newBounds.push_back(
        b.create<affine::AffineMinOp>(op.getLoc(), b.getIndexType(), minMap,
                                      ValueRange{newStep, upperBound, iv}));
  }
  auto innerLoop = b.create<ParallelOp>(
      op.getLoc(), SmallVector<Value, 2>(newBounds.size(), zero), newBounds,
      op.getStep());

  if (noMinMaxBounds && needInboundCheck) {
    b.setInsertionPointToStart(innerLoop.getBody());
    // Insert in-bound check
    Value inbound =
        b.create<arith::ConstantIntOp>(op.getLoc(), 1, b.getIntegerType(1));
    for (auto [outerUpperBound, outerIV, innerIV, innerStep] :
         llvm::zip(outerLoop.getUpperBound(), outerLoop.getInductionVars(),
                   innerLoop.getInductionVars(), innerLoop.getStep())) {
      // %in_bound = %in_bound &&
      //             (%inner_iv * %inner_step + %outer_iv < %outer_upper_bound)
      Value index = b.create<arith::AddIOp>(
          op.getLoc(), b.create<arith::MulIOp>(op.getLoc(), innerIV, innerStep),
          outerIV);
      Value dimInbound = b.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ult, index, outerUpperBound);
      inbound = b.create<arith::AndIOp>(op.getLoc(), inbound, dimInbound);
    }
    auto ifInbound = b.create<IfOp>(op.getLoc(),
                                    /*resultTypes*/ llvm::ArrayRef<Type>{}, inbound,
                                    /*hasElseRegion*/ false);
    ifInbound.getThenRegion().takeBody(op.getRegion());
    Block &thenBlock = ifInbound.getThenRegion().front();
    // Replace the scf.reduce terminator with an scf.yield terminator.
    Operation *reduceOp = thenBlock.getTerminator();
    b.setInsertionPointToEnd(&thenBlock);
    b.create<scf::YieldOp>(reduceOp->getLoc());
    reduceOp->erase();
    b.setInsertionPointToStart(innerLoop.getBody());
    for (const auto &ivs : llvm::enumerate(llvm::zip(
             innerLoop.getInductionVars(), outerLoop.getInductionVars()))) {
      auto newIndex = b.create<arith::AddIOp>(
          op.getLoc(), std::get<0>(ivs.value()), std::get<1>(ivs.value()));
      thenBlock.getArgument(ivs.index())
          .replaceAllUsesExcept(newIndex, newIndex);
    }
    thenBlock.eraseArguments(0, thenBlock.getNumArguments());
  } else {
    innerLoop.getRegion().takeBody(op.getRegion());
    b.setInsertionPointToStart(innerLoop.getBody());
    for (auto ivs : llvm::zip(innerLoop.getInductionVars(),
                              outerLoop.getInductionVars())) {
      Value innerIndex = std::get<0>(ivs);
      auto newIndex = b.create<arith::AddIOp>(op.getLoc(), std::get<0>(ivs),
                                              std::get<1>(ivs));
      innerIndex.replaceAllUsesExcept(newIndex, newIndex);
    }
  }

  op.erase();
  return std::make_pair(outerLoop, innerLoop);
}

namespace {
struct ParallelLoopTiling
    : public impl::SCFParallelLoopTilingBase<ParallelLoopTiling> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParallelLoopTiling)
  explicit ParallelLoopTiling(bool noMinMaxBounds = false) {
    this->noMinMaxBounds = noMinMaxBounds;
  }

  void runOnOperation() override {
    // for (auto tileSize : tileSizes)
    //   if (tileSize == 0) {
    //     mlir::emitError(mlir::UnknownLoc::get(&Pass::getContext()),
    //                     "tile size cannot be 0");
    //     return signalPassFailure();
    //   }
    // auto *parentOp = getOperation();
    // SmallVector<ParallelOp, 2> innermostPloops;
    // getInnermostParallelLoops(parentOp, innermostPloops);
    // for (ParallelOp ploop : innermostPloops) {
    //   // FIXME: Add reduction support.
    //   if (ploop.getNumReductions() == 0)
    //     tileParallelLoopPlus(ploop, tileSizes, noMinMaxBounds);
    // }
    auto *parentOp = getOperation();
    SmallVector<scf::ParallelOp> innermostPloops;
    getInnermostParallelLoops(parentOp, innermostPloops);
  
    for (scf::ParallelOp ploop : innermostPloops) {
      if (ploop.getNumReductions() != 0) continue;
  
      // 收集维度信息
      SmallVector<LoopDim> loopDims;
      for (auto [i, bounds] : llvm::enumerate(llvm::zip(
               ploop.getLowerBound(), ploop.getUpperBound(), ploop.getStep()))) {
        auto [lower, upper, step] = bounds;
        auto lowerConst = dyn_cast<arith::ConstantIndexOp>(lower.getDefiningOp());
        auto upperConst = dyn_cast<arith::ConstantIndexOp>(upper.getDefiningOp());
        auto stepConst = dyn_cast<arith::ConstantIndexOp>(step.getDefiningOp());
  
        if (!lowerConst || !upperConst || !stepConst) {
          ploop.emitWarning("Dynamic loop bounds not supported");
          loopDims.clear();
          break;
        }
  
        int length = (upperConst.value() - lowerConst.value()) / stepConst.value();
        int maxAxis = i < 3 ? MAX_BLOCK_DIM[i] : 32;
        loopDims.push_back({length, maxAxis});
      }
  
      if (loopDims.empty()) continue;
  
      // 计算最佳分块
      TilingSolution solution;
      optimizeTiling(loopDims, solution);
  
      // 转换分块方案
      SmallVector<int64_t> tileSizes;
      for (int t : solution.thread_tiles)
        tileSizes.push_back(t);
  
      // 应用分块
      tileParallelLoopPlus(ploop, tileSizes, noMinMaxBounds);
    }
  }


  
  StringRef getArgument() const final { return "scf-parallel-loop-tiling-plus"; }
  StringRef getDescription() const final { 
    return "....."; 
  }
};
} // namespace

// std::unique_ptr<Pass>
// mlir::createParallelLoopTilingPass(ArrayRef<int64_t> tileSizes,
//                                    bool noMinMaxBounds) {
//   return std::make_unique<ParallelLoopTiling>(tileSizes, noMinMaxBounds);
// }


namespace onnx_mlir {
  std::unique_ptr<mlir::Pass> createParallelLoopTilingPass(bool noMinMaxBounds) {
    return std::make_unique<ParallelLoopTiling>(noMinMaxBounds);
  }
} // namespace onnx_mlir
    
static mlir::PassRegistration<ParallelLoopTiling> pass;
  