//===- LoopFusion.cpp - Code to perform loop fusion -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements affine fusion.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <iomanip>
#include <optional>
#include <sstream>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINELOOPFUSION
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-fusion"

using namespace mlir;
using namespace mlir::affine;

namespace {
/// Loop fusion pass. This pass currently supports a greedy fusion policy,
/// which fuses loop nests with single-writer/single-reader memref dependences
/// with the goal of improving locality.
// TODO: Support fusion of source loop nests which write to multiple
// memrefs, where each memref can have multiple users (if profitable).
struct LoopFusion : public affine::impl::AffineLoopFusionBase<LoopFusion> { //LoopFusion类的定义，包含两种构造函数
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoopFusion)
  LoopFusion() = default;
  LoopFusion(unsigned fastMemorySpace, uint64_t localBufSizeThresholdBytes,
             bool maximalFusion, enum FusionMode affineFusionMode) {
    this->fastMemorySpace = fastMemorySpace;
    this->localBufSizeThreshold = localBufSizeThresholdBytes / 1024;
    this->maximalFusion = maximalFusion;
    this->affineFusionMode = affineFusionMode;
  }

  void runOnBlock(Block *block);
  void runOnOperation() override;
  StringRef getArgument() const final { return "affine-loop-fusion-plus"; }
  StringRef getDescription() const final { 
    return "....."; 
  }
};

} // namespace

struct LoopNestStateCollectorPlus {
  SmallVector<AffineForOp, 4> forOps;
  // Affine loads.
  SmallVector<Operation *, 4> loadOpInsts;
  // Affine stores.
  SmallVector<Operation *, 4> storeOpInsts;
  // Non-affine loads.
  SmallVector<Operation *, 4> memrefLoads;
  // Non-affine stores.
  SmallVector<Operation *, 4> memrefStores;
  // Free operations.
  SmallVector<Operation *, 4> memrefFrees;
  
  // 收集循环嵌套状态的方法
  void collect(Operation *opToWalk) {
    opToWalk->walk([&](Operation *op) {
      if (auto forOp = dyn_cast<AffineForOp>(op)) {
        forOps.push_back(forOp);
      } else if (isa<AffineReadOpInterface>(op)) {
        loadOpInsts.push_back(op);
      } else if (isa<AffineWriteOpInterface>(op)) {
        storeOpInsts.push_back(op);
      } else {
        auto memInterface = dyn_cast<MemoryEffectOpInterface>(op);
        if (!memInterface) {
          if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
            // This op itself is memory-effect free.
            return;
          // Check operands. Eg. ops like the `call` op are handled here.
          for (Value v : op->getOperands()) {
            if (!isa<MemRefType>(v.getType()))
              continue;
            // Conservatively, we assume the memref is read and written to.
            memrefLoads.push_back(op);
            memrefStores.push_back(op);
          }
        } else {
          // Non-affine loads and stores.
          if (hasEffect<MemoryEffects::Read>(op))
            memrefLoads.push_back(op);
          if (hasEffect<MemoryEffects::Write>(op))
            memrefStores.push_back(op);
          if (hasEffect<MemoryEffects::Free>(op))
            memrefFrees.push_back(op);
        }
      }
    });
  }
};

class DirectedOpGraph {
  public:
    /// Add a node to the graph.
    void addNode(Operation *op) {
      assert(!hasNode(op) && "node already added");
      nodes.emplace_back(op);
      edges[op] = {};
    }
  
    /// Add an edge from `src` to `dest`.
    void addEdge(Operation *src, Operation *dest) {
      // This is a multi-graph.
      assert(hasNode(src) && "src node does not exist in graph");
      assert(hasNode(dest) && "dest node does not exist in graph");
      edges[src].push_back(getNode(dest));
    }
  
    /// Returns true if there is a (directed) cycle in the graph.
    bool hasCycle() { return dfs(/*cycleCheck=*/true); }
  
    void printEdges() {
      for (auto &en : edges) {
        llvm::dbgs() << *en.first << " (" << en.first << ")"
                     << " has " << en.second.size() << " edges:\n";
        for (auto *node : en.second) {
          llvm::dbgs() << '\t' << *node->op << '\n';
        }
      }
    }
  
  private:
    /// A node of a directed graph between MLIR Operations to model various
    /// relationships. This is meant to be used internally.
    struct DGNode {
      DGNode(Operation *op) : op(op) {};
      Operation *op;
  
      // Start and finish visit numbers are standard in DFS to implement things
      // like finding strongly connected components. These numbers are modified
      // during analyses on the graph and so seemingly const API methods will be
      // non-const.
  
      /// Start visit number.
      int vn = -1;
  
      /// Finish visit number.
      int fn = -1;
    };
  
    /// Get internal node corresponding to `op`.
    DGNode *getNode(Operation *op) {
      auto *value =
          llvm::find_if(nodes, [&](const DGNode &node) { return node.op == op; });
      assert(value != nodes.end() && "node doesn't exist in graph");
      return &*value;
    }
  
    /// Returns true if `key` is in the graph.
    bool hasNode(Operation *key) const {
      return llvm::find_if(nodes, [&](const DGNode &node) {
               return node.op == key;
             }) != nodes.end();
    }
  
    /// Perform a depth-first traversal of the graph setting visited and finished
    /// numbers. If `cycleCheck` is set, detects cycles and returns true as soon
    /// as the first cycle is detected, and false if there are no cycles. If
    /// `cycleCheck` is not set, completes the DFS and the `return` value doesn't
    /// have a meaning.
    bool dfs(bool cycleCheck = false) {
      for (DGNode &node : nodes) {
        node.vn = 0;
        node.fn = -1;
      }
  
      unsigned time = 0;
      for (DGNode &node : nodes) {
        if (node.vn == 0) {
          bool ret = dfsNode(node, cycleCheck, time);
          // Check if a cycle was already found.
          if (cycleCheck && ret)
            return true;
        } else if (cycleCheck && node.fn == -1) {
          // We have encountered a node whose visit has started but it's not
          // finished. So we have a cycle.
          return true;
        }
      }
      return false;
    }
  
    /// Perform depth-first traversal starting at `node`. Return true
    /// as soon as a cycle is found if `cycleCheck` was set. Update `time`.
    bool dfsNode(DGNode &node, bool cycleCheck, unsigned &time) const {
      auto nodeEdges = edges.find(node.op);
      assert(nodeEdges != edges.end() && "missing node in graph");
      node.vn = ++time;
  
      for (auto &neighbour : nodeEdges->second) {
        if (neighbour->vn == 0) {
          bool ret = dfsNode(*neighbour, cycleCheck, time);
          if (cycleCheck && ret)
            return true;
        } else if (cycleCheck && neighbour->fn == -1) {
          // We have encountered a node whose visit has started but it's not
          // finished. So we have a cycle.
          return true;
        }
      }
  
      // Update finish time.
      node.fn = ++time;
  
      return false;
    }
  
    // The list of nodes. The storage is owned by this class.
    SmallVector<DGNode> nodes;
  
    // Edges as an adjacency list.
    DenseMap<Operation *, SmallVector<DGNode *>> edges;
  };

bool hasCyclicDependence(AffineForOp root) {
  // Collect all the memory accesses in the source nest grouped by their
  // immediate parent block.
  DirectedOpGraph graph;
  SmallVector<MemRefAccess> accesses;
  root->walk([&](Operation *op) {
    if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op)) {
      accesses.emplace_back(op);
      graph.addNode(op);
    }
  });

  // Construct the dependence graph for all the collected acccesses.
  unsigned rootDepth = getNestingDepth(root);
  for (const auto &accA : accesses) {
    for (const auto &accB : accesses) {
      if (accA.memref != accB.memref)
        continue;
      // Perform the dependence on all surrounding loops + the body.
      unsigned numCommonLoops =
          getNumCommonSurroundingLoops(*accA.opInst, *accB.opInst);
      for (unsigned d = rootDepth + 1; d <= numCommonLoops + 1; ++d) {
        if (!noDependence(checkMemrefAccessDependence(accA, accB, d)))
          graph.addEdge(accA.opInst, accB.opInst);
      }
    }
  }
  return graph.hasCycle();
}

Block *findInnermostCommonBlockInScope(Operation *a,
                                                     Operation *b) {
  Region *aScope = mlir::affine::getAffineScope(a);
  Region *bScope = mlir::affine::getAffineScope(b);
  if (aScope != bScope)
    return nullptr;

  // Get the block ancestry of `op` while stopping at the affine scope `aScope`
  // and store them in `ancestry`.
  auto getBlockAncestry = [&](Operation *op,
                              SmallVectorImpl<Block *> &ancestry) {
    Operation *curOp = op;
    do {
      ancestry.push_back(curOp->getBlock());
      if (curOp->getParentRegion() == aScope)
        break;
      curOp = curOp->getParentOp();
    } while (curOp);
    assert(curOp && "can't reach root op without passing through affine scope");
    std::reverse(ancestry.begin(), ancestry.end());
  };

  SmallVector<Block *, 4> aAncestors, bAncestors;
  getBlockAncestry(a, aAncestors);
  getBlockAncestry(b, bAncestors);
  assert(!aAncestors.empty() && !bAncestors.empty() &&
         "at least one Block ancestor expected");

  Block *innermostCommonBlock = nullptr;
  for (unsigned a = 0, b = 0, e = aAncestors.size(), f = bAncestors.size();
       a < e && b < f; ++a, ++b) {
    if (aAncestors[a] != bAncestors[b])
      break;
    innermostCommonBlock = aAncestors[a];
  }
  return innermostCommonBlock;
}


/// Returns true if node 'srcId' can be removed after fusing it with node
/// 'dstId'. The node can be removed if any of the following conditions are met:
///   1. 'srcId' has no output dependences after fusion and no escaping memrefs.
///   2. 'srcId' has no output dependences after fusion, has escaping memrefs
///       and the fusion slice is maximal.
///   3. 'srcId' has output dependences after fusion, the fusion slice is
///      maximal and the fusion insertion point dominates all the dependences.
static bool canRemoveSrcNodeAfterFusion(
    unsigned srcId, unsigned dstId, const ComputationSliceState &fusionSlice,
    Operation *fusedLoopInsPoint, DenseSet<Value> &escapingMemRefs,
    const MemRefDependenceGraph &mdg) { //用于判断fusion之后是否可以移除源节点
// 参数包括：源节点ID、目标节点ID、融合切片状态、融合后循环的插入点、逃逸内存引用集合和内存引用依赖图

  // Operation *dstNodeOp = mdg.getNode(dstId)->op; // 获取目标节点的操作指针
  Operation *dstNodeOp = const_cast<MemRefDependenceGraph&>(mdg).getNode(dstId)->op;
  bool hasOutDepsAfterFusion = false; // 用于表示融合后源节点是否有输出依赖

  for (auto &outEdge : mdg.outEdges.lookup(srcId)) { // 遍历源节点的所有输出边（代表依赖关系）
    // Operation *depNodeOp = mdg.getNode(outEdge.id)->op; // 获取每个依赖节点的操作指针
    Operation *depNodeOp = const_cast<MemRefDependenceGraph&>(mdg).getNode(outEdge.id)->op;
    // Skip dependence with dstOp since it will be removed after fusion.
    if (depNodeOp == dstNodeOp) // 如果依赖节点就是目标节点,则跳过
      continue;

    // Only fusion within the same block is supported. Use domination analysis
    // when needed.
    if (depNodeOp->getBlock() != dstNodeOp->getBlock()) // 检查依赖节点和目标节点是否在同一个基本块中（目前不支持不同block的循环fusion）
      return false; // 如果不在同一个基本块，函数返回false（不能移除源节点）

    // Check if the insertion point of the fused loop dominates the dependence.
    // Otherwise, the src loop can't be removed.
    if (fusedLoopInsPoint != depNodeOp &&
        !fusedLoopInsPoint->isBeforeInBlock(depNodeOp)) { // 如果插入点不是依赖节点本身，且插入点不在依赖节点之前，则返回false,保证插入点在源节点的依赖节点之前才能保证不破坏原本的依赖关系
      LLVM_DEBUG(llvm::dbgs() << "Src loop can't be removed: dst loop doesn't "
                                 "dominate dependence\n");
      return false;
    }

    hasOutDepsAfterFusion = true;
  }

  // If src loop has dependences after fusion or it writes to an live-out or
  // escaping memref, we can only remove it if the fusion slice is maximal so
  // that all the dependences are preserved.
  if (hasOutDepsAfterFusion || !escapingMemRefs.empty()) { // 如果融合后源节点有输出依赖，或者有逃逸的内存引用，则需要额外检查
    std::optional<bool> isMaximal = fusionSlice.isMaximal();
    if (!isMaximal) { // 检查融合切片是否是最大的
      LLVM_DEBUG(llvm::dbgs() << "Src loop can't be removed: can't determine "
                                 "if fusion is maximal\n");
      return false;
    }

    if (!*isMaximal) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Src loop can't be removed: fusion is not maximal\n");
      return false;
    }
  }

  return true;
}

/// Returns in 'srcIdCandidates' the producer fusion candidates for consumer
/// 'dstId'. Candidates are sorted by node id order. This order corresponds to
/// the program order when the 'mdg' is created. However, program order is not
/// guaranteed and must not be required by the client. Program order won't be
/// held if the 'mdg' is reused from a previous fusion step or if the node
/// creation order changes in the future to support more advance cases.
// TODO: Move this to a loop fusion utility once 'mdg' is also moved.
static void getProducerCandidates(unsigned dstId,
                                  const MemRefDependenceGraph &mdg,
                                  SmallVectorImpl<unsigned> &srcIdCandidates) { // 该函数用于寻找可以与消费者节点（consumer）进行融合的生产者节点（producer）候选项
  // 参数：目标节点ID(dstId)、内存引用依赖图(mdg)和用于存储结果的候选节点ID向量(srcIdCandidates)
  // Skip if no input edges along which to fuse.
  if (mdg.inEdges.count(dstId) == 0) // 如果目标节点没有输入边（即没有其他节点向它提供数据），则直接返回
    return;

  // Gather memrefs from loads in 'dstId'.
  // auto *dstNode = mdg.getNode(dstId);
  auto *dstNode = const_cast<MemRefDependenceGraph&>(mdg).getNode(dstId);
  DenseSet<Value> consumedMemrefs; // 创建一个集合来存储目标节点读取的所有内存引用
  for (Operation *load : dstNode->loads) // 遍历目标节点的所有加载操作，将它们读取的内存引用添加到集合中
    consumedMemrefs.insert(cast<AffineReadOpInterface>(load).getMemRef());

  // Traverse 'dstId' incoming edges and gather the nodes that contain a store
  // to one of the consumed memrefs.
  for (const auto &srcEdge : mdg.inEdges.lookup(dstId)) { // 遍历目标节点的所有输入边（代表生产者-消费者关系）
    // const auto *srcNode = mdg.getNode(srcEdge.id); // 获取边连接的源节点（潜在的生产者）
    const auto *srcNode = const_cast<MemRefDependenceGraph&>(mdg).getNode(srcEdge.id);
    // Skip if 'srcNode' is not a loop nest.
    if (!isa<AffineForOp>(srcNode->op))
      continue;

    if (any_of(srcNode->stores, [&](Operation *op) { // 使用any_of算法检查源节点的存储操作是否有任何一个写入了目标节点读取的内存引用
          auto storeOp = cast<AffineWriteOpInterface>(op);
          return consumedMemrefs.count(storeOp.getMemRef()) > 0;
        }))
      srcIdCandidates.push_back(srcNode->id); // 如果有这样的存储操作，则将源节点ID添加到候选列表中
  }

  llvm::sort(srcIdCandidates); // 对候选ID列表进行排序（按照节点ID的数值顺序）
  srcIdCandidates.erase(llvm::unique(srcIdCandidates), srcIdCandidates.end()); // 使用unique算法移除重复的候选ID
}

/// Returns in 'producerConsumerMemrefs' the memrefs involved in a
/// producer-consumer dependence between 'srcId' and 'dstId'.
static void
gatherProducerConsumerMemrefs(unsigned srcId, unsigned dstId,
                              const MemRefDependenceGraph &mdg,
                              DenseSet<Value> &producerConsumerMemrefs) { 
// 该函数的作用是找出源节点(srcId)和目标节点(dstId)之间存在生产者-消费者依赖关系的内存引用，并将结果存储在producerConsumerMemrefs参数中。
  // auto *dstNode = mdg.getNode(dstId);
  auto *dstNode = const_cast<MemRefDependenceGraph&>(mdg).getNode(dstId);
  // auto *srcNode = mdg.getNode(srcId);
  auto *srcNode = const_cast<MemRefDependenceGraph&>(mdg).getNode(srcId);
  gatherProducerConsumerMemrefs(srcNode->stores, dstNode->loads,
                                producerConsumerMemrefs);
  // 函数的主要逻辑在被调用的重载函数中，该重载函数会检查哪些内存引用在一个节点中被写入(stores)，然后在另一个节点中被读取(loads)，这些内存引用就构成了节点之间的生产者-消费者关系。
}

/// A memref escapes in the context of the fusion pass if either:
///   1. it (or its alias) is a block argument, or
///   2. created by an op not known to guarantee alias freedom,
///   3. it (or its alias) are used by ops other than affine dereferencing ops
///   (e.g., by call op, memref load/store ops, alias creating ops, unknown ops,
///   terminator ops, etc.); such ops do not deference the memref in an affine
///   way.
static bool isEscapingMemref(Value memref, Block *block) { // 用于判断给定的内存引用(memref)是否在特定基本块(block)的上下文中逃逸
  Operation *defOp = memref.getDefiningOp(); // 获取定义这个内存引用的操作
  // Check if 'memref' is a block argument.
  if (!defOp) // 如果没有定义操作(defOp为空)，则这个内存引用可能是一个基本块参数，返回true表示它是逃逸的
    return true;

  // Check if this is defined to be an alias of another memref.
  if (auto viewOp = dyn_cast<mlir::ViewLikeOpInterface>(defOp)) // 检查定义操作是否是一个"视图操作"(如reshape, slice等),视图操作创建了对现有内存引用的新视图，实际是一个别名
    if (isEscapingMemref(viewOp.getViewSource(), block)) // 如果是视图操作，递归检查原始内存引用(viewSource)是否逃逸
      return true;

  // Any op besides allocating ops wouldn't guarantee alias freedom
  if (!hasSingleEffect<mlir::MemoryEffects::Allocate>(defOp, memref)) // 检查定义操作是否仅有分配内存的效果,如果不是纯粹的分配操作，就无法保证别名自由
    return true;

  // Check if 'memref' is used by a non-deferencing op (including unknown ones)
  // (e.g., call ops, alias creating ops, etc.).
  return llvm::any_of(memref.getUsers(), [&](Operation *user) { // 检查使用这个内存引用的所有操作，看是否有任何一个操作以非仿射方式访问它
    // Ignore users outside of `block`.
    Operation *ancestorOp = block->getParent()->findAncestorOpInRegion(*user); // 寻找用户操作在当前区域中的祖先操作
    if (!ancestorOp) // 如果找不到祖先操作，说明用户在当前区域之外，返回true表示逃逸
      return true;
    if (ancestorOp->getBlock() != block) // 如果祖先操作(嵌套层次)不在当前检查的基本块中，忽略这个用户
      return false;
    return !isa<AffineMapAccessInterface>(*user); // 检查用户操作是否不是仿射访问接口的实例
  });
}

/// Returns in 'escapingMemRefs' the memrefs from affine store ops in node 'id'
/// that escape the block or are accessed in a non-affine way.
// 找出节点'id'中的仿射存储操作中使用的内存引用，这些内存引用要么逃逸出当前基本块，要么以非仿射方式被访问。
static void gatherEscapingMemrefs(unsigned id, const MemRefDependenceGraph &mdg,
                                  DenseSet<Value> &escapingMemRefs) { // 用于收集从给定节点中"逃逸"的内存引用
  // auto *node = mdg.getNode(id);
  auto *node = const_cast<MemRefDependenceGraph&>(mdg).getNode(id);
  for (Operation *storeOp : node->stores) { // 遍历节点中的所有存储操作（写入内存的操作）
    auto memref = cast<AffineWriteOpInterface>(storeOp).getMemRef();
    if (escapingMemRefs.count(memref)) // 检查这个内存引用是否已经在结果集合中,如果已收集，则跳过
      continue;
    if (isEscapingMemref(memref, &mdg.block)) // 调用之前定义的isEscapingMemref函数，检查内存引用是否逃逸
      escapingMemRefs.insert(memref);
  }
}

// Sinks all sequential loops to the innermost levels (while preserving
// relative order among them) and moves all parallel loops to the
// outermost (while again preserving relative order among them).
// This can increase the loop depth at which we can fuse a slice, since we are
// pushing loop carried dependence to a greater depth in the loop nest.
// 将所有顺序（sequential）循环下沉到最内层，将所有并行（parallel）循环提升到最外层，同时在各自组内保持原有的相对顺序。
static void sinkSequentialLoops(MemRefDependenceGraph::Node *node) {
  assert(isa<AffineForOp>(node->op)); // 断言确保节点的操作是一个仿射循环（AffineForOp）
  AffineForOp newRootForOp = sinkSequentialLoops(cast<AffineForOp>(node->op));
  // 调用另一个同名但参数不同的sinkSequentialLoops函数
  // 将节点的操作转换为AffineForOp类型并传递给重载函数
  // 该重载函数执行实际的循环重排序，并返回新的根循环操作
  node->op = newRootForOp; // 更新节点，使其指向重排序后的新根循环操作
}

/// Get the operation that should act as a dominance filter while replacing
/// memref uses with a private memref for which `producerStores` and
/// `sliceInsertionBlock` are provided. This effectively determines in what
/// part of the IR we should be performing the replacement.
static Operation *
getDominanceFilterForPrivateMemRefRepl(Block *sliceInsertionBlock,
                                       ArrayRef<Operation *> producerStores) { // 这个函数帮助编译器确定"在IR（中间表示）的哪部分执行内存引用替换"
  // 在fusion之后，一定区域内的op需要更新它所操作的内存地址（引用）
  // 参数：sliceInsertionBlock：融合切片插入的基本块，producerStores：生产者存储操作的数组引用
  assert(!producerStores.empty() && "expected producer store");

  // We first find the common block that contains the producer stores and
  // the slice computation. The first ancestor among the ancestors of the
  // producer stores in that common block is the dominance filter to use for
  // replacement.
  // 寻找包含生产者存储操作和切片计算的公共基本块，然后找到生产者存储操作在该公共块中的第一个祖先操作
  Block *commonBlock = nullptr; // 用于存储公共基本块
  // Find the common block of all relevant operations.
  for (Operation *store : producerStores) { // 循环遍历所有生产者存储操作,寻找公共基本块
    Operation *otherOp =
        !commonBlock ? &*sliceInsertionBlock->begin() : &*commonBlock->begin();
    commonBlock = findInnermostCommonBlockInScope(store, otherOp);
  }
  assert(commonBlock &&
         "common block of producer stores and slice should exist");

  // Find the first ancestor among the ancestors of `producerStores` in
  // `commonBlock`.
  // 在公共块中找到生产者存储操作的最早祖先操作
  Operation *firstAncestor = nullptr;
  for (Operation *store : producerStores) { // 遍历所有生产者存储操作
    Operation *ancestor = commonBlock->findAncestorOpInBlock(*store); // 对于每个存储操作，找到它在公共块中的祖先操作
    assert(ancestor && "producer store should be contained in common block");
    firstAncestor = !firstAncestor || ancestor->isBeforeInBlock(firstAncestor)
                        ? ancestor
                        : firstAncestor;
    // 如果firstAncestor尚未设置，使用当前找到的祖先
    // 如果当前祖先在现有firstAncestor之前（在块内的顺序），更新为当前祖先
  }
  // 核心逻辑是找到一个能够支配所有生产者存储操作的操作
  return firstAncestor;
}

/// Returns the amount of additional (redundant) computation that will be done
/// as a fraction of the total computation if `srcForOp` is fused into
/// `dstForOp` at depth `depth`. The method returns the compute cost of the
/// slice and the fused nest's compute cost in the trailing output arguments.
static std::optional<double> getAdditionalComputeFraction(
    AffineForOp srcForOp, AffineForOp dstForOp, unsigned depth,
    ArrayRef<ComputationSliceState> depthSliceUnions, int64_t &sliceCost,
    int64_t &fusedLoopNestComputeCost) { // 该函数用于计算循环融合操作可能引入的计算冗余量,返回融合后引入的额外计算量，以占总计算量的比例表示，返回类型是可选的双精度浮点数
  // 参数：源循环、目标循环、融合深度、深度切片联合数组、切片代价和融合后循环嵌套计算代价
  LLVM_DEBUG(llvm::dbgs() << "Determining additional compute fraction...\n";);
  // Compute cost of sliced and unsliced src loop nest.
  // Walk src loop nest and collect stats.
  LoopNestStats srcLoopNestStats; // 用于存储源循环嵌套的统计信息
  if (!getLoopNestStats(srcForOp, &srcLoopNestStats)) { // 调用getLoopNestStats函数收集源循环嵌套的统计数据
    LLVM_DEBUG(llvm::dbgs() << "Failed to get source loop nest stats.\n");
    return std::nullopt;
  }

  // Compute cost of dst loop nest.
  LoopNestStats dstLoopNestStats;
  if (!getLoopNestStats(dstForOp, &dstLoopNestStats)) { // 类似地，为目标循环嵌套收集统计信息
    LLVM_DEBUG(llvm::dbgs() << "Failed to get destination loop nest stats.\n");
    return std::nullopt;
  }

  // Compute op instance count for the src loop nest without iteration slicing.
  uint64_t srcLoopNestCost = getComputeCost(srcForOp, srcLoopNestStats); // 计算源循环嵌套的计算代价（不考虑迭代切片）

  // Compute op cost for the dst loop nest.
  uint64_t dstLoopNestCost = getComputeCost(dstForOp, dstLoopNestStats); // 计算目标循环嵌套的计算代价

  const ComputationSliceState &slice = depthSliceUnions[depth - 1]; // 获取指定融合深度的计算切片状态
  // Skip slice union if it wasn't computed for this depth.
  if (slice.isEmpty()) { // 如果该深度的切片为空（未计算），输出调试信息并返回空值
    LLVM_DEBUG(llvm::dbgs() << "Slice wasn't computed.\n");
    return std::nullopt;
  }

  if (!getFusionComputeCost(srcForOp, srcLoopNestStats, dstForOp,
                            dstLoopNestStats, slice,
                            &fusedLoopNestComputeCost)) { // 计算融合后的循环嵌套计算代价，结果存储在fusedLoopNestComputeCost中
    LLVM_DEBUG(llvm::dbgs() << "Unable to compute fusion compute cost\n");
    return std::nullopt;
  }

  double additionalComputeFraction =
      fusedLoopNestComputeCost /
          (static_cast<double>(srcLoopNestCost) + dstLoopNestCost) -
      1; // 计算额外计算量的比例,减去1（如果结果大于0表示有冗余计算，等于0表示没有额外计算）

  return additionalComputeFraction;
}

// Creates and returns a private (single-user) memref for fused loop rooted at
// 'forOp', with (potentially reduced) memref size based on the memref region
// written to by `storeOps` at depth 'dstLoopDepth'. 'sliceInsertionBlock'
// specifies the block in which the slice was/will be inserted.
static Value createPrivateMemRef(AffineForOp forOp,
                                 ArrayRef<Operation *> storeOps,
                                 unsigned dstLoopDepth,
                                 std::optional<unsigned> fastMemorySpace,
                                 Block *sliceInsertionBlock,
                                 uint64_t localBufSizeThreshold) {
  // 参数：融合循环、存储操作数组、目标循环深度、可选的快速内存空间、切片插入基本块、本地缓冲区大小阈值
  assert(!storeOps.empty() && "no source stores supplied");
  Operation *srcStoreOp = storeOps[0]; // 获取第一个源存储操作作为参考

  // 创建两个操作构建器
  // 一个用于在循环前插入分配操作
  // 一个用于在顶层区域创建常量
  // Create builder to insert alloc op just before 'forOp'.
  OpBuilder b(forOp);
  // Builder to create constants at the top level.
  OpBuilder top(forOp->getParentRegion());
  // Create new memref type based on slice bounds.
  // 获取原始内存引用及其类型
  auto oldMemRef = cast<AffineWriteOpInterface>(srcStoreOp).getMemRef();
  auto oldMemRefType = cast<MemRefType>(oldMemRef.getType());
  unsigned rank = oldMemRefType.getRank(); // 获取内存引用的秩（维度数量）

  // Compute MemRefRegion for 'srcStoreOpInst' at depth 'dstLoopDepth'.
  MemRefRegion region(srcStoreOp->getLoc()); // 为源存储操作创建一个内存引用区域对象
  bool validRegion = succeeded(region.compute(srcStoreOp, dstLoopDepth)); // 计算该操作在指定深度访问的内存区域
  (void)validRegion;
  assert(validRegion && "unexpected memref region failure");
  // 声明变量用于存储新形状、下界和下界除数
  SmallVector<int64_t, 4> newShape;
  std::vector<SmallVector<int64_t, 4>> lbs;
  SmallVector<int64_t, 8> lbDivisors;
  lbs.reserve(rank);
  // Query 'region' for 'newShape' and lower bounds of MemRefRegion accessed
  // by 'srcStoreOpInst' at depth 'dstLoopDepth'.
  std::optional<int64_t> numElements =
      region.getConstantBoundingSizeAndShape(&newShape, &lbs, &lbDivisors); // 查询内存区域，获取访问区域的新形状、下界和大小
  assert(numElements && "non-constant number of elts in local buffer"); // 断言确保元素数量是常量

  const FlatAffineValueConstraints *cst = region.getConstraints(); // 获取内存区域的约束条件
  // 'outerIVs' holds the values that this memory region is symbolic/parametric
  // on; this would correspond to loop IVs surrounding the level at which the
  // slice is being materialized.
  SmallVector<Value, 8> outerIVs;
  cst->getValues(rank, cst->getNumVars(), &outerIVs); // 收集外层迭代变量（IVs），这些变量是内存区域的符号/参数

  // Build 'rank' AffineExprs from MemRefRegion 'lbs'
  // 从内存区域下界构建仿射表达式偏移量
  SmallVector<AffineExpr, 4> offsets;
  offsets.reserve(rank);
  for (unsigned d = 0; d < rank; ++d) {
    assert(lbs[d].size() == cst->getNumCols() - rank && "incorrect bound size");

    AffineExpr offset = top.getAffineConstantExpr(0);
    for (unsigned j = 0, e = cst->getNumCols() - rank - 1; j < e; j++) {
      offset = offset + lbs[d][j] * top.getAffineDimExpr(j);
    }
    assert(lbDivisors[d] > 0);
    offset =
        (offset + lbs[d][cst->getNumCols() - 1 - rank]).floorDiv(lbDivisors[d]);
    offsets.push_back(offset);
  }

  // Create 'newMemRefType' using 'newShape' from MemRefRegion accessed
  // by 'srcStoreOpInst'.
  auto eltSize = getMemRefIntOrFloatEltSizeInBytes(oldMemRefType); // 获取元素大小（字节数）
  assert(eltSize && "memrefs with size elt types expected");
  uint64_t bufSize = *eltSize * *numElements; // 计算总缓冲区大小（元素大小 × 元素数量）
  Attribute newMemSpace;
  if (bufSize <= localBufSizeThreshold && fastMemorySpace.has_value()) { // 如果缓冲区足够小并且提供了快速内存空间选项，使用快速内存
    newMemSpace = b.getI64IntegerAttr(*fastMemorySpace);
  } else { // 否则保持原有的内存空间
    newMemSpace = oldMemRefType.getMemorySpace();
  }
  auto newMemRefType = MemRefType::get(newShape, oldMemRefType.getElementType(),
                                       /*map=*/AffineMap(), newMemSpace); // 创建新的内存引用类型，使用新形状、原始元素类型和确定的内存空间

  // Create new private memref for fused loop 'forOp'. 'newShape' is always
  // a constant shape.
  // TODO: Create/move alloc ops for private memrefs closer to their
  // consumer loop nests to reduce their live range. Currently they are added
  // at the beginning of the block, because loop nests can be reordered
  // during the fusion pass.
  Value newMemRef = top.create<memref::AllocOp>(forOp.getLoc(), newMemRefType); // 在顶层创建新的私有内存引用分配操作

  // Build an AffineMap to remap access functions based on lower bound offsets.
  // 构建仿射映射，用于重映射访问函数
  SmallVector<AffineExpr, 4> remapExprs;
  remapExprs.reserve(rank);
  for (unsigned i = 0; i < rank; i++) { // 对每个维度，基于偏移量创建重映射表达式
    auto dimExpr = b.getAffineDimExpr(outerIVs.size() + i);

    auto remapExpr =
        simplifyAffineExpr(dimExpr - offsets[i], outerIVs.size() + rank, 0);
    remapExprs.push_back(remapExpr);
  }

  auto indexRemap =
      AffineMap::get(outerIVs.size() + rank, 0, remapExprs, forOp.getContext()); // 创建完整的索引重映射仿射映射

  // Replace all users of 'oldMemRef' with 'newMemRef'.
  Operation *domFilter =
      getDominanceFilterForPrivateMemRefRepl(sliceInsertionBlock, storeOps); // 获取内存引用替换的支配过滤器，确定替换范围
  LogicalResult res = replaceAllMemRefUsesWith(
      oldMemRef, newMemRef, /*extraIndices=*/{}, indexRemap,
      /*extraOperands=*/outerIVs,
      /*symbolOperands=*/{}, domFilter); // 使用新内存引用替换所有原内存引用的使用
  assert(succeeded(res) &&
         "replaceAllMemrefUsesWith should always succeed here");
  (void)res;
  return newMemRef; // 返回新创建的私有内存引用
}

// Checks the profitability of fusing a backwards slice of the loop nest
// `srcForOp` into the loop nest surrounding 'dstLoadOpInsts'. The argument
// 'srcStoreOpInst' is used to calculate the storage reduction on the memref
// being produced and consumed, which is an input to the cost model. For
// producer-consumer fusion, 'srcStoreOpInst' will be the same as 'srcOpInst',
// as we are slicing w.r.t to that producer. For input-reuse fusion, 'srcOpInst'
// will be the src loop nest LoadOp which reads from the same memref as dst loop
// nest load ops, and 'srcStoreOpInst' will be the unique store op in the src
// node, which will be used to check that the write region is the same after
// input-reuse fusion. Computation slices are provided in 'depthSliceUnions' for
// each legal fusion depth. The maximal depth at which fusion is legal is
// provided in 'maxLegalFusionDepth'. Returns true if it is profitable to fuse
// the candidate loop nests. Returns false otherwise. `dstLoopDepth` is set to
// the most profitable depth at which to materialize the source loop nest slice.
// The profitability model executes the following steps:
// *) Computes the backward computation slice at 'srcOpInst'. This
//    computation slice of the loop nest surrounding 'srcOpInst' is
//    represented by modified src loop bounds in 'sliceState', which are
//    functions of loop IVs in the loop nest surrounding 'srcOpInst'.
// *) Computes the cost of unfused src/dst loop nests (currently the cost of a
//    loop nest is the total number of dynamic operation instances in the loop
//    nest).
// *) Computes the cost of fusing a slice of the src loop nest into the dst
//    loop nest at various values of dst loop depth, attempting to fuse
//    the largest computation slice at the maximal dst loop depth (closest to
//    the load) to minimize reuse distance and potentially enable subsequent
//    load/store forwarding.
//    NOTE: 'dstLoopDepth' refers to the loop depth within the destination loop
//    nest, at which the src computation slice is inserted/fused.
//    NOTE: We attempt to maximize the dst loop depth, but there are cases
//    where a particular setting for 'dstLoopNest' might fuse an unsliced
//    loop (within the src computation slice) at a depth which results in
//    excessive recomputation (see unit tests for examples).
// *) Compares the total cost of the unfused loop nests to the min cost fused
//    loop nest computed in the previous step, and returns true if the latter
//    is lower.
// TODO: Extend profitability analysis to support scenarios with multiple
// stores.
static bool isFusionProfitable(AffineForOp srcForOp,
                               ArrayRef<Operation *> producerStores,
                               AffineForOp dstForOp,
                               ArrayRef<ComputationSliceState> depthSliceUnions,
                               unsigned maxLegalFusionDepth,
                               unsigned *dstLoopDepth,
                               double computeToleranceThreshold) { // 用于评估将循环嵌套的向后切片融合到另一个循环嵌套中的盈利性
  // 参数包括：源循环、生产者存储操作数组、目标循环、各深度的计算切片状态、最大合法融合深度、目标循环深度指针（输出参数）和计算冗余容忍阈值
  LLVM_DEBUG({
    llvm::dbgs()
        << "Checking whether fusion is profitable between source nest:\n";
    llvm::dbgs() << ' ' << srcForOp << " and destination nest:\n";
    llvm::dbgs() << dstForOp << "\n";
  });

  if (maxLegalFusionDepth == 0) { // 如果最大合法融合深度为0，无法融合，返回false
    LLVM_DEBUG(llvm::dbgs() << "Can't fuse: maxLegalFusionDepth is 0\n");
    return false;
  }

  // Compute cost of sliced and unsliced src loop nest.

  // Walk src loop nest and collect stats.
  LoopNestStats srcLoopNestStats;
  if (!getLoopNestStats(srcForOp, &srcLoopNestStats)) // 收集源循环嵌套的统计信息
    return false;

  // Compute cost of dst loop nest.
  LoopNestStats dstLoopNestStats;
  if (!getLoopNestStats(dstForOp, &dstLoopNestStats)) // 收集目标循环嵌套的统计信息
    return false;

  // We limit profitability analysis to only scenarios with
  // a single producer store for now. Note that some multi-store
  // producer scenarios will still go through profitability analysis
  // if only one of the stores is involved in the producer-consumer
  // relationship of the candidate loops.
  // TODO: Suppport multiple producer stores in profitability
  // analysis.
  if (producerStores.size() > 1) { // 对于多个生产者存储操作的情况，使用有限的盈利性分析
    LLVM_DEBUG(llvm::dbgs() << "Limited profitability analysis. Not "
                               "supported for multiple producer store case.\n");
    int64_t sliceCost;
    int64_t fusedLoopNestComputeCost;
    // We will still fuse if fusion obeys the specified compute
    // tolerance at the max legal depth.
    auto fraction = getAdditionalComputeFraction(
        srcForOp, dstForOp, maxLegalFusionDepth, depthSliceUnions, sliceCost,
        fusedLoopNestComputeCost); // 在多存储操作情况下，计算在最大合法深度融合的额外计算比例
    if (!fraction || fraction > computeToleranceThreshold) { // 如果无法计算额外计算比例，或者比例超过容忍阈值，返回false
      LLVM_DEBUG(llvm::dbgs() << "Additional computation exceeds "
                                 "compute tolerance. Not fusing.\n");
      return false;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "Considering fusion profitable at max legal depth.\n"); // 在多存储情况下，如果额外计算在容忍范围内，认为融合有利
    return true;
  }

  Operation *srcStoreOp = producerStores.front(); // 获取第一个（在单存储情况下是唯一的）生产者存储操作

  // Search for min cost value for 'dstLoopDepth'. At each value of
  // 'dstLoopDepth' from 'maxLegalLoopDepth' to '1', compute computation slice
  // bounds between 'srcOpInst' and each op in 'dstOpinsts' (taking the union
  // of these bounds). Next the union slice bounds are used to calculate
  // the cost of the slice and the cost of the slice inserted into the dst
  // loop nest at 'dstLoopDepth'.
  // 初始化变量，用于搜索最优融合深度
  uint64_t minFusedLoopNestComputeCost = std::numeric_limits<uint64_t>::max(); // 最小融合循环嵌套计算成本
  double maxStorageReduction = 0.0; // 最大存储减少比例
  std::optional<uint64_t> sliceMemEstimate; // 切片内存估计值

  // The best loop depth at which to materialize the slice.
  std::optional<unsigned> bestDstLoopDepth; // 存储找到的最佳目标循环深度

  // Compute src loop nest write region size.
  MemRefRegion srcWriteRegion(srcStoreOp->getLoc());
  if (failed(srcWriteRegion.compute(srcStoreOp, /*loopDepth=*/0))) { // 计算源循环嵌套的写入区域
    LLVM_DEBUG(llvm::dbgs()
               << "Unable to compute MemRefRegion for source operation\n");
    return false;
  }

  std::optional<int64_t> maybeSrcWriteRegionSizeBytes =
      srcWriteRegion.getRegionSize(); // 获取源写入区域的大小（字节）
  if (!maybeSrcWriteRegionSizeBytes.has_value()) // 如果无法获取，返回false
    return false;
  int64_t srcWriteRegionSizeBytes = *maybeSrcWriteRegionSizeBytes;

  // Compute op instance count for the src loop nest without iteration slicing.
  uint64_t srcLoopNestCost = getComputeCost(srcForOp, srcLoopNestStats); // 计算源循环嵌套的计算成本（不考虑切片）

  // Compute op instance count for the destination loop nest.
  uint64_t dstLoopNestCost = getComputeCost(dstForOp, dstLoopNestStats); // 计算目标循环嵌套的计算成本

  // Evaluate all depth choices for materializing the slice in the destination
  // loop nest.
  for (unsigned i = maxLegalFusionDepth; i >= 1; --i) { // 从最大合法融合深度开始，向下遍历所有可能的融合深度
    const ComputationSliceState &slice = depthSliceUnions[i - 1]; // 获取当前深度的计算切片状态
    // Skip slice union if it wasn't computed for this depth.
    if (slice.isEmpty()) // 如果此深度没有计算切片，跳过
      continue;

    // Compute cost of the slice separately, i.e, the compute cost of the slice
    // if all outer trip counts are one.
    // 声明变量，用于存储切片成本和融合循环嵌套计算成本
    int64_t sliceCost;

    int64_t fusedLoopNestComputeCost;

    auto mayAdditionalComputeFraction =
        getAdditionalComputeFraction(srcForOp, dstForOp, i, depthSliceUnions,
                                     sliceCost, fusedLoopNestComputeCost); // 计算在当前深度融合会导致的额外计算比例
    if (!mayAdditionalComputeFraction) { // 如果无法计算，跳过当前深度
      LLVM_DEBUG(llvm::dbgs()
                 << "Can't determine additional compute fraction.\n");
      continue;
    }
    double additionalComputeFraction = *mayAdditionalComputeFraction;

    // Determine what the slice write MemRefRegion would be, if the src loop
    // nest slice 'slice' were to be inserted into the dst loop nest at loop
    // depth 'i'.
    MemRefRegion sliceWriteRegion(srcStoreOp->getLoc());
    if (failed(sliceWriteRegion.compute(srcStoreOp, /*loopDepth=*/0, &slice))) { // 计算切片的写入区域,如果计算失败，跳过当前深度
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to compute slice write region at loopDepth: " << i
                 << "\n");
      continue;
    }

    std::optional<int64_t> maybeSliceWriteRegionSizeBytes =
        sliceWriteRegion.getRegionSize(); // 获取切片写入区域的大小（字节）
    if (!maybeSliceWriteRegionSizeBytes.has_value() ||
        *maybeSliceWriteRegionSizeBytes == 0) { // 如果无法获取或大小为0，跳过当前深度
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to get slice write region size at loopDepth: " << i
                 << "\n");
      continue;
    }
    int64_t sliceWriteRegionSizeBytes = *maybeSliceWriteRegionSizeBytes;

    double storageReduction = static_cast<double>(srcWriteRegionSizeBytes) /
                              static_cast<double>(sliceWriteRegionSizeBytes); // 计算存储减少比例：源写入区域大小除以切片写入区域大小

    LLVM_DEBUG({ // 输出调试信息，包括当前深度的评估结果
      std::stringstream msg;
      msg << "  evaluating fusion profitability at depth : " << i << "\n"
          << std::fixed << std::setprecision(2)
          << "   additional compute fraction: "
          << 100.0 * additionalComputeFraction << "%\n"
          << "   storage reduction factor: " << storageReduction << "x\n"
          << "   fused nest cost: " << fusedLoopNestComputeCost << "\n"
          << "   src write region size: " << srcWriteRegionSizeBytes << "\n"
          << "   slice write region size: " << sliceWriteRegionSizeBytes
          << "\n";
      llvm::dbgs() << msg.str();
    });

    // TODO: This is a placeholder cost model.
    // Among all choices that add an acceptable amount of redundant computation
    // (as per computeToleranceThreshold), we will simply pick the one that
    // reduces the intermediary size the most.
    if ((storageReduction > maxStorageReduction) &&
        (additionalComputeFraction <= computeToleranceThreshold)) { // 成本模型：在额外计算不超过阈值的所有选择中，选择存储减少最大的那个
      maxStorageReduction = storageReduction;
      bestDstLoopDepth = i;
      minFusedLoopNestComputeCost = fusedLoopNestComputeCost;
      sliceMemEstimate = sliceWriteRegionSizeBytes;
    }
  }

  // A simple cost model: fuse if it reduces the memory footprint.

  if (!bestDstLoopDepth) { // 如果没有找到最佳深度（所有选择都超过计算容忍阈值），返回false
    LLVM_DEBUG(
        llvm::dbgs()
        << "All fusion choices involve more than the threshold amount of "
           "redundant computation; NOT fusing.\n");
    return false;
  }

  if (!bestDstLoopDepth) { // 重复检查，确保找到了最佳深度
    LLVM_DEBUG(llvm::dbgs() << "no fusion depth could be evaluated.\n");
    return false;
  }

  // Set dstLoopDepth based on best values from search.
  *dstLoopDepth = *bestDstLoopDepth; // 设置输出参数dstLoopDepth为找到的最佳深度

  LLVM_DEBUG( // 输出调试信息，包括最佳融合深度和相关计算成本
      llvm::dbgs() << " LoopFusion fusion stats:"
                   << "\n  best loop depth: " << bestDstLoopDepth
                   << "\n  src loop nest compute cost: " << srcLoopNestCost
                   << "\n  dst loop nest compute cost: " << dstLoopNestCost
                   << "\n  fused loop nest compute cost: "
                   << minFusedLoopNestComputeCost << "\n");

  // 获取源循环和目标循环的内存占用（字节）
  auto dstMemSize = getMemoryFootprintBytes(dstForOp);
  auto srcMemSize = getMemoryFootprintBytes(srcForOp);

  std::optional<double> storageReduction;

  if (!dstMemSize || !srcMemSize) { // 如果无法获取内存占用，返回false
    LLVM_DEBUG(llvm::dbgs()
               << "  fusion memory benefit cannot be evaluated; NOT fusing.\n");
    return false;
  }

  // 获取内存占用的实际值
  auto srcMemSizeVal = *srcMemSize;
  auto dstMemSizeVal = *dstMemSize;

  assert(sliceMemEstimate && "expected value");
  auto fusedMem = dstMemSizeVal + *sliceMemEstimate; // 计算融合后的内存占用：目标循环内存占用加上切片内存占用

  LLVM_DEBUG(llvm::dbgs() << "   src mem: " << srcMemSizeVal << "\n"
                          << "   dst mem: " << dstMemSizeVal << "\n"
                          << "   fused mem: " << fusedMem << "\n"
                          << "   slice mem: " << sliceMemEstimate << "\n");

  if (static_cast<long>(fusedMem) > srcMemSizeVal + dstMemSizeVal) { // 如果融合后的内存占用超过源循环和目标循环内存占用之和，认为融合不盈利，返回false
    LLVM_DEBUG(llvm::dbgs() << "Fusion is not profitable; NOT fusing.\n");
    return false;
  }
  storageReduction =
      100.0 *
      (1.0 - fusedMem / (static_cast<double>(srcMemSizeVal) + dstMemSizeVal)); // 计算存储减少百分比

  double additionalComputeFraction =
      100.0 * (minFusedLoopNestComputeCost /
                   (static_cast<double>(srcLoopNestCost) + dstLoopNestCost) -
               1); // 计算额外计算比例（百分比形式）
  (void)additionalComputeFraction;
  LLVM_DEBUG({
    std::stringstream msg;
    msg << " fusion is most profitable at depth " << *dstLoopDepth << " with "
        << std::setprecision(2) << additionalComputeFraction
        << "% redundant computation and a ";
    msg << (storageReduction ? std::to_string(*storageReduction) : "<unknown>");
    msg << "% storage reduction.\n";
    llvm::dbgs() << msg.str();
  });

  return true;
}

namespace {

// GreedyFusion greedily fuses loop nests which have a producer/consumer or
// input-reuse relationship on a memref, with the goal of improving locality.
//
// The steps of the producer-consumer fusion algorithm are as follows:
//
// *) A worklist is initialized with node ids from the dependence graph.
// *) For each node id in the worklist:
//   *) Pop an AffineForOp of the worklist. This 'dstAffineForOp' will be a
//      candidate destination AffineForOp into which fusion will be attempted.
//   *) Add each LoadOp currently in 'dstAffineForOp' into list 'dstLoadOps'.
//   *) For each LoadOp in 'dstLoadOps' do:
//      *) Look up dependent loop nests which have a single store op to the same
//         memref.
//      *) Check if dependences would be violated by the fusion.
//      *) Get a computation slice of 'srcLoopNest', which adjusts its loop
//         bounds to be functions of 'dstLoopNest' IVs and symbols.
//      *) Fuse the 'srcLoopNest' computation slice into the 'dstLoopNest',
//         at a loop depth determined by the cost model in 'isFusionProfitable'.
//      *) Add the newly fused load/store operations to the state,
//         and also add newly fused load ops to 'dstLoopOps' to be considered
//         as fusion dst load ops in another iteration.
//      *) Remove old src loop nest and its associated state.
//
// The steps of the input-reuse fusion algorithm are as follows:
//
// *) Initialize 'worklist' with node ids from the dependence graph.
// *) For each 'dstNode' in the worklist:
//   *) Find a candidate sibling node 'sibNode' to fuse with 'dstNode' which
//      loads from the same memref, but which has no dependence paths to/from.
//   *) Get a computation slice of 'sibLoopNest', which adjusts its loop
//      bounds to be functions of 'dstLoopNest' IVs and symbols.
//   *) Fuse the 'sibLoopNest' computation slice into the 'dstLoopNest',
//      at a loop depth determined by the cost model in 'isFusionProfitable'.
//      This function also checks that the memref write region of 'sibLoopNest',
//      is preserved in the fused loop nest.
//   *) Update graph state to reflect the fusion of 'sibNode' into 'dstNode'.
//
// Given a graph where top-level operations are vertices in the set 'V' and
// edges in the set 'E' are dependences between vertices, this algorithm
// takes O(V) time for initialization, and has runtime O(V + E).
//
// This greedy algorithm is not 'maximal' due to the current restriction of
// fusing along single producer consumer edges, but there is a TODO: to fix
// this.
//
// TODO: Experiment with other fusion policies.
struct GreedyFusion {
public:
  // The data dependence graph to traverse during fusion.
  MemRefDependenceGraph *mdg; // 数据依赖图
  // Worklist of graph nodes visited during the fusion pass.
  SmallVector<unsigned, 8> worklist; // 用于存储融合过程中要访问的图节点ID
  // Parameter for local buffer size threshold.
  unsigned localBufSizeThreshold; // 定义本地缓冲区大小阈值参数
  // Parameter for fast memory space.
  std::optional<unsigned> fastMemorySpace; // 定义快速内存空间参数，它是一个可选值
  // If true, ignore any additional (redundant) computation tolerance threshold
  // that would have prevented fusion.
  bool maximalFusion;
  // The amount of additional computation that is tolerated while fusing
  // pair-wise as a fraction of the total computation.
  double computeToleranceThreshold; // 表示融合时可容忍的额外计算比例

  using Node = MemRefDependenceGraph::Node; // 定义类型别名，将MemRefDependenceGraph::Node简化为Node

  GreedyFusion(MemRefDependenceGraph *mdg, unsigned localBufSizeThreshold,
               std::optional<unsigned> fastMemorySpace, bool maximalFusion,
               double computeToleranceThreshold) // 定义结构体的构造函数，接收所有需要的参数
      : mdg(mdg), localBufSizeThreshold(localBufSizeThreshold),
        fastMemorySpace(fastMemorySpace), maximalFusion(maximalFusion),
        computeToleranceThreshold(computeToleranceThreshold) {} // 使用初始化列表来初始化所有成员变量

  /// Initializes 'worklist' with nodes from 'mdg'.
  void init() {
    // TODO: Add a priority queue for prioritizing nodes by different
    // metrics (e.g. arithmetic intensity/flops-to-bytes ratio).
    worklist.clear();
    for (auto &idAndNode : mdg->nodes) { // 遍历依赖图中的所有节点,往worklist中存储依赖图中所有节点的ID
      const Node &node = idAndNode.second; // 对于每个节点，获取其ID并添加到工作列表中
      worklist.push_back(node.id);
    }
  }
  /// Run only sibling fusion on the `mdg`.
  void runSiblingFusionOnly() {
    fuseSiblingNodes(); // 调用fuseSiblingNodes方法，执行兄弟节点融合
    eraseUnusedMemRefAllocations(); // 删除融合后不再使用的内存引用分配
  }

  /// Run only producer/consumer fusion on the `mdg`.
  void runProducerConsumerFusionOnly() {
    fuseProducerConsumerNodes(
        /*maxSrcUserCount=*/std::numeric_limits<unsigned>::max()); // 执行生产者/消费者节点融合
    eraseUnusedMemRefAllocations();
  }

  // Run the GreedyFusion pass.
  // *) First pass through the nodes fuses single-use producer nodes into their
  //    unique consumer.
  // *) Second pass fuses sibling nodes which share no dependence edges.
  // *) Third pass fuses any remaining producer nodes into their users.
  void runGreedyFusion() {
    // TODO: Run this repeatedly until a fixed-point is reached.
    fuseProducerConsumerNodes(/*maxSrcUserCount=*/1); // 第一阶段：调用fuseProducerConsumerNodes方法，但限制只融合单一用户的生产者节点
    fuseSiblingNodes(); // 第二阶段：调用fuseSiblingNodes方法，融合没有依赖边的兄弟节点
    fuseProducerConsumerNodes(
        /*maxSrcUserCount=*/std::numeric_limits<unsigned>::max()); // 第三阶段：再次调用fuseProducerConsumerNodes方法，但这次不限制源节点的用户数量
    eraseUnusedMemRefAllocations();
  }

  /// Returns true if a private memref can be created for `memref` given
  /// the fusion scenario reflected by the other arguments.
  bool canCreatePrivateMemRef(Value memref,
                              const DenseSet<Value> &srcEscapingMemRefs,
                              unsigned producerId, unsigned consumerId,
                              bool removeSrcNode) { // 用于判断在给定融合场景下，是否可以为特定内存引用创建私有版本
    // 参数包括：待评估的内存引用、源节点逃逸内存引用的集合、生产者节点ID、消费者节点ID、以及是否会移除源节点
    // We can't generate private memrefs if their size can't be computed.
    if (!getMemRefIntOrFloatEltSizeInBytes(cast<MemRefType>(memref.getType()))) // 检查是否能够计算内存引用的元素大小（字节）
      return false;
    const Node *consumerNode = mdg->getNode(consumerId); // 获取消费者节点的指针，用于后续分析
    // If `memref` is an escaping one, do not create a private memref
    // for the below scenarios, since doing so will leave the escaping
    // memref unmodified as all the writes originally meant for the
    // escaping memref would be performed on the private memref:
    // 1. The source is to be removed after fusion,
    // OR
    // 2. The destination writes to `memref`.
    if (srcEscapingMemRefs.count(memref) > 0 && // 检查内存引用是否是逃逸的
        (removeSrcNode || consumerNode->getStoreOpCount(memref) > 0)) // 如果是，进一步检查: 源节点是否会被移除，或者消费者节点是否向该内存引用写入数据
      return false;

    // Don't create a private memref if 'srcNode' has in edges on
    // 'memref' or 'dstNode' has out edges on 'memref'.
    if (mdg->getIncomingMemRefAccesses(producerId, memref) > 0 || // 检查源节点是否有针对该内存引用的输入边（表示有其他节点向它提供数据）
        mdg->getOutEdgeCount(consumerId, memref) > 0) // 检查目标节点是否有针对该内存引用的输出边（表示它向其他节点提供数据）
      return false;

    // If 'srcNode' will be removed but it has out edges on 'memref' to
    // nodes other than 'dstNode', we have to preserve dependences and
    // cannot create a private memref.
    if (removeSrcNode &&
        any_of(mdg->outEdges[producerId], [&](const auto &edge) { // 如果源节点将被移除，检查它是否有基于该内存引用的输出边指向除了消费者节点以外的其他节点
          return edge.value == memref && edge.id != consumerId;
        }))
      return false; // 如果存在这样的边，返回false，因为需要保留这些依赖关系

    return true;
  }

  /// Perform fusions with node `dstId` as the destination of fusion, with
  /// No fusion is performed when producers with a user count greater than
  /// `maxSrcUserCount` for any of the memrefs involved.
  void performFusionsIntoDest(unsigned dstId, unsigned maxSrcUserCount) { // 它实现了将源循环融合到目标循环的核心逻辑
    LLVM_DEBUG(llvm::dbgs() << "Evaluating dst loop " << dstId << "\n");
    // Skip if this node was removed (fused into another node).
    if (mdg->nodes.count(dstId) == 0) // 如果目标节点已被移除（可能已被融合到其他节点中），则跳过
      return;
    // Get 'dstNode' into which to attempt fusion.
    auto *dstNode = mdg->getNode(dstId);
    // Skip if 'dstNode' is not a loop nest.
    if (!isa<AffineForOp>(dstNode->op)) // 如果目标节点不是循环嵌套（不是AffineForOp），则跳过
      return;
    // Skip if 'dstNode' is a loop nest returning values.
    // TODO: support loop nests that return values.
    if (dstNode->op->getNumResults() > 0) // 如果目标循环有返回值，则跳过
      return;

    LLVM_DEBUG(llvm::dbgs() << "Evaluating dst loop " << dstId << "\n");

    // Sink sequential loops in 'dstNode' (and thus raise parallel loops)
    // while preserving relative order. This can increase the maximum loop
    // depth at which we can fuse a slice of a producer loop nest into a
    // consumer loop nest.
    sinkSequentialLoops(dstNode); // 调用sinkSequentialLoops优化目标循环嵌套的结构（下沉顺序循环，提升并行循环）
    auto dstAffineForOp = cast<AffineForOp>(dstNode->op); // 获取目标循环的AffineForOp

    // Try to fuse 'dstNode' with candidate producer loops until a fixed point
    // is reached. Fusing two loops may expose new fusion opportunities.
    bool dstNodeChanged;
    do {
      // Gather src loop candidates for 'dstNode' and visit them in "quasi"
      // reverse program order to minimize the number of iterations needed to
      // reach the fixed point. Note that this is a best effort approach since
      // 'getProducerCandidates' does not always guarantee that program order
      // in 'srcIdCandidates'.
      dstNodeChanged = false; // 标记表示目标节点是否发生了变化
      SmallVector<unsigned, 16> srcIdCandidates; // 创建一个向量用于存储源候选节点ID
      getProducerCandidates(dstId, *mdg, srcIdCandidates); // 调用getProducerCandidates获取目标节点的生产者候选节点

      for (unsigned srcId : llvm::reverse(srcIdCandidates)) { // 按反向顺序遍历候选源节点
        // Get 'srcNode' from which to attempt fusion into 'dstNode'.
        auto *srcNode = mdg->getNode(srcId);
        auto srcAffineForOp = cast<AffineForOp>(srcNode->op); // 获取源节点和其对应的循环操作

        LLVM_DEBUG(llvm::dbgs()
                   << "Trying to fuse producer loop nest " << srcId
                   << " with consumer loop nest " << dstId << "\n");
        LLVM_DEBUG(llvm::dbgs() << "Compute tolerance threshold: "
                                << computeToleranceThreshold << '\n');
        LLVM_DEBUG(llvm::dbgs()
                   << "Producer loop nest:\n"
                   << *srcNode->op << "\n and consumer loop nest:\n"
                   << *dstNode->op << '\n');

        LLVM_DEBUG(llvm::dbgs() << "Evaluating src loop " << srcId
                                << " for dst loop " << dstId << "\n");

        // Skip if 'srcNode' is a loop nest returning values.
        // TODO: support loop nests that return values.
        if (isa<AffineForOp>(srcNode->op) && srcNode->op->getNumResults() > 0) // 如果源节点是带返回值的循环嵌套，则跳过
          continue;

        DenseSet<Value> producerConsumerMemrefs; // 创建一个集合用于存储生产者-消费者关系中涉及的内存引用
        gatherProducerConsumerMemrefs(srcId, dstId, *mdg,
                                      producerConsumerMemrefs); // 调用gatherProducerConsumerMemrefs收集这些内存引用

        // Skip if 'srcNode' out edge count on any memref is greater than
        // 'maxSrcUserCount'.
        if (any_of(producerConsumerMemrefs, [&](Value memref) { // 检查所有生产者-消费者内存引用，如果任何一个的输出边（用户）数量超过maxSrcUserCount，则跳过(为了限制融合的源循环的复杂度)
              return mdg->getOutEdgeCount(srcNode->id, memref) >
                     maxSrcUserCount;
            }))
          continue;

        // Gather memrefs in 'srcNode' that are written and escape out of the
        // block (e.g., memref block arguments, returned memrefs,
        // memrefs passed to function calls, etc.).
        DenseSet<Value> srcEscapingMemRefs; // 创建一个集合用于存储源节点中逃逸的内存引用
        gatherEscapingMemrefs(srcNode->id, *mdg, srcEscapingMemRefs); // 调用gatherEscapingMemrefs收集这些逃逸内存引用

        // Compute an operation list insertion point for the fused loop
        // nest which preserves dependences.
        Operation *fusedLoopInsPoint =
            mdg->getFusedLoopNestInsertionPoint(srcNode->id, dstNode->id); // 计算融合后循环嵌套的插入点，这个点需要保留所有依赖关系
        if (fusedLoopInsPoint == nullptr) // 如果无法找到合适的插入点，则跳过当前融合尝试
          continue;

        // It's possible this fusion is at an inner depth (i.e., there are
        // common surrounding affine loops for the source and destination for
        // ops). We need to get this number because the call to canFuseLoops
        // needs to be passed the absolute depth. The max legal depth and the
        // depths we try below are however *relative* and as such don't include
        // the common depth.
        SmallVector<AffineForOp, 4> surroundingLoops;
        getAffineForIVs(*dstAffineForOp, &surroundingLoops); // 收集目标循环外部的所有循环（共同的外层循环）
        unsigned numSurroundingLoops = surroundingLoops.size(); // 这些信息用于计算绝对融合深度

        // Compute the innermost common loop depth for dstNode
        // producer-consumer loads/stores.
        SmallVector<Operation *, 2> dstMemrefOps;
        for (Operation *op : dstNode->loads) // 收集目标节点中与生产者-消费者内存引用相关的所有加载和存储操作
          if (producerConsumerMemrefs.count(
                  cast<AffineReadOpInterface>(op).getMemRef()) > 0)
            dstMemrefOps.push_back(op);
        for (Operation *op : dstNode->stores)
          if (producerConsumerMemrefs.count(
                  cast<AffineWriteOpInterface>(op).getMemRef()))
            dstMemrefOps.push_back(op);
        unsigned dstLoopDepthTest =
            getInnermostCommonLoopDepth(dstMemrefOps) - numSurroundingLoops; // 计算这些操作的最内层共同循环深度，减去外层循环数量得到相对深度

        // Check the feasibility of fusing src loop nest into dst loop nest
        // at loop depths in range [1, dstLoopDepthTest].
        unsigned maxLegalFusionDepth = 0;
        SmallVector<ComputationSliceState, 8> depthSliceUnions; // 创建计算切片状态数组，每个深度一个
        depthSliceUnions.resize(dstLoopDepthTest);  
        FusionStrategy strategy(FusionStrategy::ProducerConsumer); // 设置融合策略为生产者-消费者
        for (unsigned i = 1; i <= dstLoopDepthTest; ++i) { // 对每个可能的融合深度（从1到最大测试深度），检查是否可以合法融合
          FusionResult result =
              affine::canFuseLoops(srcAffineForOp, dstAffineForOp,
                                   /*dstLoopDepth=*/i + numSurroundingLoops,
                                   &depthSliceUnions[i - 1], strategy); // 这里会计算每个深度的计算切片状态
          if (result.value == FusionResult::Success) { // 如果可以，更新最大合法融合深度
            maxLegalFusionDepth = i;
            LLVM_DEBUG(llvm::dbgs()
                       << "Found valid slice for depth: " << i << '\n');
          }
        }

        if (maxLegalFusionDepth == 0) { // 如果没有找到任何合法的融合深度，跳过当前融合尝试
          LLVM_DEBUG(llvm::dbgs()
                     << "Can't fuse: fusion is not legal at any depth\n");
          continue;
        }

        LLVM_DEBUG(llvm::dbgs() << "Max legal depth for fusion: "
                                << maxLegalFusionDepth << '\n');

        double computeToleranceThresholdToUse = computeToleranceThreshold; // 初始化计算容忍阈值

        // Cyclic dependences in the source nest may be violated when performing
        // slicing-based fusion. They aren't actually violated in cases where no
        // redundant execution of the source happens (1:1 pointwise dep on the
        // producer-consumer memref access for example). Check this and allow
        // fusion accordingly.
        if (hasCyclicDependence(srcAffineForOp)) { // 检查源循环是否有循环依赖
          LLVM_DEBUG(llvm::dbgs() << "Source nest has a cyclic dependence.\n");
          // Maximal fusion does not check for compute tolerance threshold; so
          // perform the maximal fusion only when the redundanation computation
          // is zero.
          if (maximalFusion) { // 如果有循环依赖，需要特殊处理计算容忍阈值
            auto srcForOp = cast<AffineForOp>(srcNode->op);
            auto dstForOp = cast<AffineForOp>(dstNode->op);
            int64_t sliceCost;
            int64_t fusedLoopNestComputeCost;
            auto fraction = getAdditionalComputeFraction(
                srcForOp, dstForOp, maxLegalFusionDepth, depthSliceUnions,
                sliceCost, fusedLoopNestComputeCost);
            if (!fraction || fraction > 0) {
              LLVM_DEBUG(
                  llvm::dbgs()
                  << "Can't perform maximal fusion with a cyclic dependence "
                     "and non-zero additional compute.\n");
              return;
            }
          } else {
            // Set redundant computation tolerance to zero regardless of what
            // the user specified. Without this, fusion would be invalid.
            LLVM_DEBUG(llvm::dbgs()
                       << "Setting compute tolerance to zero since "
                          "source has a cylic dependence.\n");
            computeToleranceThresholdToUse = 0;
          }
        }

        // Check if fusion would be profitable. We skip profitability analysis
        // for maximal fusion since we already know the maximal legal depth to
        // fuse.
        unsigned bestDstLoopDepth = maxLegalFusionDepth;
        if (!maximalFusion) { // 如果不是最大化融合，评估融合的盈利性
          // Retrieve producer stores from the src loop.
          SmallVector<Operation *, 2> producerStores;
          for (Operation *op : srcNode->stores) // 收集生产者存储操作，用于盈利性分析
            if (producerConsumerMemrefs.count(
                    cast<AffineWriteOpInterface>(op).getMemRef()))
              producerStores.push_back(op);

          assert(!producerStores.empty() && "Expected producer store");
          if (!isFusionProfitable(srcAffineForOp, producerStores,
                                  dstAffineForOp, depthSliceUnions,
                                  maxLegalFusionDepth, &bestDstLoopDepth,
                                  computeToleranceThresholdToUse)) { // 调用isFusionProfitable判断融合是否有利，并确定最佳融合深度
            continue;
          }
        }

        assert(bestDstLoopDepth > 0 && "Unexpected loop fusion depth");
        ComputationSliceState &bestSlice =
            depthSliceUnions[bestDstLoopDepth - 1]; // 获取最佳融合深度对应的计算切片状态
        assert(!bestSlice.isEmpty() && "Missing slice union for depth");

        // Determine if 'srcId' can be removed after fusion, taking into
        // account remaining dependences, escaping memrefs and the fusion
        // insertion point.
        bool removeSrcNode = canRemoveSrcNodeAfterFusion(
            srcId, dstId, bestSlice, fusedLoopInsPoint, srcEscapingMemRefs,
            *mdg); // 判断融合后是否可以安全移除源节点

        DenseSet<Value> privateMemrefs;
        for (Value memref : producerConsumerMemrefs) { // 检查每个生产者-消费者内存引用，判断是否可以创建私有版本
          if (canCreatePrivateMemRef(memref, srcEscapingMemRefs, srcId, dstId,
                                     removeSrcNode)) {
            // Create a private version of this memref.
            LLVM_DEBUG(llvm::dbgs()
                       << "Creating private memref for " << memref << '\n');
            // Create a private version of this memref.
            privateMemrefs.insert(memref); // 如果可以，将其添加到私有内存引用集合
          }
        }

        // Fuse computation slice of 'srcLoopNest' into 'dstLoopNest'.
        fuseLoops(srcAffineForOp, dstAffineForOp, bestSlice); // 执行实际的循环融合，将源循环的计算切片融合到目标循环中
        dstNodeChanged = true; // 标记目标节点已变化

        LLVM_DEBUG(llvm::dbgs()
                   << "Fused src loop " << srcId << " into dst loop " << dstId
                   << " at depth " << bestDstLoopDepth << ":\n"
                   << dstAffineForOp << "\n");

        // Move 'dstAffineForOp' before 'insertPointInst' if needed.
        if (fusedLoopInsPoint != dstAffineForOp) // 如果需要，将融合后的循环移到正确的插入点
          dstAffineForOp->moveBefore(fusedLoopInsPoint);

        // Update edges between 'srcNode' and 'dstNode'.
        mdg->updateEdges(srcNode->id, dstNode->id, privateMemrefs,
                         removeSrcNode); // 更新依赖图中源节点和目标节点之间的边

        // Create private memrefs.
        if (!privateMemrefs.empty()) { // 如果有需要创建的私有内存引用，执行创建和替换过程,对每个私有内存引用，创建新的内存引用并更新依赖图
          // Note the block into which fusion was performed. This can be used to
          // place `alloc`s that create private memrefs.
          Block *sliceInsertionBlock = bestSlice.insertPoint->getBlock();

          // Gather stores for all the private-to-be memrefs.
          DenseMap<Value, SmallVector<Operation *, 4>> privateMemRefToStores;
          dstAffineForOp.walk([&](AffineWriteOpInterface storeOp) {
            Value storeMemRef = storeOp.getMemRef();
            if (privateMemrefs.count(storeMemRef) > 0)
              privateMemRefToStores[storeMemRef].push_back(storeOp);
          });

          // Replace original memrefs with private memrefs. Note that all the
          // loads and stores on these memrefs will be replaced with a new
          // loads and stores. Any reference to the original ones becomes
          // invalid after this point.
          for (auto &memrefToStoresPair : privateMemRefToStores) {
            // TODO: Use union of memref write regions to compute
            // private memref footprint.
            SmallVector<Operation *, 4> &storesForMemref =
                memrefToStoresPair.second;
            Value newMemRef = createPrivateMemRef(
                dstAffineForOp, storesForMemref, bestDstLoopDepth,
                fastMemorySpace, sliceInsertionBlock, localBufSizeThreshold);
            // Create new node in dependence graph for 'newMemRef' alloc op.
            unsigned newMemRefNodeId = mdg->addNode(newMemRef.getDefiningOp());
            // Add edge from 'newMemRef' node to dstNode.
            mdg->addEdge(newMemRefNodeId, dstId, newMemRef);
          }
          // One or more entries for 'newMemRef' alloc op are inserted into
          // the DenseMap mdg->nodes. Since an insertion may cause DenseMap to
          // reallocate, update dstNode.
          dstNode = mdg->getNode(dstId);
        }

        // Collect dst loop stats after memref privatization transformation.
        LoopNestStateCollectorPlus dstLoopCollector;
        dstLoopCollector.collect(dstAffineForOp); // 收集融合和内存引用私有化后的目标循环统计信息

        // Clear and add back loads and stores.
        mdg->clearNodeLoadAndStores(dstNode->id); // 清除目标节点原有的加载和存储信息
        mdg->addToNode(
            dstId, dstLoopCollector.loadOpInsts, dstLoopCollector.storeOpInsts,
            dstLoopCollector.memrefLoads, dstLoopCollector.memrefStores,
            dstLoopCollector.memrefFrees); // 添加新的加载和存储信息

        if (removeSrcNode) { // 如果可以移除源节点，删除源循环操作并从依赖图中移除节点
          LLVM_DEBUG(llvm::dbgs()
                     << "Removing src loop " << srcId << " after fusion\n");
          // srcNode is no longer valid after it is removed from mdg.
          srcAffineForOp.erase();
          mdg->removeNode(srcId);
          srcNode = nullptr;
        }
      }
    } while (dstNodeChanged); // 如果目标节点在当前迭代中发生了变化，继续尝试更多融合(因为经过依此融合操作之后可能会创建新的融合机会)
  }

  /// Visit each node in the graph, and for each node, attempt to fuse it with
  /// producer-consumer candidates. No fusion is performed when producers with a
  /// user count greater than `maxSrcUserCount` for any of the memrefs involved
  /// are encountered.
  void fuseProducerConsumerNodes(unsigned maxSrcUserCount) { // 该方法访问图中的每个节点，并尝试将其与生产者-消费者候选节点融合,当任何涉及的内存引用的生产者的用户数量超过maxSrcUserCount时，不会执行融合
    LLVM_DEBUG(llvm::dbgs() << "--- Producer/Consumer Fusion ---\n");
    init();
    while (!worklist.empty()) {
      unsigned dstId = worklist.back();
      worklist.pop_back();
      performFusionsIntoDest(dstId, maxSrcUserCount); // 调用performFusionsIntoDest方法，尝试将生产者节点融合到目标节点中
    }
  }

  // Visits each node in the graph, and for each node, attempts to fuse it with
  // its sibling nodes (nodes which share a parent, but no dependence edges).
  void fuseSiblingNodes() { // 该方法访问图中的每个节点，并尝试将其与兄弟节点融合
    LLVM_DEBUG(llvm::dbgs() << "--- Sibling Fusion ---\n");
    init();
    while (!worklist.empty()) {
      unsigned dstId = worklist.back();
      worklist.pop_back();

      // Skip if this node was removed (fused into another node).
      if (mdg->nodes.count(dstId) == 0)
        continue;
      // Get 'dstNode' into which to attempt fusion.
      auto *dstNode = mdg->getNode(dstId);
      // Skip if 'dstNode' is not a loop nest.
      if (!isa<AffineForOp>(dstNode->op))
        continue;
      // Attempt to fuse 'dstNode' with its sibling nodes in the graph.
      fuseWithSiblingNodes(dstNode); // 调用fuseWithSiblingNodes方法，尝试将目标节点与其兄弟节点融合
    }
  }

  // Attempt to fuse 'dstNode' with sibling nodes in the graph.
  void fuseWithSiblingNodes(Node *dstNode) { // 接收目标节点指针作为参数，尝试将兄弟节点融合到这个目标节点中
    DenseSet<unsigned> visitedSibNodeIds; // 创建一个集合用于跟踪已访问的兄弟节点ID，避免重复处理
    std::pair<unsigned, Value> idAndMemref;
    auto dstAffineForOp = cast<AffineForOp>(dstNode->op);

    while (findSiblingNodeToFuse(dstNode, &visitedSibNodeIds, &idAndMemref)) { // 循环寻找可以与目标节点融合的兄弟节点
      unsigned sibId = idAndMemref.first; // 从返回结果中获取兄弟节点ID和共享的内存引用
      Value memref = idAndMemref.second;
      // TODO: Check that 'sibStoreOpInst' post-dominates all other
      // stores to the same memref in 'sibNode' loop nest.
      auto *sibNode = mdg->getNode(sibId);
      // Compute an operation list insertion point for the fused loop
      // nest which preserves dependences.
      assert(sibNode->op->getBlock() == dstNode->op->getBlock());
      Operation *insertPointInst = // 确定融合后循环的插入点，这个点需要保留所有依赖关系
          sibNode->op->isBeforeInBlock(dstNode->op)
              ? mdg->getFusedLoopNestInsertionPoint(sibNode->id, dstNode->id)
              : mdg->getFusedLoopNestInsertionPoint(dstNode->id, sibNode->id); // 根据兄弟节点和目标节点在块中的相对位置，选择正确的顺序调用getFusedLoopNestInsertionPoint
      if (insertPointInst == nullptr) // 如果无法找到合适的插入点，跳过当前兄弟节点
        continue;

      // Check if fusion would be profitable and at what depth.

      // Get unique 'sibNode' load op to 'memref'.
      SmallVector<Operation *, 2> sibLoadOpInsts;
      sibNode->getLoadOpsForMemref(memref, &sibLoadOpInsts); // 收集兄弟节点中访问共享内存引用的加载操作
      // Currently findSiblingNodeToFuse searches for siblings with one load.
      assert(sibLoadOpInsts.size() == 1); // 当前实现假设兄弟节点只有一个加载操作，所以断言确保这一点
      Operation *sibLoadOpInst = sibLoadOpInsts[0];

      // Gather 'dstNode' load ops to 'memref'.
      SmallVector<Operation *, 2> dstLoadOpInsts;
      dstNode->getLoadOpsForMemref(memref, &dstLoadOpInsts); // 收集目标节点中访问同一共享内存引用的加载操作

      // It's possible this fusion is at an inner depth (i.e., there are common
      // surrounding affine loops for the source and destination for ops). We
      // need to get this number because the call to canFuseLoops needs to be
      // passed the absolute depth. The max legal depth and the depths we try
      // below are however *relative* and as such don't include the common
      // depth.
      SmallVector<AffineForOp, 4> surroundingLoops;
      getAffineForIVs(*dstAffineForOp, &surroundingLoops); // 收集目标循环外部的所有循环（共同的外层循环）
      unsigned numSurroundingLoops = surroundingLoops.size(); // 计算外层循环的数量
      SmallVector<AffineForOp, 4> dstLoopIVs;
      getAffineForIVs(*dstLoadOpInsts[0], &dstLoopIVs); // 收集目标节点中加载操作所处的循环层次
      unsigned dstLoopDepthTest = dstLoopIVs.size() - numSurroundingLoops; // 计算可测试的相对循环深度
      auto sibAffineForOp = cast<AffineForOp>(sibNode->op); // 获取兄弟节点的循环操作

      // Compute loop depth and slice union for fusion.
      SmallVector<ComputationSliceState, 8> depthSliceUnions; // 准备计算切片状态数组，每个深度一个
      depthSliceUnions.resize(dstLoopDepthTest);
      unsigned maxLegalFusionDepth = 0;
      FusionStrategy strategy(memref); // 创建一个基于共享内存引用的融合策略
      for (unsigned i = 1; i <= dstLoopDepthTest; ++i) { // 对每个可能的深度，检查融合的可行性
        FusionResult result =
            affine::canFuseLoops(sibAffineForOp, dstAffineForOp,
                                 /*dstLoopDepth=*/i + numSurroundingLoops,
                                 &depthSliceUnions[i - 1], strategy); // 使用canFuseLoops函数判断在特定深度融合是否合法

        if (result.value == FusionResult::Success) // 如果可行，更新最大合法融合深度
          maxLegalFusionDepth = i;
      }

      LLVM_DEBUG(llvm::dbgs() << "Max legal depth for fusion: "
                              << maxLegalFusionDepth << '\n');

      // Skip if fusion is not feasible at any loop depths.
      if (maxLegalFusionDepth == 0) // 如果没有找到合法的融合深度，跳过当前兄弟节点
        continue;

      double computeToleranceThresholdToUse = computeToleranceThreshold; // 初始化计算容忍阈值

      // Cyclic dependences in the source nest may be violated when performing
      // slicing-based fusion. They aren't actually violated in cases where no
      // redundant execution of the source happens (1:1 pointwise dep on the
      // producer-consumer memref access for example). Check this and allow
      // fusion accordingly.
      if (hasCyclicDependence(sibAffineForOp)) { // 检查源循环是否存在循环依赖,如有循环依赖，会根据是否为最大化融合模式采取不同的处理策略
        LLVM_DEBUG(llvm::dbgs() << "Source nest has a cyclic dependence.\n");
        // Maximal fusion does not check for compute tolerance threshold; so
        // perform the maximal fusion only when the redundanation computation is
        // zero.
        if (maximalFusion) {
          auto dstForOp = cast<AffineForOp>(dstNode->op);
          int64_t sliceCost;
          int64_t fusedLoopNestComputeCost;
          auto fraction = getAdditionalComputeFraction(
              sibAffineForOp, dstForOp, maxLegalFusionDepth, depthSliceUnions,
              sliceCost, fusedLoopNestComputeCost);
          if (!fraction || fraction > 0) {
            LLVM_DEBUG(
                llvm::dbgs()
                << "Can't perform maximal fusion with a cyclic dependence "
                   "and non-zero additional compute.\n");
            return;
          }
        } else {
          // Set redundant computation tolerance to zero regardless of what the
          // user specified. Without this, fusion would be invalid.
          LLVM_DEBUG(llvm::dbgs() << "Setting compute tolerance to zero since "
                                     "source has a cyclic dependence.\n");
          computeToleranceThresholdToUse = 0.0;
        }
      }

      unsigned bestDstLoopDepth = maxLegalFusionDepth;
      if (!maximalFusion) { // 如果不是最大化融合，评估融合的盈利性
        // Check if fusion would be profitable. For sibling fusion, the sibling
        // load op is treated as the src "store" op for fusion profitability
        // purposes. The footprint of the load in the slice relative to the
        // unfused source's determines reuse.
        if (!isFusionProfitable(sibAffineForOp, sibLoadOpInst, dstAffineForOp,
                                depthSliceUnions, maxLegalFusionDepth,
                                &bestDstLoopDepth,
                                computeToleranceThresholdToUse))
          continue;
      }

      assert(bestDstLoopDepth > 0 && "Unexpected loop fusion depth");

      const ComputationSliceState &bestSlice =
          depthSliceUnions[bestDstLoopDepth - 1]; // 获取最佳融合深度对应的计算切片状态
      assert(!bestSlice.isEmpty() &&
             "Fusion depth has no computed slice union"); // 确保切片状态非空

      // Do not perform sibling fusion if it isn't maximal. We always remove the
      // sibling node and as such fusion shouldn't be performed if a part of the
      // slice is used in the destination.
      auto isMaximal = bestSlice.isMaximal();
      if (!isMaximal.value_or(false)) { // 检查切片是否是最大的（包含整个兄弟循环）,如果不是最大的，跳过当前兄弟节点(这是因为兄弟融合总是移除源节点，如果只融合部分切片会导致计算丢失)
        LLVM_DEBUG(llvm::dbgs()
                   << "Slice isn't maximal; not performing sibling fusion.\n");
        continue;
      }

      // Check if source loop is being inserted in the innermost
      // destination loop. Based on this, the fused loop may be optimized
      // further inside `fuseLoops`.
      bool isInnermostInsertion = (bestDstLoopDepth == dstLoopDepthTest); // 判断是否在最内层插入源循环
      // Fuse computation slice of 'sibLoopNest' into 'dstLoopNest'.
      affine::fuseLoops(sibAffineForOp, dstAffineForOp, bestSlice,
                        isInnermostInsertion); // 调用fuseLoops函数执行实际的循环融合

      auto dstForInst = cast<AffineForOp>(dstNode->op);
      // Update operation position of fused loop nest (if needed).
      if (insertPointInst != dstForInst)
        dstForInst->moveBefore(insertPointInst); // 如果需要，将融合后的循环移到正确的插入点

      LLVM_DEBUG(llvm::dbgs()
                 << "Fused sibling nest " << sibId << " into destination nest "
                 << dstNode->id << " at depth " << bestDstLoopDepth << ":\n"
                 << dstAffineForOp << "\n");

      // Update data dependence graph state post fusion.
      updateStateAfterSiblingFusion(sibNode, dstNode); // 更新依赖图状态，反映兄弟融合后的变化

      // Remove old sibling loop nest.
      // Get op before we invalidate the MDG node.
      Operation *op = sibNode->op;
      mdg->removeNode(sibNode->id); // 从依赖图中移除兄弟节点
      op->erase(); // 删除兄弟节点的操作
    }
  }

  // Searches block argument uses and the graph from 'dstNode' looking for a
  // fusion candidate sibling node which shares no dependences with 'dstNode'
  // but which loads from the same memref. Returns true and sets
  // 'idAndMemrefToFuse' on success. Returns false otherwise.
  bool findSiblingNodeToFuse(Node *dstNode,
                             DenseSet<unsigned> *visitedSibNodeIds,
                             std::pair<unsigned, Value> *idAndMemrefToFuse) { // 它用于寻找可以与目标节点进行兄弟融合的候选节点
    // 参数包括目标节点指针、已访问兄弟节点ID集合和用于存储结果的对
    // Returns true if 'sibNode' can be fused with 'dstNode' for input reuse
    // on 'memref'.
    auto canFuseWithSibNode = [&](Node *sibNode, Value memref) { // 定义一个lambda函数，用于判断给定的兄弟节点是否可以与目标节点在特定内存引用上进行输入重用融合
      // Skip if 'outEdge' is not a read-after-write dependence.
      // TODO: Remove restrict to single load op restriction.
      if (sibNode->getLoadOpCount(memref) != 1) // 检查兄弟节点是否只有一个对给定内存引用的加载操作
        return false;
      // Skip if there exists a path of dependent edges between
      // 'sibNode' and 'dstNode'.
      if (mdg->hasDependencePath(sibNode->id, dstNode->id) ||
          mdg->hasDependencePath(dstNode->id, sibNode->id)) // 检查兄弟节点和目标节点之间是否有依赖路径
        return false;
      // Skip sib node if it loads to (and stores from) the same memref on
      // which it also has an input dependence edge.
      DenseSet<Value> loadAndStoreMemrefSet;
      sibNode->getLoadAndStoreMemrefSet(&loadAndStoreMemrefSet); // 获取兄弟节点同时有加载和存储操作的内存引用集合
      if (llvm::any_of(loadAndStoreMemrefSet, [=](Value memref) { // 如果兄弟节点对这些内存引用有输入依赖边，则跳过该节点
            return mdg->getIncomingMemRefAccesses(sibNode->id, memref) > 0;
          }))
        return false; // 这避免了融合可能违反的依赖关系

      // Check that all stores are to the same memref if any.
      DenseSet<Value> storeMemrefs;
      for (auto *storeOpInst : sibNode->stores) { // 检查兄弟节点的所有存储操作是否都写入同一个内存引用
        storeMemrefs.insert(
            cast<AffineWriteOpInterface>(storeOpInst).getMemRef());
      }
      return storeMemrefs.size() <= 1; // 如果兄弟节点写入多个不同的内存引用，则不适合融合
    };

    // Search for siblings which load the same memref block argument.
    Block *block = dstNode->op->getBlock(); // 获取目标节点所在的基本块
    for (unsigned i = 0, e = block->getNumArguments(); i != e; ++i) { // 开始遍历该基本块的所有参数，寻找加载相同参数的兄弟节点
      for (Operation *user : block->getArgument(i).getUsers()) { // 遍历当前块参数的所有使用者
        auto loadOp = dyn_cast<AffineReadOpInterface>(user); // 尝试将使用者转换为仿射加载操作，如果不是则跳过
        if (!loadOp)
          continue;
        // Gather loops surrounding 'use'.
        SmallVector<AffineForOp, 4> loops;
        getAffineForIVs(*user, &loops); // 收集包围该加载操作的所有循环
        // Skip 'use' if it is not within a loop nest.
        // Find the surrounding affine.for nested immediately within the
        // block.
        auto *it = llvm::find_if(loops, [&](AffineForOp loop) { // 寻找直接包含在目标基本块中的循环
          return loop->getBlock() == &mdg->block;
        });
        // Skip 'use' if it is not within a loop nest in `block`.
        if (it == loops.end()) // 如果加载操作不在目标基本块的循环中，则跳过
          continue;
        Node *sibNode = mdg->getForOpNode(*it); // 获取该循环对应的依赖图节点
        assert(sibNode != nullptr);
        // Skip 'use' if it not a sibling to 'dstNode'.
        if (sibNode->id == dstNode->id) // 如果节点就是目标节点自身，跳过
          continue;
        // Skip 'use' if it has been visited.
        if (visitedSibNodeIds->count(sibNode->id) > 0) // 如果节点已经被访问过，跳过
          continue;
        // Skip 'use' if it does not load from the same memref as 'dstNode'.
        auto memref = loadOp.getMemRef(); // 获取加载操作使用的内存引用
        if (dstNode->getLoadOpCount(memref) == 0) // 如果目标节点不从该内存引用加载数据，跳过
          continue;
        // Check if 'sibNode/dstNode' can be input-reuse fused on 'memref'.
        if (canFuseWithSibNode(sibNode, memref)) { // 调用前面定义的lambda函数检查是否可以融合
          visitedSibNodeIds->insert(sibNode->id);
          idAndMemrefToFuse->first = sibNode->id;
          idAndMemrefToFuse->second = memref;
          return true; // 如果可以融合，将兄弟节点ID和内存引用存储在结果中，并返回true
        }
      }
    }

    // Search for siblings by following edges through an intermediate src node.
    // Collect candidate 'dstNode' input edges in 'inEdges'.
    SmallVector<MemRefDependenceGraph::Edge, 2> inEdges;
    mdg->forEachMemRefInputEdge( // 如果通过块参数没有找到兄弟节点，尝试通过依赖图边寻找
        dstNode->id, [&](MemRefDependenceGraph::Edge inEdge) {
          // Add 'inEdge' if it is a read-after-write dependence.
          if (dstNode->getLoadOpCount(inEdge.value) > 0 &&
              mdg->getNode(inEdge.id)->getStoreOpCount(inEdge.value) > 0)
            inEdges.push_back(inEdge);
        });

    // Search for sibling nodes to fuse by visiting output edges from each input
    // edge in 'inEdges'.
    for (auto &inEdge : inEdges) { // 遍历所有输入边，寻找通过共同生产者连接的兄弟节点
      // Collect candidate output edges from each node 'inEdge.id' in 'inEdges'.
      SmallVector<MemRefDependenceGraph::Edge, 2> outEdges;
      mdg->forEachMemRefOutputEdge( // 对于每个输入边，收集其源节点的所有输出边
          inEdge.id, [&](MemRefDependenceGraph::Edge outEdge) {
            unsigned sibNodeId = outEdge.id;
            if (visitedSibNodeIds->count(sibNodeId) > 0) // 如果节点已被访问，跳过
              return;
            // Skip output edge if not a sibling using the same memref.
            if (outEdge.id == dstNode->id || outEdge.value != inEdge.value) // 如果输出边指向目标节点自身，或者使用的不是相同的内存引用，跳过
              return;
            auto *sibNode = mdg->getNode(sibNodeId);
            if (!isa<AffineForOp>(sibNode->op)) // 获取节点指针，如果节点不是循环嵌套，跳过
              return;
            // Check if 'sibNode/dstNode' can be input-reuse fused on 'memref'.
            if (canFuseWithSibNode(sibNode, outEdge.value)) { // 获取节点指针，如果节点不是循环嵌套，跳过
              // Add candidate 'outEdge' to sibling node.
              outEdges.push_back(outEdge);
            }
          });

      // Add first candidate if any were returned.
      if (!outEdges.empty()) { // 如果找到了候选边，选择第一个作为结果并返回true
        visitedSibNodeIds->insert(outEdges[0].id);
        idAndMemrefToFuse->first = outEdges[0].id;
        idAndMemrefToFuse->second = outEdges[0].value;
        return true;
      }
    }
    return false; // 如果所有搜索路径都未找到合适的兄弟节点，返回false
  }

  /// Update data dependence graph state to reflect sibling fusion of 'sibNode'
  /// into 'dstNode'.
  void updateStateAfterSiblingFusion(Node *sibNode, Node *dstNode) { // 它用于在兄弟融合后更新依赖图的状态
    // Update 'sibNode' and 'dstNode' input/output edges to reflect fusion.
    mdg->updateEdges(sibNode->id, dstNode->id); // 调用依赖图的updateEdges方法，更新节点间的依赖边

    // Collect dst loop stats after memref privatization transformation.
    auto dstForInst = cast<AffineForOp>(dstNode->op);
    LoopNestStateCollectorPlus dstLoopCollector;
    dstLoopCollector.collect(dstForInst);
    // Clear and add back loads and stores
    mdg->clearNodeLoadAndStores(dstNode->id);
    mdg->addToNode(dstNode->id, dstLoopCollector.loadOpInsts,
                   dstLoopCollector.storeOpInsts, dstLoopCollector.memrefLoads,
                   dstLoopCollector.memrefStores, dstLoopCollector.memrefFrees);
  }

  // Clean up any allocs with no users.
  void eraseUnusedMemRefAllocations() {
    for (auto &pair : mdg->memrefEdgeCount) {
      if (pair.second > 0)
        continue;
      auto memref = pair.first;
      // Skip if there exist other uses (return operation or function calls).
      if (!memref.use_empty())
        continue;
      // Use list expected to match the dep graph info.
      auto *op = memref.getDefiningOp();
      if (isa_and_nonnull<memref::AllocOp>(op))
        op->erase();
    }
  }
};

} // namespace

/// Run fusion on `block`.
void LoopFusion::runOnBlock(Block *block) {
  MemRefDependenceGraph g(*block);
  if (!g.init()) {
    LLVM_DEBUG(llvm::dbgs() << "MDG init failed\n");
    return;
  }

  std::optional<unsigned> fastMemorySpaceOpt;
  if (fastMemorySpace.hasValue())
    fastMemorySpaceOpt = fastMemorySpace;
  unsigned localBufSizeThresholdBytes = localBufSizeThreshold * 1024;
  GreedyFusion fusion(&g, localBufSizeThresholdBytes, fastMemorySpaceOpt,
                      maximalFusion, computeToleranceThreshold);

  if (affineFusionMode == FusionMode::ProducerConsumer)
    fusion.runProducerConsumerFusionOnly();
  else if (affineFusionMode == FusionMode::Sibling)
    fusion.runSiblingFusionOnly();
  else
    fusion.runGreedyFusion();
}

void LoopFusion::runOnOperation() {
  // Call fusion on every op that has at least two affine.for nests (in post
  // order).
  getOperation()->walk([&](Operation *op) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        auto affineFors = block.getOps<AffineForOp>();
        if (!affineFors.empty() && !llvm::hasSingleElement(affineFors))
          runOnBlock(&block);
      }
    }
  });
}

// std::unique_ptr<Pass> mlir::affine::createLoopFusionPass(
//     unsigned fastMemorySpace, uint64_t localBufSizeThreshold,
//     bool maximalFusion, enum FusionMode affineFusionMode) {
//   return std::make_unique<LoopFusion>(fastMemorySpace, localBufSizeThreshold,
//                                       maximalFusion, affineFusionMode);
// }

namespace onnx_mlir {
std::unique_ptr<Pass> createLoopFusionPass(
    unsigned fastMemorySpace, uint64_t localBufSizeThreshold,
    bool maximalFusion, enum FusionMode affineFusionMode) {
  return std::make_unique<LoopFusion>(fastMemorySpace, localBufSizeThreshold,
                                      maximalFusion, affineFusionMode);
}
} // namespace onnx_mlir
    
static mlir::PassRegistration<LoopFusion> pass;