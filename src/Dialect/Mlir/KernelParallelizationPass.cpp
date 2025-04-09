#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"
#include <memory>

// Include our component headers
// In a real build system, these would be proper includes
#include "DependencyGraph.h"
#include "TopoLogicalSort.h"
#include "IrReorganization.h"

using namespace mlir;
using namespace onnx_mlir;

#define DEBUG_TYPE "kernel-parallelization"

namespace {

struct KernelParallelizationPass
    : public PassWrapper<KernelParallelizationPass, OperationPass<ModuleOp>> {
  
  StringRef getArgument() const final { return "kernel-parallelization"; }
  StringRef getDescription() const final {
    return "Parallelize GPU kernels and loop nests based on dependencies";
  }
  
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    
    // Steps 1-2: Build dependency graph for each function
    llvm::SmallVector<std::unique_ptr<DependencyGraph>, 4> functionGraphs;
    
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
        LLVM_DEBUG(llvm::dbgs() << "Processing function: " << funcOp.getName() << "\n");
        
        // Build dependency graph
        auto graph = buildDependencyGraph(funcOp);
        // dumpDependencyGraph(*graph); // Print dependency graph
        LLVM_DEBUG(llvm::dbgs() << "Built dependency graph with " 
                << graph->nodes.size() << " nodes\n");
        
        // Perform topological sort
        performTopologicalSort(*graph);
        LLVM_DEBUG(dumpTopologicalLevels(*graph));
        
        // Reorganize IR based on topological levels
        reorganizeIR(funcOp, *graph);
        LLVM_DEBUG(llvm::dbgs() << "Reorganized IR for function: " << funcOp.getName() << "\n");
        
        // Save graph for module reorganization
        functionGraphs.push_back(std::move(graph));
    }
    
    // Step 3: Combine all function graphs for module reorganization
    auto combinedGraph = std::make_unique<DependencyGraph>();
    for (auto &graph : functionGraphs) {
        for (auto &nodePair : graph->nodes) {
        if (nodePair->type == NodeType::Kernel) {
            // Only need to keep kernel nodes for module reorganization
            combinedGraph->addNode(std::move(nodePair));
        }
        }
    }
    
    // Reorganize GPU modules
    reorganizeGPUModules(moduleOp, *combinedGraph);
    LLVM_DEBUG(llvm::dbgs() << "Reorganized GPU modules\n");
  }
};

} // end anonymous namespace

// Pass registration
// std::unique_ptr<Pass> createKernelParallelizationPass() {
//   return std::make_unique<KernelParallelizationPass>();
// }

// void registerKernelParallelizationPass() {
//   PassRegistration<KernelParallelizationPass>();
// }

namespace onnx_mlir {
  namespace krnl {
  
    std::unique_ptr<Pass> createKernelParallelizationPass() {
    return std::make_unique<KernelParallelizationPass>();
    }
  
  } // namespace krnl
  } // namespace onnx_mlir
  
static mlir::PassRegistration<KernelParallelizationPass> pass;