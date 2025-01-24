#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace {
class RemoveRedundantSCFIfPass 
    : public PassWrapper<RemoveRedundantSCFIfPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveRedundantSCFIfPass)

    bool isTrue(Value value, SmallPtrSet<Operation*, 4>& visited) {
        Operation *defOp = value.getDefiningOp();
        if (!defOp || !visited.insert(defOp).second)
            return false;

        // Check if it's a krnl.global operation
        if (defOp->getName().getStringRef() == "krnl.global") {
            if (auto valueAttr = defOp->getAttrOfType<DenseElementsAttr>("value")) {
                if (valueAttr.getType().cast<ShapedType>().getNumElements() == 1) {
                    return valueAttr.getSplatValue<bool>();
                }
            }
            return false;
        }

        // Check if it's an affine.load operation
        if (auto loadOp = dyn_cast<affine::AffineLoadOp>(defOp)) {
            Value memref = loadOp.getMemRef();
            
            // If it's allocated memory, find the most recent store
            if (memref.getDefiningOp() && 
                isa<memref::AllocOp>(memref.getDefiningOp())) {
                    
                // Only search and check values, don't modify any operations
                for (Operation* user : memref.getUsers()) {
                    // First check if we've encountered our load operation
                    if (user == defOp) {
                        continue;
                    }
                    
                    // Check if it's a store operation
                    if (auto storeOp = dyn_cast<affine::AffineStoreOp>(user)) {
                        return isTrue(storeOp.getValue(), visited);
                    }
                }

                // If no store operation found, check the source of memref
                return isTrue(memref, visited);
            }
            
            // If not allocated memory, continue tracing the source of memref
            return isTrue(memref, visited);
        }

        // Check if it's a constant operation
        if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
            if (auto boolAttr = constOp.getValue().dyn_cast<BoolAttr>()) {
                return boolAttr.getValue();
            }
        }

        return false;
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        SmallVector<scf::IfOp, 8> toRemove;
        
        // Traverse and check all if operations
        module.walk([&](scf::IfOp ifOp) {
            // Skip if operations with results or else branch
            if (!ifOp.getResults().empty() || !ifOp.getElseRegion().empty())
                return;

            SmallPtrSet<Operation*, 4> visited;
            Value condition = ifOp.getCondition();
            if (isTrue(condition, visited)) {
                toRemove.push_back(ifOp);
            }
        });

        // Process if operations that need to be removed
        OpBuilder builder(&getContext());
        for (auto ifOp : toRemove) {
            builder.setInsertionPoint(ifOp);
            
            Block &thenBlock = ifOp.getThenRegion().front();
            
            // Create a value mapping table to maintain the correspondence between old and new values
            IRMapping valueMapper;
            
            // Clone all operations in the then branch (except the last yield)
            for (auto &op : llvm::make_range(thenBlock.begin(), 
                                          std::prev(thenBlock.end()))) {
                // Clone operation using the value mapper
                Operation *newOp = builder.clone(op, valueMapper);
                
                // Map original operation results to new operation results
                for (auto it : llvm::zip(op.getResults(), newOp->getResults())) {
                    valueMapper.map(std::get<0>(it), std::get<1>(it));
                }
            }
            
            // Remove the if operation
            ifOp.erase();
        }
    }

    StringRef getArgument() const final { 
        return "remove-redundant-scf-if"; 
    }

    StringRef getDescription() const final { 
        return "Remove SCF if operations whose condition originates from a true krnl.global value through affine operations"; 
    }
};
} // end anonymous namespace

namespace onnx_mlir {
namespace krnl {

std::unique_ptr<mlir::Pass> createRemoveRedundantSCFIfPass() {
    return std::make_unique<RemoveRedundantSCFIfPass>();
}

} // namespace krnl
} // namespace onnx_mlir

static mlir::PassRegistration<RemoveRedundantSCFIfPass> pass;