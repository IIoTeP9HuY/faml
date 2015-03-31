#ifndef ID3_PRUNING_TRAINER
#define ID3_PRUNING_TRAINER
#include "faml/models/tree/trainers/ID3Trainer.hpp"
#include "faml/models/tree.hpp"
#include "faml/quality/classification.hpp"
#include "faml/cross_validation/shuffle_split.hpp"
#include <cassert>
#include <stdexcept>

namespace faml {

template <typename _Row, typename _Label>
class ID3PruningTrainer  {
public:
	typedef _Row Row;
	typedef _Label Label;
	typedef ID3Trainer<Row, Label> Trainer;
	typedef typename Trainer::TrainedTree TrainedTree;
	ID3PruningTrainer(std::shared_ptr<InformativityCriteria<Label>> criteria, double trainingPart, uint_fast32_t seed):
	    trainer(criteria), trainingPart(trainingPart), seed(seed)
	{

	}

	TrainedTree train(const TableView<Row>& x, const TableView<Label>& y) {
		if(x.rowsNumber() != y.rowsNumber()) {
			throw std::invalid_argument("ID3PruningTree::train() x.size() != y.size()");
		}
		auto indices = trainTestSplit(x.rowsNumber(), trainingPart, seed);
		TrainedTree prepruned = trainer.train(x[indices.first], y[indices.first]);
		assert(prepruned.root == 0);
		rec(prepruned, prepruned.root, -1, x[indices.second], y[indices.second]);
		return prepruned;
	}

private:

	void rec(TrainedTree &tree, size_t v, int parent, const TableView<Row> &x, const TableView<Label> &y) {
		assert(x.rowsNumber() == y.rowsNumber());
		assert(v < tree.size());
		assert(parent < static_cast<int>(tree.size()) || parent == -1);
		if(x.rowsNumber() == 0)
			return;
		if(tree.isLeaf(v))
			return;
		Label major = majorantClass(y);
		size_t correctMajor = 0;
		for(const auto& cls: y) {
			if(major == cls) {
				++correctMajor;
			}
		}
		auto l = tree.getAsInner(v).l;
		auto r = tree.getAsInner(v).r;
		size_t correctAsIs = countCorrectPredictions(y, impl::predict<Label>(tree, x, v));
		size_t correctL = countCorrectPredictions(y, impl::predict<Label>(tree, x, l));
		size_t correctR = countCorrectPredictions(y, impl::predict<Label>(tree, x, r));
		size_t maxCorrect = std::max(std::max(correctMajor, correctAsIs), std::max(correctL, correctR));
		if(maxCorrect == correctMajor) {
			tree.setLeaf(v, major);
			return;
		}
		if(maxCorrect == correctAsIs) {
			std::vector<size_t> indL, indR;
			for(size_t i = 0; i < x.rowsNumber(); ++i) {
				if(tree.getAsInner(v).nextNode(x[i]) == l)
					indL.push_back(i);
				else
					indR.push_back(i);
			}
			rec(tree, l, v, x[indL], y[indL]);
			rec(tree, r, v, x[indR], y[indR]);
			return;
		}
		if(maxCorrect == correctL) {
			changeChild(tree, parent, v, l);
			rec(tree, l, parent, x, y);
			return;
		}
		if(maxCorrect == correctR) {
			changeChild(tree, parent, v, r);
			rec(tree, r, parent, x, y);
			return;
		}

		assert(false);
	}

	void changeChild(TrainedTree &tree, int parent, size_t oldV, size_t newV) {
		assert(oldV < tree.size());
		assert(parent < static_cast<int>(tree.size()) || parent == -1);
		assert(newV < tree.size());
		if(parent == -1) {
			tree.root = newV;
		}
		else {
			if(tree.getAsInner(parent).l == oldV) {
				tree.getAsInner(parent).l = newV;
			}
			if(tree.getAsInner(parent).r == oldV) {
				tree.getAsInner(parent).r = newV;
			}
		}
	}

	Trainer trainer;
	double trainingPart;
	uint_fast32_t seed;
};

} // faml


#endif //ID3_PRUNING_TRAINER
