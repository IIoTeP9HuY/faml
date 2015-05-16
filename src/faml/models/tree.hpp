#ifndef TREE_HPP
#define TREE_HPP

#include <memory>
#include <cassert>

#include "faml/models/predictor.hpp"

namespace faml {
namespace impl {
	template <typename Label, typename Tree, typename Row>
	Label predict(const Tree& tree, const Row& sample, size_t root) {
		typename Tree::Node currentNode = root;
		while (!tree.isLeaf(currentNode)) {
			currentNode = tree.getAsInner(currentNode).nextNode(sample);
		}
		return tree.getAsLeaf(currentNode);
	}

	template <typename Label, typename Tree, typename Row>
	Table<Label> predict(const Tree& tree, const TableView<Row>& data, size_t root) {
		return data.cast(
				[&](const Row& row) {
					return predict<Label>(tree, row, root);
				}
			);
	} 
} //namesapce impl
template<typename Trainer>
class TreeClassifier : public Predictor<typename Trainer::Row, typename Trainer::Label> {
public:
	typedef typename Trainer::Row Row;
	typedef typename Trainer::Label Label;
	typedef decltype(std::declval<Trainer>().train(std::declval<TableView<Row>>(),
												   std::declval<TableView<Label>>())) Tree;

	TreeClassifier(const Trainer &trainer): trainer(trainer) {
	}

	void train(const TableView<Row> &samples, const TableView<Label> &labels) {
		std::vector<size_t> ind;
		for(size_t i = 0; i < samples.rowsNumber(); ++i) {
            ind.push_back(i);
		}
		tree = std::unique_ptr<Tree>(new Tree(trainer.train(samples[ind], labels[ind])));
		assert(tree->root < tree->size());
	}

	using Predictor<Row, Label>::predict;
	Label predict(const Row &sample) {
		assert(tree != nullptr);
		assert(tree->root < tree->size());
//		if(bad(sample))
//			return "<=50K";
		return impl::predict<Label>(*tree, sample, tree->root);
	}
	bool bad(const Row& sample) const {
		for(auto x: sample) {
			if(x == "?") {
				return true;
			}
		}
		return false;
	}
private:
	std::unique_ptr<Tree> tree;
	Trainer trainer;
};

} // namespace faml

#endif // TREE_HPP
