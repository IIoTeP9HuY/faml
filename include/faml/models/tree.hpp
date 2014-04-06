#ifndef TREE_HPP
#define TREE_HPP

#include <memory>

#include "faml/models/predictor.hpp"

namespace faml {

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
		tree = std::unique_ptr<Tree>(new Tree(trainer.train(samples, labels)));
	}

	using Predictor<Row, Label>::predict;
	Label predict(const Row &sample) {
		typename Tree::Node currentNode = tree.root;
		while (!tree->isLeaf(currentNode)) {
			currentNode = tree->getAsInner(currentNode).nextNode(sample);
		}
		return tree->getAsLeaf(currentNode);
	}

private:
	std::unique_ptr<Tree> tree;
	Trainer trainer;
};

} // namespace faml

#endif // TREE_HPP
