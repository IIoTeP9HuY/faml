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
	typedef decltype(std::declval<Trainer>().train(std::declval<Row>(), std::declval<Label>())) Tree;

	TreeClassifier(const Trainer &trainer): trainer(trainer) {
	}

	void train() {
		tree = std::unique_ptr<Tree>(new Tree(trainer.train()));
	}

	Label predict(const Row &sample) {
		typename Tree::Node currentNode;
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
