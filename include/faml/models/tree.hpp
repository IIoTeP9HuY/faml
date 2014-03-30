#ifndef TREE_HPP
#define TREE_HPP

#include <memory>

namespace faml {

template<typename Trainer>
class TreeClassifier {
public:
	typedef Trainer::Row Row;
	typedef Trainer::Label Label;
	typedef decltype(std::declval<Trainer>().train(std::declval<Row>(), std::declval<Label>())) Tree;

	TreeClassifier(const Trainer &trainer): trainer(trainer) {
	}

	void train() {
		tree = std::unique_ptr<Tree>(new Tree(trainer.train()));
	}

private:
	std::unique_ptr<Tree> tree;
	Trainer trainer;
};

} // namespace faml

#endif // TREE_HPP
