#ifndef ID3TRAINER_HPP
#define ID3TRAINER_HPP

#include <limits>
#include <unordered_map>
#include <type_traits>
#include <memory>

#include "faml/models/tree/tree.hpp"
#include "faml/statistics/informativity_criteria.hpp"
#include "faml/utility/utility.hpp"

#include <cassert>
#include <stdexcept>
#include <iostream>
namespace faml {

template<typename Row, typename T = typename std::decay<decltype(std::declval<Row>()[0])>::type>
struct ID3Node {
	size_t l, r;
	size_t index;
	T value;
	ID3Node(size_t l, size_t r, size_t index, const T& value): l(l), r(r), index(index), value(value) {

	}

	size_t nextNode(const Row& row) const {
		return row[index] == value ? l : r;
	}
};

template<typename _Row, typename _Label>
class ID3Trainer {
public:
	typedef _Row Row;
	typedef _Label Label;
	typedef Tree<ID3Node<Row>, Label> TrainedTree;

	ID3Trainer(std::shared_ptr<InformativityCriteria<Label>> criteria): criteria(criteria) {
		
	}
	TrainedTree train(const TableView<Row>& x, const TableView<Label>& y) {
		TrainedTree tree;
		tree.newNode();
		train(x, y, tree, 0, 0);
		return tree;
	}

private:
	void train(const TableView<Row>& x, const TableView<Label>& y, TrainedTree &tree, size_t node, size_t level) {

		if (x.rowsNumber() != y.rowsNumber()) {
			throw std::invalid_argument("x.rowsNumber() != y.rowsNumber()");
		}

		typedef typename std::decay<decltype(x[0][0])>::type T;
		size_t size = x.columnsNumber();
		bool split = false;
		double bestInformativity = std::numeric_limits<double>::min();
		double bestIndex;
		std::vector<size_t> bestIndices, bestOtherIndices;
		for(size_t i = 0; i < size; ++i) {
			if(i == 1 || i == 6 || i == 13)
				continue;
			std::unordered_map<T, std::vector<size_t>> valueIndices;
			for(size_t row = 0; row < x.rowsNumber(); ++row) {
				valueIndices[x[row][i]].push_back(row);
			}
			if (valueIndices.size() > 30) {
				continue;
			}
			valueIndices.erase("?");
			for(const auto& col : valueIndices) {
				if(col.second.size() == x.rowsNumber()) {
					continue;
				}
				auto others = otherIndices(col.second, x.rowsNumber());
				double informativity = (*criteria)(y[col.second], y[others]);
				if(informativity > bestInformativity) {
					split = true;
					bestInformativity = informativity;
					bestIndex = i;
					bestIndices = col.second;
					bestOtherIndices = others;
				}
			}
		}
		if(split) {
			size_t l = tree.newNode();
			size_t r = tree.newNode();
			ID3Node<Row> currentNode(l, r, bestIndex, x[bestIndices[0]][bestIndex]);
			tree.setInnerNode(node, currentNode);
			train(x[bestIndices], y[bestIndices], tree, l, level + 1);
			train(x[bestOtherIndices], y[bestOtherIndices], tree, r, level + 1);
		}
		else {
			tree.setLeaf(node, majorantClass(y));
		}
	}

private:
	std::vector<size_t> otherIndices (const std::vector<size_t>& valueIndices, size_t size) {
		assert(valueIndices.empty() || valueIndices.back() < size);
		assert(std::is_sorted(valueIndices.begin(), valueIndices.end()));
		std::vector<size_t> result;
		size_t pos = 0;
		for(size_t i = 0; i < size; ++i) {
			if(pos != valueIndices.size() && valueIndices[pos] == i) {
				++pos;
			}
			else
				result.push_back(i);
		}
		return result;
	}

	std::shared_ptr<InformativityCriteria<Label>> criteria;
};

} // namespace faml

#endif // ID3TRAINER_HPP
