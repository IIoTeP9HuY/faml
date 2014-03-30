#ifndef ID3Trainer
#define ID3Trainer
#include "faml/models/tree/tree.hpp"
#include <limits>
namespace faml {

template<typename Row, typename T = decltype(declval<Row>()[0])>
struct ID3Node {
	size_t l, r;
	size_t index;
	T value;
	size_t nextNode(const Row& row) const {
		return row[index] == value ? l : r;
	}
};
template<typename Row, typename Label>
class ID3Trainer {
	public:
	Tree<ID3Node, Label> train(const TableView<Row>& x, const TableView<Label>& y) {
		Tree<ID3Node, Label> tree;
		train(x, y, 0);
		return tree;
	}

private:
	void train(const TableView<Row>& x, const TableView<Label>& y, size_t node) { 
		typedef decltype(x[0]) T;
		int size = x.columnsNumber();
		bool split = false;
		double bestInformativity = std::numeric_limits<double>::min();
		double bestIndex;
		vector<size_t> bestIndices, bestOtherIndices;
		for(size_t i = 0; i < size; ++i) {
			std::map<T, vector<size_t>> valueIndices;
			for(size_t row = 0; row < x.rowsNumber(); ++row) {
				valueIndices[x[row]].push_back(row);
			}
			for(const auto& col : valueIndices) {
				if(col.valueIndices.size() == x.rowsNumber()) {
					continue;
				}
				auto others = otherIndices(col.second, size);
				double informativity = criteria(y[col.second], y[others]);
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
			ID3Node currentNode(l, r, bestIndex, x[bestIndicies[0]][bestIndex]);
			tree.setInnerNode(node currentNode);
			train(x[bestIndices], y[bestIndices], l);
			train(x[bestOtherIndicies], y[bestOtherIndices], r);
		}
		else {
			tree.setLeaf(node, majorantClass(y));
		}
	}

private:
	std::vector<size_t> otherInices (const std::vector<size_t>& valueIndices, size_t size) {
		std::vector<size_t> result;
		size_t pos;
		for(size_t i = 0; i < size; ++i) {
			if(pos != valueIndices.size() && valueIndices[pos] == i) {
				++pos;
			}
			else
				result.push_back(i);
		}
		return result;
	}

	std::shared_ptr<InformativityCriteria> criteria;
};

} //faml

#endif
