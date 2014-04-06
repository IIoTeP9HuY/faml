#ifndef TREE_TREE_HPP
#define TREE_TREE_HPP

#include <cassert>
namespace faml {

template<typename Inner, typename Leaf>
class Tree {
public:
	typedef size_t Node;
private:
	std::vector<std::unique_ptr<Inner>> inners;
	std::vector<std::unique_ptr<Leaf>> leafs;

public:

	bool isLeaf(Node node) const {
		assert(node < size());
		return leafs[node] != nullptr;
	}

	Leaf& getAsLeaf(Node node) const {
		assert(node < size());
		return *leafs[node];
	}

	Inner& getAsInner(Node node) const {
		assert(node < size());
		return *inners[node];
	}

	void setLeaf(Node node, const Leaf& leaf) {
		assert(node < size());
		inners[node] = nullptr;
		leafs[node] = std::unique_ptr<Leaf>(new Leaf(leaf));
	}

	void setInnerNode(Node node, const Inner& inner) {
		assert(node < size());
		inners[node] = std::unique_ptr<Inner>(new Inner(inner));
		leafs[node] = nullptr;
	}

	size_t newNode() {
		size_t res = size();
		inners.emplace_back(nullptr);
		leafs.emplace_back(nullptr);
		return res;
	}
	size_t size() const {
		return inners.size();
	}
	size_t root = 0;
};

} // namespace faml

#endif // TREE_TREE_HPP
