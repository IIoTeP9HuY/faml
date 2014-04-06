#ifndef TREE_TREE_HPP
#define TREE_TREE_HPP

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
		return leafs[node] != nullptr;
	}

	Leaf getAsLeaf(Node node) const {
		return *leafs[node];
	}

	Inner getAsInner(Node node) const {
		return *inners[node];
	}

	void setLeaf(Node node, const Leaf& leaf) {
		inners[node] = nullptr;
		leafs[node] = std::unique_ptr<Leaf>(new Leaf(leaf));
	}

	void setInnerNode(Node node, const Inner& inner) {
		inners[node] = std::unique_ptr<Inner>(new Inner(inner));
		leafs[node] = nullptr;
	}

	size_t newNode() {
		size_t res = inners.size();
		inners.emplace_back(nullptr);
		leafs.emplace_back(nullptr);
		return res;
	}
	size_t root = 0;
};

} // namespace faml

#endif // TREE_TREE_HPP
