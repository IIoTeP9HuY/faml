#ifndef PREDICTOR_HPP
#define PREDICTOR_HPP
#include "faml/data/table.hpp"

namespace faml {
template <typename Row, typename Label>
class Predictor {
public:
	virtual void train(const TableView<Row>&, const TableView<Label>&) = 0;
	virtual Label predict(const Row&) = 0;
	virtual Table<Label> predict(const TableView<Row>&) = 0;

	virtual ~Predictor() {}
};

} // namespace faml
#endif
