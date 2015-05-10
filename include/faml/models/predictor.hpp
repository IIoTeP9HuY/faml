#ifndef PREDICTOR_HPP
#define PREDICTOR_HPP

#include "faml/data/table.hpp"

namespace faml {

template <typename Row, typename Label>
class Predictor {
public:
	virtual ~Predictor() {}

	virtual void train(const TableView<Row>&, const TableView<Label>&) = 0;
	virtual Label predict(const Row&) = 0;

	virtual Label predict(const std::pair<Row, Label> sampleAndLabel) {
		return predict(sampleAndLabel.first);
	}

	virtual Table<Label> predict(const TableView<Row> &samples) {
		std::vector<Label> predictions(samples.rowsNumber());

		#pragma omp parallel for
		for (size_t i = 0; i < samples.rowsNumber(); ++i) {
			predictions[i] = predict(samples[i]);
		}

		return Table<Label> ({"prediction"},
							 std::make_move_iterator(predictions.begin()),
							 std::make_move_iterator(predictions.end()));
	}
};

} // namespace faml

#endif // PREDICTOR_HPP
