#ifndef IO_HPP
#define IO_HPP

#include "faml/data/table.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>

namespace faml {

std::vector<std::string> parseCSVHeader(const std::string& line) {
	std::vector<std::string> header;

	std::istringstream ss(line);
	std::string token;
	while (std::getline(ss, token, ',')) {
		header.push_back(token);
	}

	return header;
}

std::vector<std::string> readCSVHeader(std::ifstream &inputStream) {
	std::string line;
	std::getline(inputStream, line);
	return parseCSVHeader(line);
}

Table< std::vector<std::string> > readCSV(const std::string &filename,
											bool withHeader=true) {
	typedef std::vector<std::string> RowType;
	std::ifstream inputStream(filename);
	std::vector<std::string> header;
	if (withHeader) {
		header = readCSVHeader(inputStream);
	}
	size_t headerLength = header.size();
	Table<RowType> table(header);

	while (!inputStream.eof()) {
		std::string line;
		getline(inputStream, line);

		if (!headerLength) {
			headerLength = std::count(line.begin(), line.end(), ',') + 1;
			for (size_t i = 0; i < headerLength; ++i) {
				header.push_back(std::to_string(i));
			}
			table = Table<RowType>(header);
		}

		if (inputStream.eof()) {
			break;
		}

		std::istringstream ss(line);

		RowType data(headerLength);

		for (size_t i = 0; i < headerLength; ++i) {
			std::getline(ss, data[i], ',');
		}

		table.addRow(data);
	}
	return table;
}

} // namespace faml

#endif // IO_HPP
