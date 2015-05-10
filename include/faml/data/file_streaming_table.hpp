#pragma once

#include <fstream>

#include "streaming_table.hpp" 

namespace faml {

template<typename RowType, typename FileReader, typename ParserFactory>
class FileStreamingTable : public StreamingTable<RowType> {
	typedef StreamingTable<RowType> BaseTable;
	typedef typename BaseTable::BaseIterator ParentBaseIterator;
	typedef typename BaseTable::iterator iterator;
	typedef typename ParserFactory::Parser Parser;
public:
	class BaseIterator : public ParentBaseIterator {
	public:
		BaseIterator(std::unique_ptr<Parser> parser): parser(std::move(parser)) {}

		virtual BaseIterator& operator ++ () {
			if (parser && !parser->isFinished()) {
				currentRow = parser->getRow();
			}
			return *this;
		}

		virtual BaseIterator& operator -- () {
			throw std::logic_error("Can't decrement forward iterator");
		}

		virtual const RowType& operator * () const {
			if (!parser) {
				throw std::logic_error("Dereferencing end iterator");
			}
			return currentRow;
		}

		virtual bool operator == (const ParentBaseIterator& rhs) const {
			try {
				const BaseIterator &r = dynamic_cast<const BaseIterator&> (rhs);
				if (parser && r.parser) {
					return parser->getRowNumber() == r.parser->getRowNumber();
				}
				if (!parser) {
					return r.parser->isFinished();
				} else {
					return parser->isFinished();
				}
			} catch (std::bad_cast&) {
				return false;
			}
		}
	private:
		RowType currentRow;
		std::unique_ptr<Parser> parser;
	};

	FileStreamingTable(const std::string& filename, ParserFactory parserFactory)
		: filename(filename), parserFactory(parserFactory)
	{ }

	virtual ~FileStreamingTable() {}

	virtual iterator begin() const {
		return iterator(std::unique_ptr<BaseIterator>(
			new BaseIterator(parserFactory.getInstance(
					std::unique_ptr<FileReader>(new FileReader(filename))))));
	}

	virtual iterator end() const {
		return iterator(std::unique_ptr<BaseIterator>(new BaseIterator(nullptr)));
	}

private:
	const std::string& filename;
	ParserFactory parserFactory;
};

} // namespace faml
