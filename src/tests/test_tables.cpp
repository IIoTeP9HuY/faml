#include <gtest/gtest.h>

#include <faml/new_table/table.hpp>
#include <faml/new_table/in_memory_table.hpp>

using namespace faml;

TEST(in_memory_table, ConstructorDefault) {
	InMemoryTable<int> table;
}

TEST(in_memory_table, ConstructorFromVector) {
	InMemoryTable<int> table({1, 2, 3});

	EXPECT_EQ(table[0], 1);
	EXPECT_EQ(table[1], 2);
	EXPECT_EQ(table[2], 3);
}

TEST(in_memory_table, Map) {
	auto table = std::make_shared<InMemoryTable<int>> (std::vector<int>({1, 2, 3}));
	auto mapped = table->map<int>([] (int x) -> int { return x * x; } );
}
