#include "uf.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace testing;

TEST(UnionFindTest, Init) {
  UnionFind<std::string> uf;

  EXPECT_THAT(uf.size(), Eq(0));
}

TEST(UnionFindTest, Unconnected) {
  UnionFind<std::string> uf;
  uf.add("one");
  EXPECT_THAT(uf.size(), Eq(1));
  EXPECT_THAT(uf.set_count("one"), Eq(1));
  uf.add("two");
  EXPECT_THAT(uf.size(), Eq(2));
  EXPECT_THAT(uf.set_count("one"), Eq(1));
  EXPECT_THAT(uf.set_count("two"), Eq(1));
}

TEST(UnionFindTest, Connected) {
  UnionFind<std::string> uf;
  uf.add("one");
  uf.add("two");
  uf.connect("one", "two");
  EXPECT_THAT(uf.size(), Eq(1));
  EXPECT_THAT(uf.set_count("one"), Eq(2));
  EXPECT_THAT(uf.set_count("two"), Eq(2));
}

TEST(UnionFindTest, IntType) {
  UnionFind<int> uf;
  uf.add(1);
  uf.add(2);
  uf.add(3);
  EXPECT_THAT(uf.size(), Eq(3));
  uf.connect(1, 2);
  EXPECT_THAT(uf.size(), Eq(2));
  EXPECT_THAT(uf.query(1, 2), Eq(true));
  EXPECT_THAT(uf.query(1, 3), Eq(false));
  uf.connect(2, 3);
  EXPECT_THAT(uf.size(), Eq(1));
  EXPECT_THAT(uf.query(1, 3), Eq(true));
}
