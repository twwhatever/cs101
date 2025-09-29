#include <gtest/gtest.h>

#include "ring.h"

TEST(Ring, BasicTest) {
  StaticRingBuffer<int, 5> ring;
  EXPECT_EQ(ring.size(), 0);
  EXPECT_TRUE(ring.empty());
  ring.enqueue(0);
  EXPECT_EQ(ring.size(), 1);
  EXPECT_FALSE(ring.empty());
  EXPECT_EQ(ring.dequeue(), 0);
  EXPECT_EQ(ring.size(), 0);
  EXPECT_TRUE(ring.empty());
}

TEST(Ring, WraparoundTest) {
  StaticRingBuffer<int, 5> ring;

  for (int i = 0; i < 10; ++i) {
    ring.enqueue(i);
    EXPECT_LE(ring.size(), 5);
  }
  EXPECT_EQ(ring.size(), 5);
  for (int i = 5; i < 10; ++i) {
    EXPECT_EQ(ring.dequeue(), i);
  }
}

TEST(Ring, AddRemoveTest) {
  StaticRingBuffer<int, 2> ring;

  ring.enqueue(1);
  EXPECT_EQ(ring.size(), 1);
  ring.enqueue(2);
  EXPECT_EQ(ring.size(), 2);
  EXPECT_EQ(ring.dequeue(), 1);
  EXPECT_EQ(ring.size(), 1);
  ring.enqueue(3);
  EXPECT_EQ(ring.size(), 2);
  EXPECT_EQ(ring.dequeue(), 2);
  ring.enqueue(4);
  ring.enqueue(5);
  EXPECT_EQ(ring.dequeue(), 4);
}
