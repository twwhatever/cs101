#include "mylib.h"
#include <gtest/gtest.h>

TEST(MyLibTest, Add) {
    EXPECT_EQ(add(1, 2), 3);
    EXPECT_EQ(add(-1, 1), 0);
}
