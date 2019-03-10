#define BOOST_TEST_MODULE Partitioner Tests
#include <boost/test/included/unit_test.hpp>
#include <boost/assign/list_of.hpp>
#include "../lib/partitioner.h"
#include <vector>

using namespace std;

BOOST_AUTO_TEST_CASE(smoke_test)
{
  vector<int> v = boost::assign::list_of(3)(2)(1);
  BasicPartitioner<vector<int>::iterator> p;
  p.partition(v.begin(), v.end(), 1);
  vector<int> expected = boost::assign::list_of(1)(2)(3);
  BOOST_TEST(v == expected);
}
