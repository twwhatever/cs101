#include <iostream>
#include "layout.h"

struct Bar { int a; alignas(32) double b; alignas(128) char tag[7]; int xs[4]; };
BOOST_DESCRIBE_STRUCT(Bar, (), (a, b, tag, xs)) // <-- one line per type

#define BUZ_FIELDS(T, F, A) \
    F(T, int, foo) \
    F(T, bool, bar) \
    A(T, char, baz, 1) \
    F(T, double, buz)

DECL_STRUCT_AND_DESCRIBE(Buz, BUZ_FIELDS)

int main() {
    std::cout << "\n\nboost/Describe based layout: more complex struct with names\n\n";

    print_layout<Bar>();

    std::cout << "\n\nboost::Describe based layout: macro struct\n\n";

    print_layout<Buz>();

    return 0;
}
