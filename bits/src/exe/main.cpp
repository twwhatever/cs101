#include <iostream>
#include "mylib.h"

struct Bar { int a; alignas(32) double b; alignas(128) char tag[7]; int xs[4]; };
BOOST_DESCRIBE_STRUCT(Bar, (), (a, b, tag, xs)) // <-- one line per type

struct MemberDesc {
    const char* name;
    std::size_t offset;     // offset of member (arrays: offset of elem 0)
    std::size_t size;       // total size (arrays: N * sizeof(T))
    std::size_t align;      // alignof(element type)
    bool        is_array;
    std::size_t array_len;  // 0 for scalars, N for arrays
    std::size_t elem_size;  // sizeof(element type)
};

template <class T> struct Reflect; // primary template

// ---------- Expanders (same arity for both decl/descr) ----------
#define DECL_SCALAR(T, type, name)         type name;
#define DECL_ARRAY(T, type, name, N)       type name[N];

#define DESCR_SCALAR(T, type, name) \
    { #name, offsetof(T, name), sizeof(type), alignof(type), false, 0, sizeof(type) },

#define DESCR_ARRAY(T, type, name, N) \
    { #name, offsetof(T, name), sizeof(type) * (N), alignof(type), true, (N), sizeof(type) },

// ---------- One macro: declare struct + reflection ----------
#define DECL_STRUCT_AND_REFLECT(Type, FIELDS)                                  \
    struct Type { FIELDS(Type, DECL_SCALAR, DECL_ARRAY) };                     \
    static_assert(std::is_standard_layout_v<Type>, "Type must be standard-layout"); \
    template <> struct Reflect<Type> {                                         \
        static constexpr MemberDesc members[] = { FIELDS(Type, DESCR_SCALAR, DESCR_ARRAY) }; \
        static constexpr std::size_t count = sizeof(members) / sizeof(members[0]); \
    };

// ---------- Define your fields ONCE per type ----------
// NOTE: first parameter is the enclosing Type
#define POINT_FIELDS(T, F, A) \
    F(T, float,        x)     \
    F(T, float,        y)     \
    A(T, std::uint8_t, flags, 4)

DECL_STRUCT_AND_REFLECT(Point, POINT_FIELDS)

#define BDESCR_SCALAR(T, type, name) name,
#define BDESCR_ARRAY(T, type, name, N) name,

#define DECL_STRUCT_AND_DESCRIBE(Type, FIELDS) \
    struct Type { FIELDS(Type, DECL_SCALAR, DECL_ARRAY) }; \
    BOOST_DESCRIBE_STRUCT(Type, (), ( FIELDS(Type, BDESCR_SCALAR, BDESCR_ARRAY) )) \

#define BUZ_FIELDS(T, F, A) \
    F(T, int, foo) \
    F(T, bool, bar) \
    A(T, char, baz, 1) \
    F(T, double, buz)

DECL_STRUCT_AND_DESCRIBE(Buz, BUZ_FIELDS)

template <class T>
void print_reflection() {
    T obj;
    const char* base = reinterpret_cast<const char*>(&obj);
    std::cout << "sizeof(T)=" << sizeof(T) << " alignof(T)=" << alignof(T) << "\n";
    for (std::size_t i = 0; i < Reflect<T>::count; ++i) {
        const auto& m = Reflect<T>::members[i];
        std::cout << "- " << m.name
                  << " off="   << m.offset
                  << " size="  << m.size
                  << " align=" << m.align;
        if (!m.is_array) std::cout << "\n";
        else {
            std::cout << " (array len=" << m.array_len
                      << ", elem=" << m.elem_size << ")\n";
            for (std::size_t j = 0; j < m.array_len; ++j)
                std::cout << "    [" << j << "] off=" << (m.offset + j * m.elem_size) << "\n";
        }
    }
}

int main() {
    std::cout << "boost/pfr based layout: simple struct with offsets\n\n";

    struct Foo {
        double buz;
        int bar;
        bool baz;
    };

    printOffsets<Foo>();

    std::cout << "\n\nboost/Describe based layout: more complex struct with names\n\n";

    print_layout<Bar>();

    std::cout << "\n\nCustom reflect based layout: macro struct\n\n";

    print_reflection<Point>();

    std::cout << "\n\nboost::Describe based layout: macro struct\n\n";

    print_layout<Buz>();

    return 0;
}
