#pragma once

#include <cstddef>
#include <iostream>
#include <boost/describe.hpp>
#include <boost/mp11.hpp>

#define DECL_SCALAR(T, type, name)         type name;
#define DECL_ARRAY(T, type, name, N)       type name[N];

#define BDESCR_SCALAR(T, type, name) name,
#define BDESCR_ARRAY(T, type, name, N) name,

#define DECL_STRUCT_AND_DESCRIBE(Type, FIELDS) \
    struct Type { FIELDS(Type, DECL_SCALAR, DECL_ARRAY) }; \
    BOOST_DESCRIBE_STRUCT(Type, (), ( FIELDS(Type, BDESCR_SCALAR, BDESCR_ARRAY) )) \

template <class T>
void print_layout() {
    T obj;
    using D = boost::describe::describe_members<T, boost::describe::mod_public>;
    const char* base = reinterpret_cast<const char*>(&obj);

    boost::mp11::mp_for_each<D>([&](auto Dmem) {
        constexpr auto pm = Dmem.pointer;     // pointer-to-member (works for arrays)
        const char* name = Dmem.name;

        const auto* field_ptr = reinterpret_cast<const char*>(&(obj.*pm));
        std::size_t off = static_cast<std::size_t>(field_ptr - base);

        // sizeof/alignof via decltype(obj.*pm)
        using FieldRef = decltype(obj.*pm);          // e.g., int (&)[4]
        using Field    = std::remove_reference_t<FieldRef>;
        std::cout << name
                  << " off=" << off
                  << " size=" << sizeof(Field)
                  << " align=" << alignof(std::remove_all_extents_t<Field>)
                  << "\n";

        if constexpr (std::is_array_v<Field>) {
            using Elem = std::remove_extent_t<Field>;
            constexpr std::size_t N = std::extent_v<Field>;
            for (std::size_t i = 0; i < N; ++i) {
                std::cout << "    [" << i << "] off=" << off + i * sizeof(Elem)
                          << " size=" << sizeof(Elem) << "\n";
            }
        }
    });
}