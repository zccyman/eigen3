# Optional dependency for some tests (boost::variant is only supported with version >= 1.56)
find_package(Boost 1.56)

if(Boost_FOUND)
  if(NOT TARGET Boost::headers)
    add_library(Boost::headers IMPORTED INTERFACE)
    if(TARGET Boost::boost)
      # Classic FindBoost
      set_property(TARGET Boost::boost PROPERTY INTERFACE_LINK_LIBRARIES Boost::boost)
    else()
      # Very old FindBoost, or newer Boost than CMake in older CMakes
      set_property(TARGET Boost::headers PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                                  ${Boost_INCLUDE_DIRS})
    endif()
  endif()
endif()


# add target link with build boost
if(Boost_FOUND)
target_link_libraries(${target} PRIVATE Boost::headers)
target_compile_definitions(${target} PRIVATE -DPYBIND11_TEST_BOOST)
endif()