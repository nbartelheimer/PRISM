cmake_minimum_required(VERSION 3.26)
project(prism LANGUAGES C)

find_package(MPI REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OFI REQUIRED IMPORTED_TARGET libfabric)

add_library(prism SHARED prism.c)
target_compile_features(prism PRIVATE c_std_11)
target_include_directories(prism PRIVATE "../include")
target_link_libraries(prism PRIVATE MPI::MPI_C PkgConfig::OFI)
