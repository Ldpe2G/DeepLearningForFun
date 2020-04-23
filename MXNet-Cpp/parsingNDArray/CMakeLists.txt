cmake_minimum_required(VERSION 2.8)
PROJECT(read_mx_nd)

SET(CMAKE_CXX_FLAGS_DEBUG " $ENV{CXXFLAGS}  -DDEBUG -g -w")
SET(CMAKE_CXX_FLAGS_RELEASE " $ENV{CXXFLAGS}   -DNDEBUG -O3 -w")
SET(PLATFORM ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR})

if(NOT DEFINED CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE release)
endif()

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_definitions(-DDEBUG)
endif()

set (CMAKE_EXE_LINKER_FLAGS -pthread)
add_compile_options(-std=c++11)
INCLUDE_DIRECTORIES(/usr/local/include)
LINK_DIRECTORIES(/usr/local/lib /usr/lib)


include_directories("${PROJECT_SOURCE_DIR}/dmlc-core/include")
include_directories("${PROJECT_SOURCE_DIR}/include")

add_subdirectory(dmlc-core)

ADD_EXECUTABLE(read_nd read_mx_nd.cpp)
TARGET_LINK_LIBRARIES(read_nd dmlc)

