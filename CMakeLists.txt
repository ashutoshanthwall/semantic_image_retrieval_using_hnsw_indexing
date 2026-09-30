cmake_minimum_required(VERSION 3.20)
project(semantic_image_retrieval VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Single-config build. Benchmarks must use Release with exactly the pinned flags.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG" CACHE STRING "" FORCE)

include(FetchContent)
FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(json googletest)

set(SIR_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)

file(GLOB_RECURSE SIR_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
add_library(sir_core STATIC ${SIR_SOURCES})
target_include_directories(sir_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(sir_core PUBLIC nlohmann_json::nlohmann_json)
target_compile_options(sir_core PRIVATE ${SIR_WARNINGS})

foreach(app sir_cli sir_benchmark sir_build_index sir_gate)
  if(EXISTS ${CMAKE_SOURCE_DIR}/apps/${app}.cpp)
    add_executable(${app} apps/${app}.cpp)
    target_link_libraries(${app} PRIVATE sir_core)
    target_compile_options(${app} PRIVATE ${SIR_WARNINGS})
  endif()
endforeach()

enable_testing()
include(GoogleTest)
file(GLOB SIR_TESTS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/tests/*.cpp)
if(SIR_TESTS)
  add_executable(sir_tests ${SIR_TESTS})
  target_link_libraries(sir_tests PRIVATE sir_core GTest::gtest_main)
  target_compile_definitions(sir_tests PRIVATE SIR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  gtest_discover_tests(sir_tests DISCOVERY_TIMEOUT 60)
endif()
