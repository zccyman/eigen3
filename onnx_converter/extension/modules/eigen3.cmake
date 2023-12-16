set(PYBIND11_EIGEN_REPO
    "https://gitlab.com/libeigen/eigen.git"
    CACHE STRING "Eigen repository to use for tests")
# Always use a hash for reconfigure speed and security reasons
# Include the version number for pretty printing (keep in sync)
set(PYBIND11_EIGEN_VERSION_AND_HASH
    "3.4.0;929bc0e191d0927b1735b9a1ddc0e8b77e3a25ec"
    CACHE STRING "Eigen version to use for tests, format: VERSION;HASH")

list(GET PYBIND11_EIGEN_VERSION_AND_HASH 0 PYBIND11_EIGEN_VERSION_STRING)
list(GET PYBIND11_EIGEN_VERSION_AND_HASH 1 PYBIND11_EIGEN_VERSION_HASH)

# Check if Eigen is available; if not, remove from PYBIND11_TEST_FILES (but
# keep it in PYBIND11_PYTEST_FILES, so that we get the "eigen is not installed"
# skip message).
list(FIND PYBIND11_TEST_FILES test_eigen.cpp PYBIND11_TEST_FILES_EIGEN_I)
if(PYBIND11_TEST_FILES_EIGEN_I GREATER -1)
  # Try loading via newer Eigen's Eigen3Config first (bypassing tools/FindEigen3.cmake).
  # Eigen 3.3.1+ exports a cmake 3.0+ target for handling dependency requirements, but also
  # produces a fatal error if loaded from a pre-3.0 cmake.
  if(DOWNLOAD_EIGEN)
    if(CMAKE_VERSION VERSION_LESS 3.11)
      message(FATAL_ERROR "CMake 3.11+ required when using DOWNLOAD_EIGEN")
    endif()

    include(FetchContent)
    FetchContent_Declare(
      eigen
      GIT_REPOSITORY "${PYBIND11_EIGEN_REPO}"
      GIT_TAG "${PYBIND11_EIGEN_VERSION_HASH}")

    FetchContent_GetProperties(eigen)
    if(NOT eigen_POPULATED)
      message(
        STATUS
          "Downloading Eigen ${PYBIND11_EIGEN_VERSION_STRING} (${PYBIND11_EIGEN_VERSION_HASH}) from ${PYBIND11_EIGEN_REPO}"
      )
      FetchContent_Populate(eigen)
    endif()

    set(EIGEN3_INCLUDE_DIR ${eigen_SOURCE_DIR})
    set(EIGEN3_FOUND TRUE)
    # When getting locally, the version is not visible from a superprojet,
    # so just force it.
    set(EIGEN3_VERSION "${PYBIND11_EIGEN_VERSION_STRING}")

  else()
    find_package(Eigen3 3.2.7 QUIET CONFIG)

    if(NOT EIGEN3_FOUND)
      # Couldn't load via target, so fall back to allowing module mode finding, which will pick up
      # tools/FindEigen3.cmake
      find_package(Eigen3 3.2.7 QUIET)
    endif()
  endif()

  if(EIGEN3_FOUND)
    if(NOT TARGET Eigen3::Eigen)
      add_library(Eigen3::Eigen IMPORTED INTERFACE)
      set_property(TARGET Eigen3::Eigen PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                                 "${EIGEN3_INCLUDE_DIR}")
    endif()

    # Eigen 3.3.1+ cmake sets EIGEN3_VERSION_STRING (and hard codes the version when installed
    # rather than looking it up in the cmake script); older versions, and the
    # tools/FindEigen3.cmake, set EIGEN3_VERSION instead.
    if(NOT EIGEN3_VERSION AND EIGEN3_VERSION_STRING)
      set(EIGEN3_VERSION ${EIGEN3_VERSION_STRING})
    endif()
    message(STATUS "Building tests with Eigen v${EIGEN3_VERSION}")
  else()
    list(REMOVE_AT PYBIND11_TEST_FILES ${PYBIND11_TEST_FILES_EIGEN_I})
    message(
      STATUS "Building tests WITHOUT Eigen, use -DDOWNLOAD_EIGEN=ON on CMake 3.11+ to download")
  endif()
endif()

# add target link with build eigen3
#if(EIGEN3_FOUND)
#    target_link_libraries(${target} PRIVATE Eigen3::Eigen)
#    target_compile_definitions(${target} PRIVATE -DPYBIND11_TEST_EIGEN)
#endif()

