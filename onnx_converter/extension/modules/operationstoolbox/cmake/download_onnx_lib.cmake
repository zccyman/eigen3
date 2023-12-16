find_file(
    LOCAL_ORT_LIB_ZIP
    NAMES ${ORT_LIB_ZIP_NAME}
    PATHS "ort_package" ${SOLUTION_DIR}/build/ort_package)
find_file(
    LOCAL_ORT_LIB_DIR
    NAMES ${ORT_LIB_NAME}
    PATHS "ort_lib"
    NO_DEFAULT_PATH)
find_file(
    LOCAL_ORT_LIB_ZIP_DL_DIR
    NAMES ""
    PATHS "ort_lib"
    NO_DEFAULT_PATH)
# message(${LOCAL_ORT_LIB_ZIP})
if(NOT LOCAL_ORT_LIB_ZIP)
    set(EXIST_ROOT_DIR 1)
else()
    string(COMPARE LESS LOCAL_ORT_LIB_ZIP ${SOLUTION_DIR}/build EXIST_ROOT_DIR)
endif()
# message(${EXIST_ROOT_DIR})
if(NOT LOCAL_ORT_LIB_ZIP)
    message(
        STATUS "Downloading ${ORT_LIB_ZIP_NAME} to ${LOCAL_ORT_LIB_ZIP_DL_DIR}")
    file(
        DOWNLOAD ${ORT_LIB_DOWNLOAD_URL}
        ${LOCAL_ORT_LIB_ZIP_DL_DIR}/${ORT_LIB_ZIP_NAME}
        TIMEOUT ${DOWNLOAD_ORT_LIB_TIMEOUT}
        STATUS ERR
        SHOW_PROGRESS)
    if(ERR EQUAL 0)
        set(LOCAL_ORT_LIB_ZIP "${LOCAL_ORT_LIB_ZIP_DL_DIR}/${ORT_LIB_ZIP_NAME}")
    else()
        message(STATUS "Download failed, error: ${ERR}")
        message(
            FATAL_ERROR
                "You can try downloading ${ORT_LIB_DOWNLOAD_URL} manually"
                " using curl/wget or a similar tool")
    endif()
endif()
if(LOCAL_ORT_LIB_ZIP)
    message(
        STATUS
            "Checking dblink ${ORT_LIB_UNZIP_DIR_NAME} + ${LOCAL_ORT_INCLUDE_DIR_NAME} + ${LOCAL_ORT_INCLUDE_DIR_NAME} "
    )
    if(NOT EXISTS "${LOCAL_ORT_LIB_ZIP_DL_DIR}/${ORT_LIB_UNZIP_DIR_NAME}"
       OR NOT EXISTS "${LOCAL_ORT_LIB_ZIP_DL_DIR}/${LOCAL_ORT_INCLUDE_DIR_NAME}"
       OR NOT EXISTS
          "${LOCAL_ORT_LIB_ZIP_DL_DIR}/${LOCAL_ORT_INCLUDE_DIR_NAME}")

        message(
            STATUS "cd ${LOCAL_ORT_LIB_ZIP_DL_DIR}; unzip ${LOCAL_ORT_LIB_ZIP}")
        set(PYPATH ${SOLUTION_DIR}/cmake)

        if(${EXIST_ROOT_DIR} EQUAL 0)
            set(DECOMMPRESS_PATH ${LOCAL_ORT_LIB_ZIP})
        else()
            set(DECOMMPRESS_PATH ${SOLUTION_DIR}/build/${LOCAL_ORT_LIB_ZIP})
        endif()

        # message(${PYPATH})
        # message(${DECOMMPRESS_PATH})
        # message(${PYTHON_EXECUTABLE})
        # message(${DECOMMPRESS_PATH}/${LOCAL_ORT_LIB_ZIP})
        # message(${LOCAL_ORT_LIB_ZIP_DL_DIR})
        if(CMAKE_SYSTEM_NAME MATCHES "Linux")
            # message(${ORTPACKAGE_ROOT})
            execute_process(
                COMMAND mkdir -p ${LIBRARY_DIR} WORKING_DIRECTORY ${SOLUTION_DIR}
                COMMAND tar xfv ${DECOMMPRESS_PATH} --strip-components 1 
                        -C ${LIBRARY_DIR} WORKING_DIRECTORY ${SOLUTION_DIR}            
	    )
	    file(GLOB LIB_ALL ${LIBRARY_DIR}/lib/libonnxruntime*)
	    file(COPY ${LIB_ALL} DESTINATION ${LIBRARY_DIR})
	    execute_process(
		COMMAND rm -rf ${LIBRARY_DIR}/lib WORKING_DIRECTORY ${SOLUTION_DIR}
	    )
            set(SOURCE_FILE
                ${ORTPACKAGE_ROOT}/libonnxruntime.so)
            set(TARGET_FILE
                ${ORTPACKAGE_ROOT}/libonnxruntime.so.1.12.1)
        elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
            execute_process(
                COMMAND mkdir -p ${LIBRARY_DIR}/onnxruntime WORKING_DIRECTORY ${SOLUTION_DIR}
                COMMAND bsdtar xfv ${DECOMMPRESS_PATH} --strip-components 1 
                        -C ${LIBRARY_DIR}/onnxruntime WORKING_DIRECTORY ${SOLUTION_DIR}
                )         
            set(SOURCE_FILE
                ${ORTPACKAGE_ROOT}/libonnxruntime.so)
            set(TARGET_FILE
                ${ORTPACKAGE_ROOT}/libonnxruntime.so.1.12.1)
        endif()

        # configure_file(${SOURCE_FILE} ${TARGET_FILE} COPYONLY)

        # if(py_result MATCHES 0)
        #     set(ORT_LIB_FOUND
        #         1
        #         CACHE INTERNAL "")
        # else()
        #     message(
        #         STATUS "Failed to extract files.\n"
        #                "   Please try downloading and extracting yourself.\n"
        #                "   The url is: ${ORT_LIB_DOWNLOAD_URL}")
        # endif()
    endif()
endif()

