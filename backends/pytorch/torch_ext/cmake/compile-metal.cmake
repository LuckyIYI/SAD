function(compile_metal_shaders TARGET_NAME METAL_SOURCES EXTRA_INCLUDE_DIRS)
    if(NOT DEFINED METAL_TOOLCHAIN)
      execute_process(
        COMMAND "xcodebuild" "-showComponent" "MetalToolchain"
        OUTPUT_VARIABLE FIND_METAL_OUT
        RESULT_VARIABLE FIND_METAL_ERROR_CODE
        ERROR_VARIABLE FIND_METAL_STDERR
        OUTPUT_STRIP_TRAILING_WHITESPACE)

      if(NOT FIND_METAL_ERROR_CODE EQUAL 0)
        message(FATAL_ERROR "Failed to locate Metal toolchain: ${FIND_METAL_STDERR}")
      endif()

      string(REGEX MATCH "Toolchain Search Path: ([^\n]+)" MATCH_RESULT "${FIND_METAL_OUT}")
      set(METAL_TOOLCHAIN "${CMAKE_MATCH_1}/Metal.xctoolchain")
    endif()

    set(METAL_FLAGS "-std=metal3.1" "-O2")

    set(METALLIB_OUTPUT_DIR "${CMAKE_BINARY_DIR}/metallib")
    file(MAKE_DIRECTORY ${METALLIB_OUTPUT_DIR})

    foreach(INC ${EXTRA_INCLUDE_DIRS})
        list(APPEND METAL_FLAGS "-I${INC}")
    endforeach()

    set(AIR_FILES)
    set(METAL_FILES)
    set(HEADER_FILES)

    foreach(SOURCE_FILE ${METAL_SOURCES})
        if(SOURCE_FILE MATCHES "\\.metal$")
            list(APPEND METAL_FILES ${SOURCE_FILE})
        elseif(SOURCE_FILE MATCHES "\\.h$")
            list(APPEND HEADER_FILES ${SOURCE_FILE})
        endif()
    endforeach()

    foreach(METAL_FILE ${METAL_FILES})
        get_filename_component(METAL_NAME ${METAL_FILE} NAME_WE)
        set(AIR_FILE "${CMAKE_BINARY_DIR}/${METAL_NAME}.air")

        set(ALL_DEPENDENCIES ${METAL_FILE})
        foreach(HEADER_FILE ${HEADER_FILES})
            list(APPEND ALL_DEPENDENCIES ${HEADER_FILE})
        endforeach()

        add_custom_command(
            OUTPUT ${AIR_FILE}
            COMMAND "${METAL_TOOLCHAIN}/usr/bin/metal" ${METAL_FLAGS}
                    -c ${METAL_FILE}
                    -o ${AIR_FILE}
            DEPENDS ${ALL_DEPENDENCIES}
            COMMENT "Compiling Metal shader ${METAL_FILE} to ${AIR_FILE}"
            VERBATIM
        )

        list(APPEND AIR_FILES ${AIR_FILE})
    endforeach()

    set(METALLIB_FILE "${METALLIB_OUTPUT_DIR}/${TARGET_NAME}.metallib")
    add_custom_command(
        OUTPUT ${METALLIB_FILE}
        COMMAND "${METAL_TOOLCHAIN}/usr/bin/metallib" ${AIR_FILES}
                -o ${METALLIB_FILE}
        DEPENDS ${AIR_FILES}
        COMMENT "Linking Metal library ${METALLIB_FILE}"
        VERBATIM
    )

    set(METALLIB_HEADER "${CMAKE_BINARY_DIR}/${TARGET_NAME}_metallib.h")
    set(METALLIB_TO_HEADER_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/metallib_to_header.py")

    add_custom_command(
        OUTPUT ${METALLIB_HEADER}
        COMMAND ${Python3_EXECUTABLE} ${METALLIB_TO_HEADER_SCRIPT} ${METALLIB_FILE} ${METALLIB_HEADER} ${TARGET_NAME}
        DEPENDS ${METALLIB_FILE} ${METALLIB_TO_HEADER_SCRIPT}
        COMMENT "Generating embedded Metal library header ${METALLIB_HEADER}"
        VERBATIM
    )

    add_custom_target(${TARGET_NAME}_metallib ALL DEPENDS ${METALLIB_FILE} ${METALLIB_HEADER})
    add_dependencies(${TARGET_NAME} ${TARGET_NAME}_metallib)

    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_BINARY_DIR})
    target_compile_definitions(${TARGET_NAME} PRIVATE
        EMBEDDED_METALLIB_HEADER="${TARGET_NAME}_metallib.h"
        EMBEDDED_METALLIB_NAMESPACE=${TARGET_NAME}_metal
    )
endfunction()
