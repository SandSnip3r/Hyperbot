project(ui-proto LANGUAGES CXX)

find_package(Protobuf REQUIRED)

include_directories(${Protobuf_INCLUDE_DIRS})

# Root for generated files
set(GENERATED_PROTO_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
# Put headers in a subfolder called "ui-proto"
file(MAKE_DIRECTORY "${GENERATED_PROTO_DIR}/ui-proto")

set(PROTO_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/broadcast.proto"
  "${CMAKE_CURRENT_SOURCE_DIR}/character_config.proto"
  "${CMAKE_CURRENT_SOURCE_DIR}/entity.proto"
  "${CMAKE_CURRENT_SOURCE_DIR}/old_config.proto"
  "${CMAKE_CURRENT_SOURCE_DIR}/position.proto"
  "${CMAKE_CURRENT_SOURCE_DIR}/request.proto"
  "${CMAKE_CURRENT_SOURCE_DIR}/server_config.proto"
  "${CMAKE_CURRENT_SOURCE_DIR}/stats.proto"
)

protobuf_generate_cpp(
  PROTO_SRCS
  PROTO_HEADER
  ${PROTO_FILES}
  PROTOC_OUT_DIR "${GENERATED_PROTO_DIR}/ui-proto"
)

add_library(ui-proto STATIC
  ${PROTO_SRCS}
  ${PROTO_HEADER}
)

target_include_directories(ui-proto
  PUBLIC
    "${GENERATED_PROTO_DIR}"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(ui-proto
  PUBLIC
    ${Protobuf_LIBRARIES}
)