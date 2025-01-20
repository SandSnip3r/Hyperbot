project(ui_proto LANGUAGES CXX)

find_package(Protobuf REQUIRED)

# include_directories(${Protobuf_INCLUDE_DIRS})

# Root for generated files
set(GENERATED_PROTO_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
# Put headers in a subfolder
file(MAKE_DIRECTORY "${GENERATED_PROTO_DIR}/${PROJECT_NAME}")

set(PROTO_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/broadcast.proto"
  "${CMAKE_CURRENT_SOURCE_DIR}/character_config.proto"
  "${CMAKE_CURRENT_SOURCE_DIR}/client_manager_request.proto"
  "${CMAKE_CURRENT_SOURCE_DIR}/entity.proto"
  "${CMAKE_CURRENT_SOURCE_DIR}/old_config.proto"
  "${CMAKE_CURRENT_SOURCE_DIR}/position.proto"
  "${CMAKE_CURRENT_SOURCE_DIR}/request.proto"
  "${CMAKE_CURRENT_SOURCE_DIR}/rl_ui_config.proto"
  "${CMAKE_CURRENT_SOURCE_DIR}/server_config.proto"
  "${CMAKE_CURRENT_SOURCE_DIR}/stats.proto"
)

set(PROTOBUF_EXTRA_FLAGS "--experimental_allow_proto3_optional")

protobuf_generate_cpp(
  PROTO_SRCS
  PROTO_HEADER
  ${PROTO_FILES}
  PROTOC_OUT_DIR "${GENERATED_PROTO_DIR}/${PROJECT_NAME}"
  PROTOC_OPTIONS "${PROTOBUF_EXTRA_FLAGS}"
)

add_library(${PROJECT_NAME} STATIC
  ${PROTO_SRCS}
  ${PROTO_HEADER}
)

target_include_directories(${PROJECT_NAME}
PUBLIC
  "${GENERATED_PROTO_DIR}"
  ${Protobuf_INCLUDE_DIRS}  # Add Protobuf includes as PUBLIC
PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(${PROJECT_NAME}
PUBLIC
  protobuf::libprotobuf
)