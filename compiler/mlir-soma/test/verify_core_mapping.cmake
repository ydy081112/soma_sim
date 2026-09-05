execute_process(
  COMMAND "${SOMA_OPT}" --verify-each --snnop-core-mapping "${INPUT}"
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "core mapping failed: ${error}")
endif()
foreach(required
    "neuron_model = \"st_bif\""
    "snn_arch.q_core"
    "snn_arch.fc_core"
    "partition_id = 1"
    "partition_offset = 4"
    "partition_size = 2"
    "core_id = 1"
    "coord = array<i64: 1, 0>"
    "source_partition = array<i64: 0, 1>"
    "source_offset = array<i64: 0, 4>"
    "source_size = array<i64: 4, 2>"
    "noc.send_router"
    "noc.recv_router")
  string(FIND "${output}" "${required}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "missing '${required}' in mapped IR")
  endif()
endforeach()
string(REGEX MATCHALL "noc\\.send_router" sends "${output}")
string(REGEX MATCHALL "noc\\.recv_router" recvs "${output}")
list(LENGTH sends send_count)
list(LENGTH recvs recv_count)
if(NOT send_count EQUAL 1 OR NOT recv_count EQUAL 1)
  message(FATAL_ERROR "expected exactly one cross-core pair, got send=${send_count}, recv=${recv_count}")
endif()
