execute_process(
  COMMAND "${SOMA_OPT}" --verify-each "${INPUT}"
  RESULT_VARIABLE result
  ERROR_VARIABLE diagnostics
  OUTPUT_QUIET)

if(result EQUAL 0)
  message(FATAL_ERROR "${INPUT} unexpectedly passed verification")
endif()

if(EXPECTED STREQUAL "missing")
  set(expected_text "references unknown snn_op.param")
elseif(EXPECTED STREQUAL "wrong_kind")
  set(expected_text "must reference kind=\"weight\"")
elseif(EXPECTED STREQUAL "affine_shape")
  set(expected_text "requires matching tensor shapes")
else()
  message(FATAL_ERROR "unknown expected failure case: ${EXPECTED}")
endif()

string(FIND "${diagnostics}" "${expected_text}" expected_offset)
if(expected_offset EQUAL -1)
  message(FATAL_ERROR "missing expected diagnostic '${expected_text}':\n${diagnostics}")
endif()
