execute_process(
  COMMAND "${SOMA_OPT}" --verify-each "${INPUT}"
  RESULT_VARIABLE result
  ERROR_VARIABLE diagnostics
  OUTPUT_QUIET)

if(result EQUAL 0)
  message(FATAL_ERROR "${INPUT} unexpectedly passed verification")
endif()

string(FIND "${diagnostics}" "${EXPECTED}" expected_offset)
if(expected_offset EQUAL -1)
  message(FATAL_ERROR "missing expected diagnostic '${EXPECTED}':\n${diagnostics}")
endif()
