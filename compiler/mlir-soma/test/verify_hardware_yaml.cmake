execute_process(
  COMMAND "${PYTHON}" "${IMPORTER}" --input "${INPUT}" --output "${OUTPUT}"
  RESULT_VARIABLE import_result
  ERROR_VARIABLE import_diagnostics)
if(NOT import_result EQUAL 0)
  message(FATAL_ERROR "hardware YAML import failed:\n${import_diagnostics}")
endif()

execute_process(
  COMMAND "${SOMA_OPT}" --verify-each "${OUTPUT}"
  RESULT_VARIABLE verify_result
  ERROR_VARIABLE verify_diagnostics
  OUTPUT_QUIET)
if(NOT verify_result EQUAL 0)
  message(FATAL_ERROR "generated architecture IR failed verification:\n${verify_diagnostics}")
endif()
