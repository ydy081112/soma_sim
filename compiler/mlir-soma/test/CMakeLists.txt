add_test(
  NAME snnop-param-refs
  COMMAND $<TARGET_FILE:soma-opt> --verify-each
          ${CMAKE_CURRENT_SOURCE_DIR}/param_refs.mlir)

foreach(case missing wrong_kind affine_shape)
  add_test(
    NAME snnop-param-${case}
    COMMAND ${CMAKE_COMMAND}
            -DSOMA_OPT=$<TARGET_FILE:soma-opt>
            -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/param_${case}.mlir
            -DEXPECTED=${case}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_param_failure.cmake)
endforeach()

add_test(
  NAME snnop-model-neuron-fusion
  COMMAND $<TARGET_FILE:soma-opt> --verify-each --model-neuron-fusion
          ${CMAKE_CURRENT_SOURCE_DIR}/model_neuron_fusion.mlir)

add_test(
  NAME snnop-dead-neuron-out-eliminate
  COMMAND $<TARGET_FILE:soma-opt> --verify-each --model-neuron-fusion
          --dead-neuron-out-eliminate
          ${CMAKE_CURRENT_SOURCE_DIR}/model_neuron_fusion.mlir)

add_test(
  NAME snnexec-parse-print-verify
  COMMAND $<TARGET_FILE:soma-opt> --verify-each
          ${CMAKE_CURRENT_SOURCE_DIR}/snnexec_ops.mlir)

add_test(
  NAME snnexec-lower-fused-snnop
  COMMAND $<TARGET_FILE:soma-opt> --verify-each --model-neuron-fusion
          --dead-neuron-out-eliminate --lower-snnop-to-snnexec
          ${CMAKE_CURRENT_SOURCE_DIR}/model_neuron_fusion.mlir)

add_test(
  NAME snnexec-missing-weight
  COMMAND ${CMAKE_COMMAND}
          -DSOMA_OPT=$<TARGET_FILE:soma-opt>
          -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/snnexec_missing_weight.mlir
          -DEXPECTED=missing
          -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_param_failure.cmake)

add_test(
  NAME noc-snnarch-parse-print-verify
  COMMAND $<TARGET_FILE:soma-opt> --verify-each
          ${CMAKE_CURRENT_SOURCE_DIR}/noc_snnarch_valid.mlir)

add_test(
  NAME noc-snnarch-invalid-dimensions
  COMMAND ${CMAKE_COMMAND}
          -DSOMA_OPT=$<TARGET_FILE:soma-opt>
          -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/noc_snnarch_invalid.mlir
          -DEXPECTED=dimensions
          -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake)

add_test(
  NAME noc-invalid-routing
  COMMAND ${CMAKE_COMMAND}
          -DSOMA_OPT=$<TARGET_FILE:soma-opt>
          -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/noc_invalid_routing.mlir
          -DEXPECTED=routing
          -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake)

add_test(
  NAME snnarch-invalid-neuron-capacity
  COMMAND ${CMAKE_COMMAND}
          -DSOMA_OPT=$<TARGET_FILE:soma-opt>
          -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/snnarch_invalid_capacity.mlir
          -DEXPECTED=neuron_capacity
          -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake)

add_test(
  NAME snnop-core-mapping
  COMMAND ${CMAKE_COMMAND}
          -DSOMA_OPT=$<TARGET_FILE:soma-opt>
          -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/core_mapping.mlir
          -P ${CMAKE_CURRENT_SOURCE_DIR}/verify_core_mapping.cmake)

add_test(
  NAME snnop-core-mapping-mesh-overflow
  COMMAND ${CMAKE_COMMAND}
          -DSOMA_OPT=$<TARGET_FILE:soma-opt>
          -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/core_mapping_overflow.mlir
          -DEXPECTED=more
          -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_core_mapping_failure.cmake)

add_test(
  NAME snnop-core-mapping-attention-residual
  COMMAND ${CMAKE_COMMAND}
          -DSOMA_OPT=$<TARGET_FILE:soma-opt>
          -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/core_mapping_attention.mlir
          -P ${CMAKE_CURRENT_SOURCE_DIR}/verify_core_attention.cmake)

set(SOMA_MLIR_PYTHON "/home/ydy/miniconda3/envs/sim_snn/bin/python3"
    CACHE FILEPATH "Python interpreter with PyYAML for MLIR hardware frontend tests")
if(EXISTS "${SOMA_MLIR_PYTHON}")
  add_test(
    NAME hardware-yaml-to-core-level
    COMMAND ${CMAKE_COMMAND}
            -DPYTHON=${SOMA_MLIR_PYTHON}
            -DIMPORTER=${PROJECT_SOURCE_DIR}/tools/hardware-yaml-to-arch.py
            -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/hardware_core_level.yaml
            -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/hardware_core_level.mlir
            -DSOMA_OPT=$<TARGET_FILE:soma-opt>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/verify_hardware_yaml.cmake)
endif()
