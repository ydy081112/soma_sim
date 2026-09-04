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
