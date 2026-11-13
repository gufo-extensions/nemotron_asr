# Nemotron 3.5 ASR (NVIDIA) speech recognition extension.

# Host side: configuration, probe, tokenizer, front end and the CPU-only
# registration.
gufo_extension_target(
  NAME gufo_nemotron_asr
  SOURCES
    config.cpp
    language.cpp
    tokenizer.cpp
    features.cpp
    resampler.cpp
    safetensors.cpp
    probe.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
)
if(TARGET gufo_nemotron_asr)
  target_include_directories(gufo_nemotron_asr PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
  if(ENGINE_ENABLE_HIP)
    target_compile_definitions(gufo_nemotron_asr PRIVATE ENGINE_ENABLE_HIP=1)
  endif()
endif()

# Device side: weights, encoder and transducer.
gufo_extension_target(
  NAME gufo_nemotron_asr_hip
  SOURCES
    model.cpp
  LINK_LIBRARIES
    gufo_nemotron_asr
    gufo_ext_runtime
    gufo_ext_hip
    hip::host
    roc::hipblas
)
if(TARGET gufo_nemotron_asr_hip)
  gufo_extension_hip_sources(gufo_nemotron_asr_hip
    SOURCES
      kernels/kernels.hip
      kernels/wmma.hip
    OPTIONS
      -O3 -Wno-unused-command-line-argument
  )
  target_include_directories(gufo_nemotron_asr_hip PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
endif()

# Speech recognition provider: the AsrRunner and its registration.
gufo_extension_target(
  NAME gufo_nemotron_asr_runner
  SOURCES
    runner.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_nemotron_asr_hip
    gufo_nemotron_asr
    gufo_ext_runtime
    gufo_ext_hip
)

# gufo_ext_runtime only declares the public services; these link their
# implementations into executables that stand apart from `gufo`.
set(_gufo_nemotron_asr_test_libs gufo_ext_core gufo_ext_cache)
if(ENGINE_ENABLE_HIP)
  list(APPEND _gufo_nemotron_asr_test_libs gufo_ext_hip)
endif()
set(_gufo_nemotron_asr_fixtures "${GUFO_EXTENSION_BUILD_ROOT}/tests/fixtures")

gufo_extension_target(
  NAME nemotron_asr_host_test
  SOURCES
    tests/host_test.cpp
  LINK_LIBRARIES
    gufo_nemotron_asr
    gufo_ext_runtime
    ${_gufo_nemotron_asr_test_libs}
)
if(TARGET nemotron_asr_host_test)
  add_test(NAME nemotron_asr.host COMMAND nemotron_asr_host_test samples)
  set_tests_properties(nemotron_asr.host PROPERTIES LABELS "nemotron_asr;cpu")
  add_test(NAME nemotron_asr.host_model COMMAND nemotron_asr_host_test model)
  set_tests_properties(nemotron_asr.host_model PROPERTIES
    LABELS "nemotron_asr;cpu;external-model"
    SKIP_RETURN_CODE 77)
endif()

gufo_extension_target(
  NAME nemotron_asr_features_test
  SOURCES
    tests/features_test.cpp
  LINK_LIBRARIES
    gufo_nemotron_asr
    gufo_ext_runtime
    ${_gufo_nemotron_asr_test_libs}
)
if(TARGET nemotron_asr_features_test)
  target_compile_definitions(nemotron_asr_features_test PRIVATE
    GUFO_NEMOTRON_ASR_FIXTURES="${_gufo_nemotron_asr_fixtures}")
  add_test(NAME nemotron_asr.features COMMAND nemotron_asr_features_test)
  set_tests_properties(nemotron_asr.features PROPERTIES
    LABELS "nemotron_asr;cpu")
endif()

gufo_extension_target(
  NAME nemotron_asr_tokenizer_test
  SOURCES
    tests/tokenizer_test.cpp
  LINK_LIBRARIES
    gufo_nemotron_asr
    gufo_ext_runtime
    ${_gufo_nemotron_asr_test_libs}
)
if(TARGET nemotron_asr_tokenizer_test)
  target_compile_definitions(nemotron_asr_tokenizer_test PRIVATE
    GUFO_NEMOTRON_ASR_FIXTURES="${_gufo_nemotron_asr_fixtures}")
  add_test(NAME nemotron_asr.tokenizer COMMAND nemotron_asr_tokenizer_test)
  set_tests_properties(nemotron_asr.tokenizer PROPERTIES
    LABELS "nemotron_asr;cpu;external-model"
    SKIP_RETURN_CODE 77)
endif()

gufo_extension_target(
  NAME nemotron_asr_model_test
  SOURCES
    tests/model_test.cpp
  LINK_LIBRARIES
    gufo_nemotron_asr_runner
    gufo_nemotron_asr_hip
    gufo_nemotron_asr
    gufo_ext_runtime
    ${_gufo_nemotron_asr_test_libs}
)
if(TARGET nemotron_asr_model_test)
  target_compile_definitions(nemotron_asr_model_test PRIVATE
    GUFO_NEMOTRON_ASR_FIXTURES="${_gufo_nemotron_asr_fixtures}")
  add_test(NAME nemotron_asr.encoder COMMAND nemotron_asr_model_test encoder)
  set_tests_properties(nemotron_asr.encoder PROPERTIES
    LABELS "nemotron_asr;gpu;external-model"
    SKIP_RETURN_CODE 77)
  add_test(NAME nemotron_asr.transcription COMMAND nemotron_asr_model_test transcription)
  set_tests_properties(nemotron_asr.transcription PROPERTIES
    LABELS "nemotron_asr;gpu;external-model"
    SKIP_RETURN_CODE 77)
  add_test(NAME nemotron_asr.runner COMMAND nemotron_asr_model_test runner)
  set_tests_properties(nemotron_asr.runner PROPERTIES
    LABELS "nemotron_asr;gpu;external-model"
    SKIP_RETURN_CODE 77)
endif()

gufo_extension_target(
  NAME nemotron_asr_transcribe
  SOURCES
    tools/transcribe/transcribe.cpp
  LINK_LIBRARIES
    gufo_nemotron_asr_runner
    gufo_nemotron_asr_hip
    gufo_nemotron_asr
    gufo_ext_runtime
    ${_gufo_nemotron_asr_test_libs}
)
