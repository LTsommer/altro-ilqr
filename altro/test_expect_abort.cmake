execute_process(COMMAND "${TEST_PROGRAM}" "${TEST_CASE}"
                RESULT_VARIABLE result
                ERROR_VARIABLE diagnostic)
if(NOT result MATCHES "[Aa]bort")
    message(FATAL_ERROR "${TEST_CASE}: expected process abort, got ${result}\n${diagnostic}")
endif()
