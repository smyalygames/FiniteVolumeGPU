from hip import hip


def hip_check(call_request):
    """
    Function that checks if the HIP function executed successfully.
    """

    err = call_request[0]
    result = call_request[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result
