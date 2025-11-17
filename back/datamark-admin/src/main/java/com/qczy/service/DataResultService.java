package com.qczy.service;

import com.qczy.model.request.SavaResultRequest;

import java.util.Map;

public interface DataResultService {

    Map<String, Object> savaResult(SavaResultRequest request);
}
