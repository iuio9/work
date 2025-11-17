package com.qczy.service;

import com.qczy.model.entity.AlgorithmTaskEntity;
import io.swagger.models.auth.In;
import org.springframework.stereotype.Service;

import java.util.Map;

/**
 * @author ：gwj
 * @date ：Created in 2025-05-29 13:44
 * @description：
 * @modified By：
 * @version: $
 */
public interface ThirdModelAssessService {
    void uploadJsonData(AlgorithmTaskEntity algorithmTaskEntity);
    void uploadClassifyJsonData(AlgorithmTaskEntity algorithmTaskEntity);
    Map viewAssessResult(AlgorithmTaskEntity algorithmTaskEntity);
    void controlTask(Long taskId, Integer status);

}
