package com.qczy.service;

import com.qczy.model.entity.AlgorithmTaskEntity;
import com.qczy.model.entity.MarkInfoEntity;
import io.swagger.models.auth.In;

import java.util.HashMap;
import java.util.Map;

/**
 * @author ：gwj
 * @date ：Created in 2024-08-27 15:30
 * @description：
 * @modified By：
 * @version: $
 */
public interface DefectGenerationService {
    String startDefectGen(AlgorithmTaskEntity algorithmTaskEntity);

}
