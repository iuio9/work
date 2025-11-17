package com.qczy.model.request;

import lombok.Data;
import org.jpedal.parser.shape.S;

import java.util.List;
import java.util.Map;

/**
 * @author ：gwj
 * @date ：Created in 2024-09-04 16:32
 * @description：训练参数
 * @modified By：
 * @version: $
 */
@Data
public class TrainEntity {
    private String modelId;
    private String datasetId;
    private String datasetOutId;
    private String algorithmId;
    private Map<String,Object> algorithmParam;
    private String taskInputName;
    private String taskDesc;

}
