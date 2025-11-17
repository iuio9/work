package com.qczy.model.request;

import lombok.Data;

import java.util.HashMap;
import java.util.Map;

/**
 * @author ：gwj
 * @date ：Created in 2024-08-27 16:49
 * @description：
 * @modified By：
 * @version: $
 */
@Data
public class AlgorithmParams {
    private String image_path;
    private String markInfo;
    private String algorithmId;
    private Map<String,Object> params;
    private String imageAbsoute;
}
