package com.qczy.model.request;

import lombok.Data;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/14 10:03
 * @Description:
 */
@Data
public class DataSonQueryRequest {

    private String groupName;

    private Integer dataTypeId;

    // 字典的ids
    private String dataTypeIds;

}
