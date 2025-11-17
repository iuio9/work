package com.qczy.model.request;

import lombok.Data;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/22 14:40
 * @Description:
 */
@Data
public class AddDataSetLabelRequest {

    // 数据集id
    private String sonId;

    // 数据集级id
    private String labelGroupId;


}
