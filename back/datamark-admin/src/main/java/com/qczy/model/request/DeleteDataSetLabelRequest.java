package com.qczy.model.request;

import lombok.Data;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/22 16:28
 * @Description:
 */
@Data
public class DeleteDataSetLabelRequest {

    //数据集id
    private String sonId;

    //标签id
    private Integer labelId;

    // 多个标签
    private String labelIds;


}
