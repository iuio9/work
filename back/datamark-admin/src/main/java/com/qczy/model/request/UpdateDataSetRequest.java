package com.qczy.model.request;

import lombok.Data;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/14 9:49
 * @Description:
 */
@Data
public class UpdateDataSetRequest {

    // 数据集组id
    private String groupId;

    // 数据集id
    private String sonId;

    // 数据集组名称
    private String groupName;

    // 备注
    private String remark;


}
