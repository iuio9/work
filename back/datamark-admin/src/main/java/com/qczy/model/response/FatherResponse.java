package com.qczy.model.response;

import lombok.Data;

import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/8 16:27
 * @Description:  返回当前用户所能看到的数据集组
 */
@Data
public class FatherResponse {

    // 数据集组id
    private String groupId;

    // 数据集组名称
    private String groupName;

    // 创建时间
    private Date createTime;




}
