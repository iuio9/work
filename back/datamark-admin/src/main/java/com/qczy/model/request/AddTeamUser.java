package com.qczy.model.request;

import lombok.Data;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/2/26 14:38
 * @Description:
 */
@Data
public class AddTeamUser {
    // 用户id
    private Integer userId;
    // 备注
    private String remark;

}
