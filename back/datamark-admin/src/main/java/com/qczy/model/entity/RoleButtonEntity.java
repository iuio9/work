package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/5/29 12:37
 * @Description:
 */
@TableName("qczy_role_button")
@Data
public class RoleButtonEntity {


    private Integer roleId;

    private Integer buttonId;

}
