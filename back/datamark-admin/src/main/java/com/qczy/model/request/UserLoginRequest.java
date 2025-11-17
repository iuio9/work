package com.qczy.model.request;

import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

import javax.validation.constraints.NotBlank;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/7/15 16:01
 * @Description:
 */
@Data
public class UserLoginRequest {

    @ApiModelProperty(name = "用户名")
    @NotBlank(message = "用户名不能为空")
    private String userName;

    @ApiModelProperty(name = "密码")
    @NotBlank(message = "密码不能为空")
    private String password;
}
