package com.qczy.model.request;

import lombok.Data;

import javax.validation.constraints.NotBlank;
import javax.validation.constraints.Size;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/3/31 15:08
 * @Description:
 */
@Data
public class PasswordRequest {

    // 用户id
    private Integer userId;

    // 密码
    @NotBlank(message = "密码不能为空")
    @Size(min = 1, max = 20, message = "密码长度必须在 1 到 20 个字符之间")
    private String password;

    // 确认密码
    @NotBlank(message = "确认密码不能为空")
    @Size(min = 1, max = 20, message = "确认密码长度必须在 1 到 20 个字符之间")
    private String confirmPassword;


}
