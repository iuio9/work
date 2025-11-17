package com.qczy.model.request;

import com.baomidou.mybatisplus.annotation.TableField;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/7/29 9:55
 * @Description:
 */
@Data
public class UserRequest {

    private String  userName;

    private String  nickName;

    private Integer userGender;

    private String  userEmail;

    private String userPhone;

    private Integer status;

    private Integer deptId;





}
