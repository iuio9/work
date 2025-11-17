package com.qczy.model.response;

import lombok.Data;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/3/3 10:20
 * @Description:
 */
@Data
public class TeamUserResponse {

    /**
     *  自增id
     */
    private Integer id;

    /**
     *  用户id
     */
    private Integer userId;

    /**
     *  用户昵称
     */
    private String nickName;



}
