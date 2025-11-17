package com.qczy.model.response;

import lombok.Data;
import org.jpedal.parser.shape.S;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/2/26 14:14
 * @Description:
 */
@Data
public class UserListResponse {

    private int userId;
    private String userName;
    private String nickName;

}
