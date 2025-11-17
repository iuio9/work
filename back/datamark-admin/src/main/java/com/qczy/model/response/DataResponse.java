package com.qczy.model.response;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;
import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/8 15:13
 * @Description:
 */
@Data
public class DataResponse {


    // 第一层结构


    // 数据集组id
    private String groupId;
    // 数据集组名称
    private String groupName;

    // 创建时间
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    private Date createTime;


    // 第二层结构
    List <DataSonResponse> dataSonResponseList;








}
