package com.qczy.model.response;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/9 14:39
 * @Description:
 */
@Data
public class DataImportLogResponse {

    // id
    private int id;

    // 文件大小
    private String fileSize;

    // 数据量
    private Integer count;

    // 创建人
    private String nickName;

    //导入开始时间
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    private Date importStartTime;

    //导入完成时间
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    private Date importEndTime;

    //导入状态
    private Integer status;
}
