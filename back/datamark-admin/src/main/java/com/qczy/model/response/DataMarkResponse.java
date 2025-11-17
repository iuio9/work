package com.qczy.model.response;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.Data;
import org.jpedal.parser.shape.S;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/22 20:07
 * @Description: 在线标注（列表） 返回类
 */
@Data
public class DataMarkResponse {

    // 数据集id
    private String sonId;

    // 数据集名称
    private String groupName;

    // 版本
    private String version;

    private Integer dataTypeId;
    private String dataTypeName;

    // 数据量
    private Integer count;

    // 标注进度
    private String status;

    // 创建时间
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    private Date createTime;

    private Integer anoType;

    private Integer isMany;


}
