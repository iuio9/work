package com.qczy.model.response;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/8 15:20
 * @Description:
 */
@Data
public class DataSonResponse {


    // 版本
    private Integer version;

    // 数据集id
    private String sonId;

    // 数据量
    private Integer count;

    // 数据类型
    private String dataType;
    // 数据类型id
    private String dataTypeId;

    // 标注类型
    private String markType;
    // 标注类型id
    private Integer markTypeId;

    // 最近导入状态
    private Integer importStatus;

    // 模板
    private String markTemp;
    // 模板id
    private String markTempId;

    // 标注进度
    private String status;

    // 创建时间
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    private Date createTime;

    // 备注
    private String remark;

    // 导入总数
    private Integer dataImportCount;

    // 文件id集合
    private String fileIds;

    // 进度
    private Integer progress;

    private Integer isSocket;

    private String dataTypeName;

    private Integer isMany;


    /**
     * 标注类型
     */
    @ApiModelProperty(value = "标注类型")
    private Integer anoType;

    /**
     *  标签类型   group：标签组 、 single：标签
     */
    private String tagSelectionMode;

    // 文件总数量
    private Integer fileSumCount;

}
