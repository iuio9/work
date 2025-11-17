package com.qczy.model.entity.domain;

import com.baomidou.mybatisplus.annotation.TableName;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/7/21 14:13
 * @Description:
 */
@Data
@TableName("qczy_model_mark_info")
public class ModelMarkInfoEntity {

    /** id */
    @ApiModelProperty(value = "id")
    private Integer id;


    /** 任务id */
    @ApiModelProperty(value = "任务id")
    private Integer taskId;

    /** 数据集id */
    @ApiModelProperty(value = "数据集id")
    private String sonId;


    /** 文件id */
    @ApiModelProperty(value = "数据集id")
    private Integer fileId;

    /** 标注信息 */
    @ApiModelProperty(value = "标注信息")
    private String markInfo;



}
