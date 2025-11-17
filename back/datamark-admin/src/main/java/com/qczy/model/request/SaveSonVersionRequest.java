package com.qczy.model.request;

import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/9 16:44
 * @Description:
 */
@Data
public class SaveSonVersionRequest {

    @ApiModelProperty(value = "数据集组id")
    private String groupId;

    @ApiModelProperty(value = "历史版本")
    private Integer version;

    @ApiModelProperty(value = "新版本")
    private Integer newVersion;

    @ApiModelProperty(value = "数据类型")
    private Integer dataType;

    @ApiModelProperty(value = "标注类型")
    private Integer markType;

    @ApiModelProperty(value = "标注模板")
    private Integer markTemp;

    @ApiModelProperty(value = "备注信息")
    private String remark;

    @ApiModelProperty(value = "是否继承->   0：不继承   1：继承")
    private Integer isInherit;


}
