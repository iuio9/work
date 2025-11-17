package com.qczy.model.request;

import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

import javax.validation.constraints.NotBlank;


/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/12/30 9:57
 * @Description:
 */
@Data
public class CopyLabelGroupRequest {

    @ApiModelProperty(value = "数据集id")
    @NotBlank(message = "数据集id不能为空")
    private String sonId;


    @ApiModelProperty(value = "需要复制的数据集id")
    @NotBlank(message = "需要复制的数据集id不能为空")
    private String copySonId;


    @ApiModelProperty(value = "业务类型-> 1:全部覆盖 2：数据叠加")
    private Integer businessType;


}
